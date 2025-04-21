import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("cpu")

class HybridPINN(nn.Module):
    def __init__(self):
        super(HybridPINN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        ).to(device)
    
    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        return self.layer(inputs)

def physics_loss(model, t, x, alpha=1.0):
    u = model(t, x)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]
    f = (u_t - alpha * u_xx).pow(2).mean()
    return f

def boundary_loss(model, t_bc, x_left, x_right):
    u_left = model(t_bc, x_left)
    u_right = model(t_bc, x_right)
    loss_left = (u_left - g1(t_bc)).pow(2).mean()
    loss_right = (u_right - g2(t_bc)).pow(2).mean()
    return loss_left + loss_right

def g1(t):
    return torch.zeros_like(t)

def g2(t):
    return torch.zeros_like(t)

def initial_loss(model, x_ic):
    t_0 = torch.zeros_like(x_ic).to(device)
    u_init = model(t_0, x_ic)
    u_exact = f(x_ic)
    return (u_init - u_exact).pow(2).mean()

def f(x):
    return torch.sin(np.pi * x)

def analytical_solution(t, x, alpha=1.0, n_terms=50):
    u = torch.zeros_like(x)
    for n in range(1, n_terms + 1):
        lambda_n = (n * np.pi) ** 2
        u += (2 / (n * np.pi)) * torch.sin(n * np.pi * x) * torch.exp(-alpha * lambda_n * t)
    return u

def calculate_error(model, t, x):
    with torch.no_grad():
        u_pred = model(t, x)
        u_exact = analytical_solution(t, x)
        return torch.mean((u_pred - u_exact).abs()).item()

def train(model, optimizer, num_epochs, lr=1e-3, alpha=1.0):
    losses = []
    errors = []

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.to(device)
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()

        t = torch.rand(3000, 1).to(device)
        x = torch.rand(3000, 1).to(device) * 2 - 1
        t.requires_grad = True
        x.requires_grad = True

        f_loss = physics_loss(model, t, x, alpha)

        t_bc = torch.rand(500, 1).to(device)
        x_left = -torch.ones(500, 1).to(device)
        x_right = torch.ones(500, 1).to(device)
        bc_loss = boundary_loss(model, t_bc, x_left, x_right)

        x_ic = torch.rand(1000, 1).to(device) * 2 - 1
        ic_loss = initial_loss(model, x_ic)

        loss = 1.0 * f_loss + 0.5 * bc_loss + 0.5 * ic_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        errors.append(calculate_error(model, t, x))

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Total Loss: {loss.item()}, Error: {errors[-1]}')

    return losses, errors

def parameter_exploration():
    learning_rates = [1e-2, 1e-3, 1e-4]
    alphas = [0.5, 1.0, 2.0]
    num_epochs = 5000

    results = []

    for lr in learning_rates:
        for alpha in alphas:
            print(f"\nTraining with lr={lr}, alpha={alpha}")
            model = HybridPINN()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            losses, errors = train(model, optimizer, num_epochs, lr, alpha)
            final_error = errors[-1]
            results.append((lr, alpha, final_error))
            print(f"Final error: {final_error}")

    print("\nParameter exploration results:")
    for lr, alpha, error in results:
        print(f"lr={lr}, alpha={alpha}, final error={error}")

def generate_1d_grid(L=1.0, N=100):
    x = torch.linspace(-L, L, N).reshape(-1, 1).to(device)
    return x

def build_local_fit_matrix(x, poly_order=2):
    N = x.shape[0]
    A = torch.zeros(N, N).to(device)

    for i in range(N):
        neighbors = []
        if i > 0:
            neighbors.append(i - 1)
        neighbors.append(i)
        if i < N - 1:
            neighbors.append(i + 1)

        X = []
        for k in range(3):
            row = []
            for n in neighbors:
                xi = x[n].item()
                if k == 0:
                    row.append(1.0)
                elif k == 1:
                    row.append(xi)
                else:
                    row.append(xi ** 2)
            X.append(row)

        X = torch.tensor(X).float().to(device)
        b = torch.tensor([[0.0], [0.0], [2.0]]).float().to(device)

        w = torch.linalg.lstsq(X, b).solution

        for idx, n in enumerate(neighbors):
            A[i, n] = w[idx].item()

    return A

def physics_loss_hybrid(model, t, x_grid, A, alpha=1.0):
    u = model(t, x_grid)
    u_reshaped = u.view(-1, A.shape[0], 1)
    u_xx = torch.matmul(A, u_reshaped)
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    residual = u_t - alpha * u_xx.view(-1, 1)
    return torch.mean(residual**2)

def train_hybrid(model, optimizer, num_epochs, A, lr=1e-3, alpha=1.0):
    losses = []
    errors = []

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    x_grid = generate_1d_grid(N=100)

    model.to(device)
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()

        t = torch.rand(3000, 1).to(device)
        x = x_grid.repeat(3000 // x_grid.shape[0], 1)
        
        t = t[:len(x)]
        x = x[:len(t)]
        
        t.requires_grad = True
        x.requires_grad = True

        f_loss = physics_loss_hybrid(model, t, x, A, alpha)

        t_bc = torch.rand(500, 1).to(device)
        x_left = -torch.ones(500, 1).to(device)
        x_right = torch.ones(500, 1).to(device)
        bc_loss = boundary_loss(model, t_bc, x_left, x_right)

        x_ic = torch.rand(1000, 1).to(device) * 2 - 1
        ic_loss = initial_loss(model, x_ic)

        loss = 1.0 * f_loss + 0.5 * bc_loss + 0.5 * ic_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        errors.append(calculate_error(model, t, x))

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Total Loss: {loss.item()}, Error: {errors[-1]}')

    return losses, errors

def convergence_test():
    N_list = [50, 100, 200]
    final_errors = []
    
    for N in N_list:
        print(f"\nStarting convergence test for N={N}")
        try:
            x_grid = generate_1d_grid(L=1.0, N=N)
            print(f"Generated grid with {N} points")
            A = build_local_fit_matrix(x_grid)
            print("Local fit matrix built successfully")
            model = HybridPINN()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            print("Starting training...")
            losses, errors = train_hybrid(model, optimizer, 5000, A, alpha=2.0)
            final_error = errors[-1]
            final_errors.append(final_error)
            print(f"N={N}, Error={final_error:.4e}")
        except Exception as e:
            print(f"Error occurred for N={N}: {str(e)}")
            final_errors.append(float('nan'))
    
    plt.loglog(N_list, final_errors, marker='o')
    plt.xlabel("Number of grid points")
    plt.ylabel("Final Error")
    plt.savefig("convergence_curve.png")

if __name__ == "__main__":
    x_grid = generate_1d_grid(N=100)
    A = build_local_fit_matrix(x_grid)
    model = HybridPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses, errors = train_hybrid(model, optimizer, 10000, A, alpha=2.0)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(errors, label='Error', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_HyBrid_PINN.png')
    plt.show()

    convergence_test()

    parameter_exploration()
