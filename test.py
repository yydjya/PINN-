import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("cpu")

class PINN(nn.Module):
    """Physics-Informed Neural Network 模型类，用于求解偏微分方程"""
    def __init__(self):
        super(PINN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        ).to(device)

    def forward(self, t, x):
        u = self.layer(torch.cat([t, x], dim=1))
        return u

def physics_loss(model, t, x, alpha=1.0):
    """计算物理驱动损失项，强制满足热传导方程"""
    u = model(t, x)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]
    f = (u_t - alpha * u_xx).pow(2).mean()
    return f

def boundary_loss(model, t_bc, x_left, x_right):
    """计算边界条件损失项，强制满足左右边界约束"""
    u_left = model(t_bc, x_left)
    u_right = model(t_bc, x_right)
    loss_left = (u_left - g1(t_bc)).pow(2).mean()
    loss_right = (u_right - g2(t_bc)).pow(2).mean()
    return loss_left + loss_right

def g1(t):
    """左边界条件函数（Dirichlet边界条件）"""
    return torch.zeros_like(t)

def g2(t):
    """右边界条件函数（Dirichlet边界条件）"""
    return torch.zeros_like(t)

def initial_loss(model, x_ic):
    """计算初始条件损失项，强制满足初始时刻分布"""
    t_0 = torch.zeros_like(x_ic).to(device)
    u_init = model(t_0, x_ic)
    u_exact = f(x_ic)
    return (u_init - u_exact).pow(2).mean()

def f(x):
    """初始条件函数：u(x,0) = sin(πx)"""
    return torch.sin(np.pi * x)

def analytical_solution(t, x, alpha=1.0, n_terms=50):
    """解析解计算函数（有限项级数解）"""
    u = torch.zeros_like(x)
    for n in range(1, n_terms + 1):
        lambda_n = (n * np.pi) ** 2
        u += (2 / (n * np.pi)) * torch.sin(n * np.pi * x) * torch.exp(-alpha * lambda_n * t)
    return u

def calculate_error(model, t, x):
    """计算模型预测与解析解之间的平均绝对误差"""
    with torch.no_grad():
        u_pred = model(t, x)
        u_exact = analytical_solution(t, x)
        return torch.mean((u_pred - u_exact).abs()).item()

def train(model, optimizer, num_epochs, lr=1e-3, alpha=1.0):
    """训练物理信息神经网络模型"""
    losses = []
    errors = []

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

        loss = f_loss + bc_loss + ic_loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        errors.append(calculate_error(model, t, x))

        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Total Loss: {loss.item()}, Error: {errors[-1]}')

    return losses, errors

def parameter_exploration():
    """执行超参数探索实验，评估不同学习率和alpha参数对模型性能的影响"""
    learning_rates = [1e-2, 1e-3, 1e-4]
    alphas = [0.5, 1.0, 2.0]
    num_epochs = 5000

    results = []

    for lr in learning_rates:
        for alpha in alphas:
            print(f"\nTraining with lr={lr}, alpha={alpha}")
            model = PINN()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            losses, errors = train(model, optimizer, num_epochs, lr, alpha)
            final_error = errors[-1]
            results.append((lr, alpha, final_error))
            print(f"Final error: {final_error}")

    print("\nParameter exploration results:")
    for lr, alpha, error in results:
        print(f"lr={lr}, alpha={alpha}, final error={error}")

if __name__ == "__main__":
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses, errors = train(model, optimizer, num_epochs=10000)

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
    plt.savefig('training_curves.png')
    plt.show()

    parameter_exploration()
