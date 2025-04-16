import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("cpu")


class PINN(nn.Module):
    """Physics-Informed Neural Network 模型类，用于求解偏微分方程

    网络结构包含6个全连接层，激活函数使用Tanh
    输入维度为2（时间t和空间x），输出维度为1（物理场u预测值）
    """

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
        """前向传播过程

        Args:
            t: 时间张量，形状为(batch_size, 1)
            x: 空间坐标张量，形状为(batch_size, 1)

        Returns:
            u: 物理场预测值，形状为(batch_size, 1)
        """
        u = self.layer(torch.cat([t, x], dim=1))
        return u


def physics_loss(model, t, x, alpha=1.0):
    """计算物理驱动损失项，强制满足热传导方程

    Args:
        model: PINN模型实例
        t: 时间张量，形状为(batch_size, 1)
        x: 空间坐标张量，形状为(batch_size, 1)
        alpha: 热扩散系数，默认1.0

    Returns:
        f: 物理残差的均方误差（PDE方程不满足程度）
    """
    u = model(t, x)
    # 计算各阶导数
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]
    # 热传导方程残差：u_t - αu_xx
    f = (u_t - alpha * u_xx).pow(2).mean()
    return f


def boundary_loss(model, t_bc, x_left, x_right):
    """计算边界条件损失项，强制满足左右边界约束

    Args:
        model: PINN模型实例
        t_bc: 边界时间采样点，形状为(batch_size, 1)
        x_left: 左边界坐标（x=-1），形状为(batch_size, 1)
        x_right: 右边界坐标（x=1），形状为(batch_size, 1)

    Returns:
        边界条件残差的均方误差之和
    """
    u_left = model(t_bc, x_left)
    u_right = model(t_bc, x_right)
    # 使用预定义的边界条件函数计算损失
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
    """计算初始条件损失项，强制满足初始时刻分布

    Args:
        model: PINN模型实例
        x_ic: 初始时刻空间坐标采样点，形状为(batch_size, 1)

    Returns:
        初始条件残差的均方误差
    """
    t_0 = torch.zeros_like(x_ic).to(device)
    u_init = model(t_0, x_ic)
    u_exact = f(x_ic)
    return (u_init - u_exact).pow(2).mean()


def f(x):
    """初始条件函数：u(x,0) = sin(πx)"""
    return torch.sin(np.pi * x)


def analytical_solution(t, x, alpha=1.0, n_terms=50):
    """解析解计算函数（有限项级数解）

    Args:
        t: 时间张量
        x: 空间坐标张量
        alpha: 热扩散系数，默认1.0
        n_terms: 级数展开项数，默认50项

    Returns:
        解析解的近似值
    """
    u = torch.zeros_like(x)
    for n in range(1, n_terms + 1):
        lambda_n = (n * np.pi) ** 2
        u += (2 / (n * np.pi)) * torch.sin(n * np.pi * x) * torch.exp(-alpha * lambda_n * t)
    return u


def calculate_error(model, t, x):
    """计算模型预测与解析解之间的平均绝对误差

    Args:
        model: 训练好的PINN模型
        t: 测试时间点
        x: 测试空间点

    Returns:
        平均绝对误差值
    """
    with torch.no_grad():
        u_pred = model(t, x)
        u_exact = analytical_solution(t, x)
        return torch.mean((u_pred - u_exact).abs()).item()

def train(model, optimizer, num_epochs, lr=1e-3, alpha=1.0):
    """
    训练物理信息神经网络模型

    参数:
        model: 待训练的神经网络模型实例
        optimizer: 参数优化器实例
        num_epochs: 总训练轮次数
        lr: 学习率 (默认: 1e-3)
        alpha: 物理损失权重系数 (默认: 1.0)

    返回值:
        losses: 各epoch训练损失值的列表
        errors: 各epoch模型误差值的列表
    """
    losses = []
    errors = []

    # 设置优化器学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 主训练循环
    model.to(device)
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()

        # 生成训练数据点
        t = torch.rand(3000, 1).to(device)  # 时间维度采样
        x = torch.rand(3000, 1).to(device) * 2 - 1  # 空间维度采样 [-1,1]
        t.requires_grad = True  # 启用自动微分
        x.requires_grad = True

        # 计算各类型损失
        f_loss = physics_loss(model, t, x, alpha)  # PDE物理方程残差损失

        # 边界条件采样点
        t_bc = torch.rand(500, 1).to(device)
        x_left = -torch.ones(500, 1).to(device)  # 左边界
        x_right = torch.ones(500, 1).to(device)   # 右边界
        bc_loss = boundary_loss(model, t_bc, x_left, x_right)  # 边界条件损失

        # 初始条件采样点
        x_ic = torch.rand(1000, 1).to(device) * 2 - 1
        ic_loss = initial_loss(model, x_ic)  # 初始条件损失

        # 综合损失反向传播
        loss = f_loss + bc_loss + ic_loss
        loss.backward()
        optimizer.step()

        # 记录训练指标
        losses.append(loss.item())
        errors.append(calculate_error(model, t, x))  # 计算模型预测误差

        # 定期输出训练状态
        if epoch % 1000 == 0:
            print(f'Epoch {epoch}, Total Loss: {loss.item()}, Error: {errors[-1]}')

    return losses, errors



def parameter_exploration():
    """执行超参数探索实验，评估不同学习率和alpha参数对模型性能的影响

    实验组合：
    - 学习率：1e-2, 1e-3, 1e-4
    - 热扩散系数alpha：0.5, 1.0, 2.0
    - 固定训练轮数：5000

    结果保存格式：
    元组列表，每个元素为（学习率，alpha值，最终误差）
    """
    learning_rates = [1e-2, 1e-3, 1e-4]
    alphas = [0.5, 1.0, 2.0]
    num_epochs = 5000

    results = []  # 存储实验结果

    # 遍历所有参数组合
    for lr in learning_rates:
        for alpha in alphas:
            print(f"\nTraining with lr={lr}, alpha={alpha}")
            model = PINN()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            losses, errors = train(model, optimizer, num_epochs, lr, alpha)
            final_error = errors[-1]
            results.append((lr, alpha, final_error))
            print(f"Final error: {final_error}")

    # 打印最终实验结果
    print("\nParameter exploration results:")
    for lr, alpha, error in results:
        print(f"lr={lr}, alpha={alpha}, final error={error}")


if __name__ == "__main__":

    # 初始化模型和优化器
    model = PINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 执行主训练过程（10000轮）
    losses, errors = train(model, optimizer, num_epochs=10000)

    # 绘制训练曲线
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
    plt.savefig('training_curves.png')  # 保存训练曲线图
    plt.show()

    # 执行参数探索实验
    parameter_exploration()
