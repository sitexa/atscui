from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseConfig:
    net_file: str
    rou_file: str
    csv_path: str
    model_path: str
    eval_path: str
    predict_path: str
    single_agent: bool = True
    gui: bool = False
    render_mode: Optional[str] = None
    operation: Optional[str] = "TRAIN"
    algo_name: Optional[str] = "DQN"
    phase_control: Optional[str] = "sequential"  # 相位控制模式：sequential或flexible


@dataclass
class TrainingConfig(BaseConfig):
    total_timesteps: int = 1_000_000  # 总训练时间步
    num_seconds: int = 20_000  # 每回合episode仿真步(时长)
    n_steps: int = 1024  # A2C价值网络更新间隔时间步
    n_eval_episodes: int = 5  # 评估回合数
    tensorboard_logs: str = "logs"
    learning_rate = 1e-3
    gamma = 0.9
    verbose: int = 0
    learning_starts: int = 0
    train_freq: int = 1
    target_update_interval: int = 1000
    exploration_initial_eps: float = 0.05
    exploration_final_eps: float = 0.01
    batch_size: int = 1024
    n_epochs: int = 100
    buffer_size: int = 10_000
    tau: float = 0.001

@dataclass
class RunningConfig(BaseConfig):
    operation = "PREDICT"
    gui: bool = False
    render_mode = None
    num_seconds = 100

@dataclass
class DQNConfig(BaseConfig):
    total_timesteps: int = 1_000_000  # 总训练时间步
    num_seconds: int = 20_000  # 每回合episode仿真步(时长)
    n_steps: int = 1024  # DQN不使用A2C的更新间隔，保持原有设置
    n_eval_episodes: int = 5  # 评估回合数
    tensorboard_logs: str = "logs"
    learning_rate: float = 1e-4  # DQN较低的学习率
    gamma: float = 0.99  # DQN适合较大的gamma
    verbose: int = 0
    learning_starts: int = 0
    train_freq: int = 1
    target_update_interval: int = 1000  # 目标网络更新频率
    exploration_initial_eps: float = 0.05  # 初始探索率
    exploration_final_eps: float = 0.01  # 最终探索率
    batch_size: int = 64  # 小批量，适合DQN的稳定训练
    n_epochs: int = 100  # 不使用，DQN使用经验回放池进行训练
    buffer_size: int = 10_000  # 经验回放池大小
    tau: float = 0.001  # 软更新参数

@dataclass
class PPOConfig(BaseConfig):
    total_timesteps: int = 1_000_000  # 总训练时间步
    num_seconds: int = 20_000  # 每回合episode仿真步(时长)
    n_steps: int = 1024  # 每次迭代的数据步长
    n_eval_episodes: int = 5  # 评估回合数
    tensorboard_logs: str = "logs"
    learning_rate: float = 1e-4  # PPO通常使用较低的学习率
    gamma: float = 0.99  # PPO适合较大的gamma
    verbose: int = 0
    learning_starts: int = 0
    train_freq: int = 1  # 每步都进行训练
    target_update_interval: int = 1000  # 不适用（PPO没有目标网络）
    exploration_initial_eps: float = 0.05  # 初始探索率（适用于离散动作空间）
    exploration_final_eps: float = 0.01  # 最终探索率
    batch_size: int = 64  # PPO常用的小批量，适合训练
    n_epochs: int = 20  # 每轮策略优化的更新次数
    buffer_size: int = 10_000  # 经验回放池大小
    tau: float = 0.001  # 不适用（PPO没有目标网络）

@dataclass
class A2CConfig(BaseConfig):
    total_timesteps: int = 1_000_000  # 总训练时间步
    num_seconds: int = 20_000  # 每回合episode仿真步(时长)
    n_steps: int = 1024  # 每次更新的时间步
    n_eval_episodes: int = 5  # 评估回合数
    tensorboard_logs: str = "logs"
    learning_rate: float = 1e-4  # A2C使用较低的学习率
    gamma: float = 0.99  # A2C适合较大的gamma
    verbose: int = 0
    learning_starts: int = 0
    train_freq: int = 1  # 每步都进行训练
    target_update_interval: int = 1000  # 不适用（A2C没有目标网络）
    exploration_initial_eps: float = 0.05  # 初始探索率（适用于离散动作空间）
    exploration_final_eps: float = 0.01  # 最终探索率
    batch_size: int = 64  # A2C通常使用较小的批量
    n_epochs: int = 50  # 每轮训练进行50次更新
    buffer_size: int = 10_000  # 经验回放池大小
    tau: float = 0.001  # 不适用（A2C没有目标网络）

@dataclass
class SACConfig(BaseConfig):
    total_timesteps: int = 1_000_000  # 总训练时间步
    num_seconds: int = 20_000  # 每回合episode仿真步(时长)
    n_steps: int = 1024  # 每次更新的时间步
    n_eval_episodes: int = 5  # 评估回合数
    tensorboard_logs: str = "logs"
    learning_rate: float = 3e-4  # SAC使用较低的学习率
    gamma: float = 0.99  # SAC适合较大的gamma
    verbose: int = 0
    learning_starts: int = 0
    train_freq: int = 1  # 每步都进行训练
    target_update_interval: int = 1000  # 不适用（SAC没有目标网络）
    exploration_initial_eps: float = 0.05  # 初始探索率（适用于离散动作空间）
    exploration_final_eps: float = 0.01  # 最终探索率
    batch_size: int = 256  # SAC使用较大的批量
    n_epochs: int = 100  # 每轮训练进行100次更新
    buffer_size: int = 100_000  # SAC使用较大的经验回放池
    tau: float = 0.005  # SAC的软更新速率
