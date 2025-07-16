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
    prediction_steps: int = 100  # 预测步数

    # Curriculum Learning and Dynamic Flow Parameters
    use_curriculum_learning: bool = False
    base_template_rou_file: Optional[str] = None
    static_phase_ratio: float = 0.8
    base_flow_rate: int = 300
    dynamic_flows_rate: int = 10




@dataclass
class RunningConfig(BaseConfig):
    operation = "PREDICT"
    gui: bool = False
    render_mode = None
    num_seconds = 100


@dataclass
class AlgorithmConfig(BaseConfig):
    """包含所有算法通用超参数的配置类"""
    total_timesteps: int = 1_000_000
    num_seconds: int = 3600
    n_eval_episodes: int = 10
    tensorboard_logs: str = "logs"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    verbose: int = 0
    learning_starts: int = 0
    train_freq: int = 1
    batch_size: int = 256
    buffer_size: int = 50_000
    n_epochs: int = 20
    n_steps: int = 1024
    exploration_initial_eps: float = 0.05
    exploration_final_eps: float = 0.01


@dataclass
class DQNConfig(AlgorithmConfig):
    """DQN专属配置（重载通用配置）"""
    learning_rate: float = 1e-4          # DQN使用较低的学习率
    batch_size: int = 64               # DQN使用较小的批量
    target_update_interval: int = 1000   # DQN专属：目标网络更新频率
    tau: float = 0.001                   # DQN专属：软更新参数


@dataclass
class PPOConfig(AlgorithmConfig):
    """PPO专属配置（无需重载）"""
    pass  # PPO的常用参数与AlgorithmConfig的默认值匹配


@dataclass
class A2CConfig(AlgorithmConfig):
    """A2C专属配置（重载通用配置）"""
    batch_size: int = 64               # A2C通常使用较小的批量
    n_epochs: int = 50                 # A2C每轮训练更新次数


@dataclass
class SACConfig(AlgorithmConfig):
    """SAC专属配置（重载通用配置）"""
    buffer_size: int = 100_000           # SAC使用较大的经验回放池
    tau: float = 0.005                   # SAC的软更新速率

