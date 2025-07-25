from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# 网流算操记，时秒步课图，单模课控路
@dataclass
class BaseConfig:
    cross_name: str = "NoName"
    net_file: str = ""
    rou_file: str = ""
    algo_name: Optional[str] = "DQN"
    operation: Optional[str] = "TRAIN"
    tensorboard_logs: str = "logs"
    total_timesteps: int = 1_000_000
    num_seconds: int = 3600
    prediction_steps: int = 100  # 预测步数
    use_curriculum_learning: bool = False
    gui: bool = False

    single_agent: bool = True
    render_mode: Optional[str] = None

    curriculum_static_ratio: Optional[float] = 0.8 # 课程学习的静态阶段时长占比
    curriculum_base_flow: Optional[int] = 300 # 课程学习的基础流率
    curriculum_dynamic_rate: Optional[int] = 10 # 课程学习的动态阶段生成速率

   # 相位控制模式：sequential或flexible, 控制模式是自动的，根据流量情况自动切换相位控制模式
    phase_control: Optional[str] = "sequential"  

    # 以下路径是默认目录，通常不需要修改
    model_dir = Path("models")
    output_dir = Path("outs")
    config_dir = output_dir / "configs"
    csv_dir = output_dir / "train"
    eval_dir = output_dir / "eval"
    predict_dir = output_dir / "predict"

    csv_path: str = ""
    model_path: str = ""
    predict_path: str = ""
    eval_path: str = ""

@dataclass
class RunningConfig(BaseConfig):
    operation: str = "PREDICT"
    gui: bool = False
    render_mode = None
    num_seconds = 100


@dataclass
class AlgorithmConfig(BaseConfig):
    """包含所有算法通用超参数的配置类"""
    total_timesteps: int = 1_000_000
    num_seconds: int = 3600
    learning_rate: float = 3e-4
    learning_starts: int = 0
    gamma: float = 0.99
    train_freq: int = 1
    batch_size: int = 256
    buffer_size: int = 50_000
    n_eval_episodes: int = 10
    n_epochs: int = 20
    n_steps: int = 1024
    exploration_initial_eps: float = 0.05
    exploration_final_eps: float = 0.01
    verbose: int = 0


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
    ent_coef: str = 'auto'               # 自动调整熵系数
    gradient_steps: int = 1              # 梯度更新步数
    target_entropy: str = 'auto'         # 自动设置目标熵
