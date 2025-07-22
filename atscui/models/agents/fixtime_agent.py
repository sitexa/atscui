"""固定配时智能体模块

提供固定配时控制的智能体实现，用于与强化学习算法进行对比分析。
"""
from typing import Any, Dict, Tuple
import numpy as np

from atscui.models.base_agent import BaseAgent
from atscui.config import AlgorithmConfig
from atscui.exceptions import ModelError


class FixTimeAgent(BaseAgent):
    """固定配时智能体
    
    该智能体不进行实际的强化学习训练，而是直接运行固定配时仿真。
    主要用于与强化学习算法的性能对比分析。
    """
    
    def __init__(self, env, config: AlgorithmConfig):
        # 先调用父类初始化以设置logger
        super().__init__(env, config)
        
        # 设置环境为固定时序模式
        if hasattr(env, 'fixed_ts'):
            env.fixed_ts = True
            self.logger.info("环境已设置为固定时序模式")
        elif hasattr(env, 'env') and hasattr(env.env, 'fixed_ts'):
            env.env.fixed_ts = True
            self.logger.info("环境已设置为固定时序模式")
        else:
            self.logger.warning("无法设置环境为固定时序模式")
        
        self.logger.info("固定配时智能体已创建，将直接运行仿真")
        
        # 固定配时不需要模型，设置为None
        self.model = None
        self.fixed_ts = True
    
    def _create_model(self):
        """固定配时不需要模型"""
        return None
    
    def _run_simulation(self, total_steps: int) -> Dict[str, Any]:
        """运行固定配时仿真
        
        Args:
            total_steps: 仿真总步数
            
        Returns:
            仿真结果字典
        """
        self.logger.info(f"开始固定配时仿真，总步数: {total_steps}")
        
        step_count = 0
        
        try:
            # 重置环境
            obs = self.env.reset()
            
            while step_count < total_steps:
                # 固定配时模式下，环境会忽略动作，但仍需要提供一个动作
                action = None
                if hasattr(self.env, 'action_space'):
                    if hasattr(self.env.action_space, 'n'):
                        action = 0  # 离散动作空间的默认动作
                    else:
                        action = np.zeros(self.env.action_space.shape)  # 连续动作空间的零动作
                
                obs, reward, done, info = self.env.step(action)
                step_count += 1
                
                # 如果环境结束，重置环境
                if done:
                    obs = self.env.reset()
                
                # 定期报告进度
                if step_count % 1000 == 0:
                    self.logger.info(f"固定配时仿真进度: {step_count}/{total_steps}")
        
        except Exception as e:
            self.logger.error(f"固定配时仿真过程中发生错误: {e}")
            raise
        
        self.logger.info("固定配时仿真完成")
        return {"fixed_timing_simulation": True, "steps_completed": step_count}
    
    def _train_implementation(self, callback=None):
        """固定配时训练实现 - 直接运行仿真"""
        return self._run_simulation(self.config.total_timesteps)
    
    def evaluate(self, n_eval_episodes: int = 10) -> Dict[str, Any]:
        """固定配时评估 - 直接运行仿真"""
        # 计算评估步数（假设每个episode为配置的总步数）
        eval_steps = self.config.total_timesteps * n_eval_episodes
        return self._run_simulation(eval_steps)
    
    def predict(self, observation, deterministic: bool = True) -> Tuple[Any, Any]:
        """固定配时预测 - 直接运行仿真
        
        对于固定配时，预测就是运行一步仿真
        """
        # 固定配时模式下返回默认动作
        if hasattr(self.env, 'action_space'):
            if hasattr(self.env.action_space, 'n'):
                return 0, None  # 离散动作空间
            else:
                return np.zeros(self.env.action_space.shape), None  # 连续动作空间
        return None, None
    
    def save(self, path: str):
        """固定配时不需要保存模型"""
        self.logger.info("固定配时无需保存模型")
        pass
    
    def load(self, path: str):
        """固定配时不需要加载模型"""
        self.logger.info("固定配时无需加载模型")
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取固定配时模型信息"""
        info = super().get_model_info()
        info.update({
            "algorithm": "FIXTIME",
            "policy": "FixedTiming",
            "fixed_timing": True,
            "model": None
        })
        return info