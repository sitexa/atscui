from atscui.models.base_agent import BaseAgent
from atscui.exceptions import ModelError


class DQNAgent(BaseAgent):
    """DQN算法智能体
    
    Deep Q-Network算法的具体实现，适用于离散动作空间的强化学习任务。
    """

    def _create_model(self):
        """创建DQN模型
        
        Returns:
            DQN: 配置好的DQN模型实例
            
        Raises:
            ModelError: 模型创建失败时抛出
        """
        try:
            from stable_baselines3 import DQN
            model = DQN(
                env=self.env,
                policy="MlpPolicy",
                learning_rate=self.config.learning_rate,
                learning_starts=self.config.learning_starts,
                train_freq=self.config.train_freq,
                target_update_interval=self.config.target_update_interval,
                exploration_initial_eps=self.config.exploration_initial_eps,
                exploration_final_eps=self.config.exploration_final_eps,
                buffer_size=self.config.buffer_size,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                tau=self.config.tau,
                tensorboard_log=self.config.tensorboard_logs,
                verbose=self.config.verbose,
            )
            
            self.logger.info("DQN模型创建成功")
            self.logger.info(f"模型参数: lr={self.config.learning_rate}, "
                           f"buffer_size={self.config.buffer_size}, "
                           f"batch_size={self.config.batch_size}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"DQN模型创建失败: {e}")
            raise ModelError(f"DQN模型创建失败: {str(e)}")
    
    def _train_implementation(self, callback=None):
        """DQN特定的训练实现
        
        Args:
            callback: 训练回调函数
            
        Returns:
            训练结果
        """
        # 记录DQN特定的训练参数
        self.logger.info(f"DQN训练参数: "
                        f"exploration_eps=({self.config.exploration_initial_eps}, "
                        f"{self.config.exploration_final_eps}), "
                        f"target_update_interval={self.config.target_update_interval}")
        
        # 调用基类的训练方法
        result = super()._train_implementation(callback)
        
        # 更新DQN特定的训练指标
        if hasattr(self.model, 'exploration_rate'):
            self.training_metrics['final_exploration_rate'] = self.model.exploration_rate
        
        return result
