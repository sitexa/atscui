from atscui.models.base_agent import BaseAgent
from atscui.exceptions import ModelError


class PPOAgent(BaseAgent):
    """PPO算法智能体
    
    Proximal Policy Optimization算法的具体实现，适用于连续和离散动作空间。
    """

    def _create_model(self):
        """创建PPO模型
        
        Returns:
            PPO: 配置好的PPO模型实例
            
        Raises:
            ModelError: 模型创建失败时抛出
        """
        try:
            from stable_baselines3 import PPO
            model = PPO(
                env=self.env,
                policy="MlpPolicy",
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=getattr(self.config, 'gae_lambda', 0.95),
                clip_range=getattr(self.config, 'clip_range', 0.2),
                ent_coef=getattr(self.config, 'ent_coef', 0.0),
                vf_coef=getattr(self.config, 'vf_coef', 0.5),
                max_grad_norm=getattr(self.config, 'max_grad_norm', 0.5),
                tensorboard_log=self.config.tensorboard_logs,
                verbose=self.config.verbose,
            )
            
            self.logger.info("PPO模型创建成功")
            self.logger.info(f"模型参数: lr={self.config.learning_rate}, "
                           f"n_steps={self.config.n_steps}, "
                           f"batch_size={self.config.batch_size}, "
                           f"n_epochs={self.config.n_epochs}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"PPO模型创建失败: {e}")
            raise ModelError(f"PPO模型创建失败: {str(e)}")
    
    def _train_implementation(self, callback=None):
        """PPO特定的训练实现
        
        Args:
            callback: 训练回调函数
            
        Returns:
            训练结果
        """
        # 记录PPO特定的训练参数
        self.logger.info(f"PPO训练参数: "
                        f"n_steps={self.config.n_steps}, "
                        f"batch_size={self.config.batch_size}, "
                        f"n_epochs={self.config.n_epochs}")
        
        # 调用基类的训练方法
        result = super()._train_implementation(callback)
        
        # 更新PPO特定的训练指标
        self.training_metrics.update({
            'n_steps': self.config.n_steps,
            'batch_size': self.config.batch_size,
            'n_epochs': self.config.n_epochs
        })
        
        return result
