from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import time

from atscui.config import AlgorithmConfig
from atscui.exceptions import ModelError, TrainingError
from atscui.logging_manager import get_logger


class BaseAgent(ABC):
    """智能体基类
    
    提供统一的智能体接口，包含模型创建、训练、预测、保存和加载功能。
    所有具体的算法智能体都应该继承此类。
    """
    
    def __init__(self, env, config: AlgorithmConfig):
        self.env = env
        self.config = config
        self.logger = get_logger(f'agent_{config.algo_name}')
        self.model = None
        self.training_metrics = {}
        
        try:
            self.model = self._create_model()
            self.logger.info(f"成功创建 {config.algo_name} 智能体")
        except Exception as e:
            error_msg = f"创建 {config.algo_name} 智能体失败"
            self.logger.error(f"{error_msg}: {e}")
            raise ModelError(error_msg, str(e))

    @abstractmethod
    def _create_model(self):
        """创建并返回特定算法的模型
        
        子类必须实现此方法来创建具体的算法模型。
        
        Returns:
            具体算法的模型实例
            
        Raises:
            ModelError: 模型创建失败时抛出
        """
        pass

    def train(self, callback=None) -> Dict[str, Any]:
        """训练智能体
        
        Args:
            callback: 可选的回调函数，用于监控训练进度
            
        Returns:
            Dict[str, Any]: 训练指标字典
            
        Raises:
            TrainingError: 训练过程中出现错误时抛出
        """
        if not self.model:
            raise TrainingError("模型未初始化", self.config.algo_name)
        
        try:
            self.logger.info(f"开始训练 {self.config.algo_name} 模型")
            self.logger.info(f"训练参数: 总步数={self.config.total_timesteps}, 学习率={self.config.learning_rate}")
            
            start_time = time.time()
            
            # 执行具体的训练逻辑
            result = self._train_implementation(callback)
            
            training_time = time.time() - start_time
            self.training_metrics.update({
                'training_time': training_time,
                'total_timesteps': self.config.total_timesteps,
                'algorithm': self.config.algo_name
            })
            
            self.logger.info(f"训练完成，耗时: {training_time:.2f}秒")
            return self.training_metrics
            
        except Exception as e:
            error_msg = f"训练过程中发生错误"
            self.logger.error(f"{error_msg}: {e}")
            raise TrainingError(error_msg, self.config.algo_name)
    
    def _train_implementation(self, callback=None):
        """具体的训练实现
        
        子类可以重写此方法来自定义训练逻辑。
        默认实现调用模型的learn方法。
        """
        return self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True,
            callback=callback
        )

    def predict(self, observation, deterministic: bool = True) -> Tuple[Any, Any]:
        """进行预测
        
        Args:
            observation: 观测值
            deterministic: 是否使用确定性策略
            
        Returns:
            Tuple[Any, Any]: (动作, 状态)
            
        Raises:
            ModelError: 模型未初始化或预测失败时抛出
        """
        if not self.model:
            raise ModelError("模型未初始化，无法进行预测")
        
        try:
            return self.model.predict(observation, deterministic=deterministic)
        except Exception as e:
            error_msg = f"预测失败"
            self.logger.error(f"{error_msg}: {e}")
            raise ModelError(error_msg, str(e))

    def save(self, path: str) -> bool:
        """保存模型
        
        Args:
            path: 保存路径
            
        Returns:
            bool: 保存是否成功
            
        Raises:
            ModelError: 保存失败时抛出
        """
        if not self.model:
            raise ModelError("模型未初始化，无法保存")
        
        try:
            # 确保目录存在
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            self.model.save(path)
            self.logger.info(f"模型已保存到: {path}")
            return True
            
        except Exception as e:
            error_msg = f"保存模型失败"
            self.logger.error(f"{error_msg}: {e}")
            raise ModelError(error_msg, path)

    def load(self, path: str) -> bool:
        """加载模型
        
        Args:
            path: 模型文件路径
            
        Returns:
            bool: 加载是否成功
            
        Raises:
            ModelError: 加载失败时抛出
        """
        if not Path(path).exists():
            raise ModelError(f"模型文件不存在", path)
        
        try:
            if self.model:
                self.model = self.model.load(path)
            else:
                # 如果模型未初始化，先创建再加载
                self.model = self._create_model()
                self.model = self.model.load(path)
            
            self.logger.info(f"模型已从 {path} 加载")
            return True
            
        except Exception as e:
            error_msg = f"加载模型失败"
            self.logger.error(f"{error_msg}: {e}")
            raise ModelError(error_msg, path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        if not self.model:
            return {'status': 'not_initialized'}
        
        return {
            'algorithm': self.config.algo_name,
            'policy': getattr(self.model, 'policy', 'unknown'),
            'learning_rate': self.config.learning_rate,
            'training_metrics': self.training_metrics,
            'status': 'initialized'
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            if hasattr(self.model, 'env') and self.model.env:
                self.model.env.close()
            self.logger.info(f"{self.config.algo_name} 智能体资源已清理")
        except Exception as e:
            self.logger.warning(f"清理资源时发生错误: {e}")
