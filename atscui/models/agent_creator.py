"""智能体创建工厂模块

提供统一的智能体创建接口，支持多种强化学习算法。
"""
import inspect
from typing import Dict, Type, Any

from atscui.config import AlgorithmConfig
from atscui.models.agents import DQNAgent, A2CAgent, PPOAgent, SACAgent
from atscui.models.base_agent import BaseAgent
from atscui.exceptions import ConfigurationError, ModelError
from atscui.logging_manager import get_logger


class AgentFactory:
    """智能体工厂类
    
    负责创建和管理不同类型的强化学习智能体。
    """
    
    # 支持的智能体类型映射
    AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
        "DQN": DQNAgent,
        "A2C": A2CAgent,
        "PPO": PPOAgent,
        "SAC": SACAgent,
    }
    
    @property
    def ALGORITHM_REGISTRY(self) -> Dict[str, Type]:
        """延迟导入算法类型映射"""
        from stable_baselines3 import DQN, A2C, PPO, SAC
        return {
            "DQN": DQN,
            "A2C": A2C,
            "PPO": PPO,
            "SAC": SAC,
        }
    
    def __init__(self):
        self.logger = get_logger('agent_factory')
    
    def create_agent(self, env, config: AlgorithmConfig) -> BaseAgent:
        """创建智能体实例
        
        Args:
            env: 环境实例
            config: 训练配置
            
        Returns:
            BaseAgent: 创建的智能体实例
            
        Raises:
            ConfigurationError: 不支持的算法类型
            ModelError: 智能体创建失败
        """
        algo_name = config.algo_name.upper()
        
        if algo_name not in self.AGENT_REGISTRY:
            supported_algos = list(self.AGENT_REGISTRY.keys())
            error_msg = f"不支持的算法类型: {algo_name}，支持的算法: {supported_algos}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg, "algo_name")
        
        try:
            agent_class = self.AGENT_REGISTRY[algo_name]
            self.logger.info(f"正在创建 {algo_name} 智能体...")
            
            agent = agent_class(env, config)
            
            self.logger.info(f"{algo_name} 智能体创建成功")
            return agent
            
        except Exception as e:
            error_msg = f"创建 {algo_name} 智能体失败"
            self.logger.error(f"{error_msg}: {e}")
            raise ModelError(error_msg, str(e))
    
    def create_algorithm(self, env, algo_name: str, **kwargs) -> Any:
        """直接创建算法实例（不包装为智能体）
        
        Args:
            env: 环境实例
            algo_name: 算法名称
            **kwargs: 算法参数
            
        Returns:
            算法实例
            
        Raises:
            ConfigurationError: 不支持的算法类型
            ModelError: 算法创建失败
        """
        algo_name = algo_name.upper()
        
        algorithm_registry = self.ALGORITHM_REGISTRY
        if algo_name not in algorithm_registry:
            supported_algos = list(algorithm_registry.keys())
            error_msg = f"不支持的算法类型: {algo_name}，支持的算法: {supported_algos}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg, "algo_name")
        
        try:
            algorithm_class = algorithm_registry[algo_name]
            self.logger.info(f"正在创建 {algo_name} 算法实例...")

            # --- 动态参数过滤 ---
            # 获取算法构造函数的有效参数
            sig = inspect.signature(algorithm_class.__init__)
            valid_keys = {p.name for p in sig.parameters.values()}

            # 过滤kwargs，只传递算法构造函数接受的参数
            model_params = {key: value for key, value in kwargs.items() if key in valid_keys}

            # 确保env和policy被正确设置
            model_params['env'] = env
            if 'policy' not in model_params:
                model_params['policy'] = 'MlpPolicy'
            
            # 处理 tensorboard_log 的重命名
            if 'tensorboard_logs' in kwargs and 'tensorboard_log' in valid_keys:
                model_params['tensorboard_log'] = kwargs['tensorboard_logs']

            self.logger.info(f"使用以下有效参数创建模型: {list(model_params.keys())}")
            algorithm = algorithm_class(**model_params)
            
            self.logger.info(f"{algo_name} 算法实例创建成功")
            return algorithm
            
        except Exception as e:
            error_msg = f"创建 {algo_name} 算法实例失败"
            self.logger.error(f"{error_msg}: {e}")
            raise ModelError(error_msg, str(e))
    
    def get_supported_algorithms(self) -> list:
        """获取支持的算法列表
        
        Returns:
            list: 支持的算法名称列表
        """
        return list(self.AGENT_REGISTRY.keys())
    
    def register_agent(self, algo_name: str, agent_class: Type[BaseAgent]):
        """注册新的智能体类型
        
        Args:
            algo_name: 算法名称
            agent_class: 智能体类
        """
        algo_name = algo_name.upper()
        self.AGENT_REGISTRY[algo_name] = agent_class
        self.logger.info(f"已注册新的智能体类型: {algo_name}")
    
    def register_algorithm(self, algo_name: str, algorithm_class: Type):
        """注册新的算法类型
        
        Args:
            algo_name: 算法名称
            algorithm_class: 算法类
        """
        algo_name = algo_name.upper()
        self.ALGORITHM_REGISTRY[algo_name] = algorithm_class
        self.logger.info(f"已注册新的算法类型: {algo_name}")


# 全局工厂实例
_agent_factory = AgentFactory()


def createAgent(env, config: AlgorithmConfig) -> BaseAgent:
    """创建智能体的便捷函数
    
    Args:
        env: 环境实例
        config: 训练配置
        
    Returns:
        BaseAgent: 创建的智能体实例
    """
    return _agent_factory.create_agent(env, config)


def createAlgorithm(env, algo_name: str, **kwargs):
    """创建算法实例的便捷函数
    
    Args:
        env: 环境实例
        algo_name: 算法名称
        **kwargs: 算法参数
        
    Returns:
        算法实例
    """
    return _agent_factory.create_algorithm(env, algo_name, **kwargs)


def get_supported_algorithms() -> list:
    """获取支持的算法列表
    
    Returns:
        list: 支持的算法名称列表
    """
    return list(_agent_factory.ALGORITHM_REGISTRY.keys())
