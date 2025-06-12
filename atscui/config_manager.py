"""统一配置管理模块

提供配置的加载、验证、保存和管理功能。
"""

import json
import os
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, Any, Optional, Type, Union

from atscui.exceptions import ConfigurationError, FileOperationError, ValidationError
from atscui.logging_manager import get_logger


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.logger = get_logger('config')
        self.config_dir = Path("configs")
        self.config_dir.mkdir(exist_ok=True)
        self._config_cache = {}
    
    def validate_config(self, config: Any) -> bool:
        """验证配置对象"""
        try:
            # 检查必需字段
            required_fields = ['net_file', 'rou_file', 'algo_name']
            for field_name in required_fields:
                if hasattr(config, field_name):
                    value = getattr(config, field_name)
                    if not value:
                        raise ValidationError(
                            f"必需字段不能为空",
                            field_name=field_name,
                            value=value
                        )
                else:
                    raise ValidationError(
                        f"缺少必需字段",
                        field_name=field_name
                    )
            
            # 验证文件路径
            if hasattr(config, 'net_file') and config.net_file:
                if not os.path.exists(config.net_file):
                    raise ValidationError(
                        f"网络文件不存在",
                        field_name='net_file',
                        value=config.net_file
                    )
            
            if hasattr(config, 'rou_file') and config.rou_file:
                if not os.path.exists(config.rou_file):
                    raise ValidationError(
                        f"路由文件不存在",
                        field_name='rou_file',
                        value=config.rou_file
                    )
            
            # 验证算法名称
            if hasattr(config, 'algo_name'):
                valid_algorithms = ['DQN', 'PPO', 'A2C', 'SAC']
                if config.algo_name not in valid_algorithms:
                    raise ValidationError(
                        f"不支持的算法，支持的算法: {valid_algorithms}",
                        field_name='algo_name',
                        value=config.algo_name
                    )
            
            # 验证数值范围
            if hasattr(config, 'total_timesteps'):
                if config.total_timesteps <= 0:
                    raise ValidationError(
                        "训练步数必须大于0",
                        field_name='total_timesteps',
                        value=config.total_timesteps
                    )
            
            if hasattr(config, 'num_seconds'):
                if config.num_seconds <= 0:
                    raise ValidationError(
                        "仿真秒数必须大于0",
                        field_name='num_seconds',
                        value=config.num_seconds
                    )
            
            if hasattr(config, 'learning_rate'):
                if not (0 < config.learning_rate <= 1):
                    raise ValidationError(
                        "学习率必须在(0, 1]范围内",
                        field_name='learning_rate',
                        value=config.learning_rate
                    )
            
            self.logger.info(f"配置验证通过: {type(config).__name__}")
            return True
            
        except (ValidationError, ConfigurationError) as e:
            self.logger.error(f"配置验证失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"配置验证时发生未知错误: {e}")
            raise ConfigurationError(f"配置验证失败: {str(e)}")
    
    def save_config(self, config: Any, name: str) -> str:
        """保存配置到文件"""
        try:
            # 验证配置
            self.validate_config(config)
            
            # 转换为字典
            if hasattr(config, '__dict__'):
                config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else config.__dict__
            else:
                config_dict = dict(config)
            
            # 保存到文件
            config_file = self.config_dir / f"{name}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            # 缓存配置
            self._config_cache[name] = config_dict
            
            self.logger.info(f"配置已保存: {config_file}")
            return str(config_file)
            
        except Exception as e:
            error_msg = f"保存配置失败: {str(e)}"
            self.logger.error(error_msg)
            raise FileOperationError(error_msg, str(config_file), "save")
    
    def load_config(self, name: str, config_class: Type = None) -> Union[Dict, Any]:
        """从文件加载配置"""
        try:
            # 检查缓存
            if name in self._config_cache:
                config_dict = self._config_cache[name]
            else:
                # 从文件加载
                config_file = self.config_dir / f"{name}.json"
                if not config_file.exists():
                    raise FileOperationError(
                        f"配置文件不存在",
                        str(config_file),
                        "load"
                    )
                
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                
                # 缓存配置
                self._config_cache[name] = config_dict
            
            # 如果指定了配置类，则创建实例
            if config_class:
                try:
                    config = config_class(**config_dict)
                    self.validate_config(config)
                    return config
                except TypeError as e:
                    raise ConfigurationError(f"配置类型不匹配: {str(e)}")
            
            self.logger.info(f"配置已加载: {name}")
            return config_dict
            
        except Exception as e:
            if not isinstance(e, (FileOperationError, ConfigurationError)):
                error_msg = f"加载配置失败: {str(e)}"
                self.logger.error(error_msg)
                raise FileOperationError(error_msg, name, "load")
            raise
    
    def list_configs(self) -> list:
        """列出所有可用的配置"""
        configs = []
        for config_file in self.config_dir.glob("*.json"):
            configs.append(config_file.stem)
        return sorted(configs)
    
    def delete_config(self, name: str) -> bool:
        """删除配置文件"""
        try:
            config_file = self.config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
                # 从缓存中移除
                if name in self._config_cache:
                    del self._config_cache[name]
                self.logger.info(f"配置已删除: {name}")
                return True
            else:
                self.logger.warning(f"配置文件不存在: {name}")
                return False
        except Exception as e:
            error_msg = f"删除配置失败: {str(e)}"
            self.logger.error(error_msg)
            raise FileOperationError(error_msg, name, "delete")
    
    def create_default_configs(self):
        """创建默认配置模板"""
        from atscui.config.base_config import TrainingConfig, RunningConfig
        
        # 默认训练配置
        default_training = TrainingConfig(
            net_file="xgzd/net/xgzd.net.xml",
            rou_file="xgzd/net/xgzd-perhour.rou.xml",
            csv_path="outs/default-DQN",
            model_path="models/default-model-DQN.zip",
            eval_path="evals/default-eval-DQN.txt",
            predict_path="predicts/default-predict-DQN.json",
            algo_name="DQN",
            total_timesteps=100000,
            num_seconds=10000
        )
        
        # 默认运行配置
        default_running = RunningConfig(
            net_file="xgzd/net/xgzd.net.xml",
            rou_file="xgzd/net/xgzd-perhour.rou.xml",
            csv_path="outs/default-predict",
            model_path="models/default-model-DQN.zip",
            eval_path="evals/default-eval.txt",
            predict_path="predicts/default-predict.json",
            algo_name="DQN"
        )
        
        try:
            self.save_config(default_training, "default_training")
            self.save_config(default_running, "default_running")
            self.logger.info("默认配置模板已创建")
        except Exception as e:
            self.logger.error(f"创建默认配置失败: {e}")
    
    def get_config_template(self, config_type: str) -> Dict:
        """获取配置模板"""
        templates = {
            "training": {
                "net_file": "path/to/network.net.xml",
                "rou_file": "path/to/routes.rou.xml",
                "algo_name": "DQN",
                "total_timesteps": 100000,
                "num_seconds": 10000,
                "learning_rate": 0.001,
                "gamma": 0.9,
                "gui": False
            },
            "running": {
                "net_file": "path/to/network.net.xml",
                "rou_file": "path/to/routes.rou.xml",
                "model_path": "path/to/model.zip",
                "operation": "PREDICT",
                "gui": False,
                "num_seconds": 1000
            }
        }
        
        if config_type not in templates:
            raise ConfigurationError(f"未知的配置类型: {config_type}")
        
        return templates[config_type]
    
    def get_ui_config(self) -> Dict:
        """获取UI配置"""
        ui_config = {
            "title": "ATSC UI - 交通信号控制系统",
            "theme": "default",
            "port": 7860,
            "share": False,
            "debug": False,
            "show_api": False,
            "max_threads": 40,
            "auth": None,
            "auth_message": None,
            "root_path": None,
            "app_kwargs": {}
        }
        
        self.logger.info("UI配置已加载")
        return ui_config


# 全局配置管理器实例
config_manager = ConfigManager()