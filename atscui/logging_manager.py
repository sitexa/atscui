"""统一日志管理模块

提供项目的日志配置和管理功能，支持不同级别的日志记录。
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class LogManager:
    """日志管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.loggers = {}
            self._setup_default_logger()
            LogManager._initialized = True
    
    def _setup_default_logger(self):
        """设置默认日志配置"""
        # 创建日志目录
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建主日志记录器
        main_logger = logging.getLogger('atscui')
        main_logger.setLevel(logging.INFO)
        
        # 文件处理器
        log_file = log_dir / f"atscui_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        main_logger.addHandler(file_handler)
        main_logger.addHandler(console_handler)
        
        self.loggers['main'] = main_logger
    
    def get_logger(self, name: str = 'main') -> logging.Logger:
        """获取指定名称的日志记录器"""
        if name not in self.loggers:
            logger = logging.getLogger(f'atscui.{name}')
            logger.setLevel(logging.INFO)
            # 继承主日志记录器的处理器
            logger.parent = self.loggers['main']
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def create_training_logger(self, algorithm: str, cross_name: str) -> logging.Logger:
        """为训练过程创建专用日志记录器"""
        logger_name = f"training_{algorithm}_{cross_name}"
        
        if logger_name not in self.loggers:
            # 创建训练日志目录
            training_log_dir = Path("logs") / "training"
            training_log_dir.mkdir(exist_ok=True)
            
            # 创建训练专用日志记录器
            logger = logging.getLogger(f'atscui.{logger_name}')
            logger.setLevel(logging.DEBUG)
            
            # 训练日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # 训练日志文件处理器
            log_file = training_log_dir / f"{logger_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            
            # 控制台处理器 - 确保训练日志也能在控制台显示
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            self.loggers[logger_name] = logger
        
        return self.loggers[logger_name]
    
    def log_training_progress(self, logger_name: str, step: int, total_steps: int, 
                            metrics: dict = None):
        """记录训练进度"""
        logger = self.get_logger(logger_name)
        progress = (step / total_steps) * 100
        
        message = f"训练进度: {step}/{total_steps} ({progress:.1f}%)"
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            message += f" - {metrics_str}"
        
        logger.info(message)
    
    def log_error(self, logger_name: str, error: Exception, context: str = None):
        """记录错误信息"""
        logger = self.get_logger(logger_name)
        
        error_msg = f"错误: {str(error)}"
        if context:
            error_msg = f"{context} - {error_msg}"
        
        logger.error(error_msg, exc_info=True)
    
    def log_config(self, logger_name: str, config: dict):
        """记录配置信息"""
        logger = self.get_logger(logger_name)
        logger.info("配置信息:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
    
    def cleanup_old_logs(self, days: int = 7):
        """清理旧日志文件"""
        log_dir = Path("logs")
        if not log_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        
        for log_file in log_dir.rglob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    self.get_logger().info(f"删除旧日志文件: {log_file}")
                except Exception as e:
                    self.get_logger().warning(f"删除日志文件失败 {log_file}: {e}")


# 全局日志管理器实例
log_manager = LogManager()


def get_logger(name: str = 'main') -> logging.Logger:
    """获取日志记录器的便捷函数"""
    return log_manager.get_logger(name)