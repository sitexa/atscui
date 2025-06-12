import datetime
import json
import ntpath
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Union

# 导入新的文件管理模块
from atscui.utils.file_utils import (
    ensure_dir, save_json, load_json, write_evaluation, 
    write_eval_result, write_predict_result, write_loop_state,
    extract_crossname_from_netfile, extract_crossname_from_evalfile,
    file_manager
)
from atscui.exceptions import ValidationError
from atscui.logging_manager import get_logger


class UtilityManager:
    """工具管理器
    
    提供各种实用工具函数，包含验证、转换等功能。
    """
    
    def __init__(self):
        self.logger = get_logger('utility_manager')
    
    def change_file_extension(self, file_name: str, new_extension: str) -> str:
        """更改文件扩展名
        
        Args:
            file_name: 原文件名
            new_extension: 新扩展名（不包含点）
            
        Returns:
            str: 新文件名
        """
        try:
            base_name, _ = os.path.splitext(file_name)
            new_file_name = base_name + '.' + new_extension
            return new_file_name
        except Exception as e:
            self.logger.warning(f"更改文件扩展名失败: {e}")
            return file_name
    
    def validate_file_path(self, filepath: Union[str, Path], 
                          must_exist: bool = False) -> bool:
        """验证文件路径
        
        Args:
            filepath: 文件路径
            must_exist: 是否必须存在
            
        Returns:
            bool: 验证是否通过
            
        Raises:
            ValidationError: 验证失败时抛出
        """
        try:
            path = Path(filepath)
            
            # 检查路径是否为空
            if not str(filepath).strip():
                raise ValidationError("文件路径不能为空")
            
            # 检查是否为目录
            if path.is_dir():
                raise ValidationError(f"路径是目录而非文件: {filepath}")
            
            # 检查文件是否存在（如果要求）
            if must_exist and not path.exists():
                raise ValidationError(f"文件不存在: {filepath}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"验证文件路径失败: {e}")
            raise ValidationError(f"文件路径验证失败: {e}")
    
    def validate_directory_path(self, dirpath: Union[str, Path], 
                              must_exist: bool = False) -> bool:
        """验证目录路径
        
        Args:
            dirpath: 目录路径
            must_exist: 是否必须存在
            
        Returns:
            bool: 验证是否通过
            
        Raises:
            ValidationError: 验证失败时抛出
        """
        try:
            path = Path(dirpath)
            
            # 检查路径是否为空
            if not str(dirpath).strip():
                raise ValidationError("目录路径不能为空")
            
            # 检查是否为文件
            if path.is_file():
                raise ValidationError(f"路径是文件而非目录: {dirpath}")
            
            # 检查目录是否存在（如果要求）
            if must_exist and not path.exists():
                raise ValidationError(f"目录不存在: {dirpath}")
            
            return True
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"验证目录路径失败: {e}")
            raise ValidationError(f"目录路径验证失败: {e}")
    
    def normalize_path(self, path: Union[str, Path]) -> str:
        """标准化路径
        
        Args:
            path: 原始路径
            
        Returns:
            str: 标准化后的路径
        """
        try:
            return str(Path(path).resolve())
        except Exception as e:
            self.logger.warning(f"路径标准化失败: {e}")
            return str(path)
    
    def get_relative_path(self, path: Union[str, Path], 
                         base: Union[str, Path]) -> str:
        """获取相对路径
        
        Args:
            path: 目标路径
            base: 基础路径
            
        Returns:
            str: 相对路径
        """
        try:
            return str(Path(path).relative_to(Path(base)))
        except Exception as e:
            self.logger.warning(f"获取相对路径失败: {e}")
            return str(path)
    
    def format_timestamp(self, timestamp: float = None, 
                        format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """格式化时间戳
        
        Args:
            timestamp: 时间戳，默认为当前时间
            format_str: 格式字符串
            
        Returns:
            str: 格式化后的时间字符串
        """
        try:
            if timestamp is None:
                dt = datetime.datetime.now()
            else:
                dt = datetime.datetime.fromtimestamp(timestamp)
            return dt.strftime(format_str)
        except Exception as e:
            self.logger.warning(f"时间格式化失败: {e}")
            return str(timestamp or "unknown")
    
    def safe_dict_get(self, data: Dict[str, Any], key: str, 
                     default: Any = None, required: bool = False) -> Any:
        """安全获取字典值
        
        Args:
            data: 字典数据
            key: 键名
            default: 默认值
            required: 是否必需
            
        Returns:
            Any: 字典值
            
        Raises:
            ValidationError: 必需键不存在时抛出
        """
        try:
            if key in data:
                return data[key]
            elif required:
                raise ValidationError(f"必需的键不存在: {key}")
            else:
                return default
        except Exception as e:
            if required:
                self.logger.error(f"获取必需键失败: {e}")
                raise ValidationError(f"获取必需键失败: {key}")
            else:
                self.logger.warning(f"获取字典值失败: {e}")
                return default


# 全局工具管理器实例
utility_manager = UtilityManager()

# 为了向后兼容，提供common_utils别名
common_utils = utility_manager

# 保持向后兼容的便捷函数
def change_file_extension(file_name: str, new_extension: str) -> str:
    """更改文件扩展名的便捷函数"""
    return utility_manager.change_file_extension(file_name, new_extension)
