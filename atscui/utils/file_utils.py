"""文件操作工具模块

提供统一的文件操作接口，包含错误处理和日志记录。
"""

import datetime
import json
import ntpath
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Union

from atscui.exceptions import FileOperationError, ValidationError
from atscui.logging_manager import get_logger


class FileManager:
    """文件管理器
    
    提供统一的文件操作接口，包含创建、读取、写入、删除等功能。
    """
    
    def __init__(self):
        self.logger = get_logger('file_manager')
    
    def ensure_dir(self, directory: Union[str, Path]) -> str:
        """确保目录存在
        
        Args:
            directory: 目录路径
            
        Returns:
            str: 标准化的目录路径
            
        Raises:
            FileOperationError: 目录创建失败时抛出
        """
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"目录已确保存在: {path}")
            return str(path)
        except Exception as e:
            error_msg = f"创建目录失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(directory), "create_dir")
    
    def save_json(self, data: Dict[str, Any], filepath: Union[str, Path], 
                  indent: int = 2, ensure_ascii: bool = False) -> bool:
        """保存数据到JSON文件
        
        Args:
            data: 要保存的数据
            filepath: 文件路径
            indent: JSON缩进
            ensure_ascii: 是否确保ASCII编码
            
        Returns:
            bool: 保存是否成功
            
        Raises:
            FileOperationError: 保存失败时抛出
        """
        try:
            filepath = Path(filepath)
            # 确保目录存在
            self.ensure_dir(filepath.parent)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            
            self.logger.info(f"JSON文件已保存: {filepath}")
            return True
            
        except Exception as e:
            error_msg = f"保存JSON文件失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "save_json")
    
    def load_json(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """从JSON文件加载数据
        
        Args:
            filepath: 文件路径
            
        Returns:
            Dict[str, Any]: 加载的数据
            
        Raises:
            FileOperationError: 加载失败时抛出
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileOperationError(f"文件不存在", str(filepath), "load_json")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.debug(f"JSON文件已加载: {filepath}")
            return data
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON格式错误"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "load_json")
        except Exception as e:
            error_msg = f"加载JSON文件失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "load_json")
    
    def write_text_file(self, content: str, filepath: Union[str, Path], 
                       mode: str = 'w', encoding: str = 'utf-8') -> bool:
        """写入文本文件
        
        Args:
            content: 文件内容
            filepath: 文件路径
            mode: 写入模式 ('w', 'a')
            encoding: 文件编码
            
        Returns:
            bool: 写入是否成功
            
        Raises:
            FileOperationError: 写入失败时抛出
        """
        try:
            filepath = Path(filepath)
            # 确保目录存在
            self.ensure_dir(filepath.parent)
            
            with open(filepath, mode, encoding=encoding) as f:
                f.write(content)
            
            self.logger.debug(f"文本文件已写入: {filepath}")
            return True
            
        except Exception as e:
            error_msg = f"写入文本文件失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "write_text")
    
    def read_text_file(self, filepath: Union[str, Path], 
                      encoding: str = 'utf-8') -> str:
        """读取文本文件
        
        Args:
            filepath: 文件路径
            encoding: 文件编码
            
        Returns:
            str: 文件内容
            
        Raises:
            FileOperationError: 读取失败时抛出
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileOperationError(f"文件不存在", str(filepath), "read_text")
            
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            
            self.logger.debug(f"文本文件已读取: {filepath}")
            return content
            
        except Exception as e:
            error_msg = f"读取文本文件失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "read_text")
    
    def write_evaluation_results(self, results: Dict[str, Any], 
                               filepath: Union[str, Path]) -> bool:
        """写入评估结果
        
        Args:
            results: 评估结果字典
            filepath: 文件路径
            
        Returns:
            bool: 写入是否成功
        """
        try:
            content_lines = []
            for key, value in results.items():
                content_lines.append(f"{key}: {value}")
            
            content = "\n".join(content_lines) + "\n"
            return self.write_text_file(content, filepath)
            
        except Exception as e:
            error_msg = f"写入评估结果失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "write_eval")
    
    def write_eval_result_with_timestamp(self, mean: float, std: float, 
                                       filepath: Union[str, Path]) -> bool:
        """写入带时间戳的评估结果
        
        Args:
            mean: 平均奖励
            std: 标准差
            filepath: 文件路径
            
        Returns:
            bool: 写入是否成功
        """
        try:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{current_time}, {mean}, {std}\n"
            
            result = self.write_text_file(line, filepath, mode='a')
            self.logger.info(f"评估结果已写入: {filepath}")
            return result
            
        except Exception as e:
            error_msg = f"写入评估结果失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "write_eval_timestamp")
    
    def write_predict_results(self, data: List[Dict], 
                            filepath: Union[str, Path],
                            print_to_console: bool = False) -> bool:
        """写入预测结果
        
        Args:
            data: 预测数据列表
            filepath: 文件路径
            print_to_console: 是否打印到控制台
            
        Returns:
            bool: 写入是否成功
        """
        try:
            if print_to_console:
                self.logger.info(f"预测结果: {data}")
            
            return self.save_json(data, filepath)
            
        except Exception as e:
            error_msg = f"写入预测结果失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "write_predict")
    
    def write_loop_state(self, state_list: List[str], 
                        filepath: Union[str, Path]) -> bool:
        """写入循环状态信息
        
        Args:
            state_list: 状态信息列表
            filepath: 文件路径
            
        Returns:
            bool: 写入是否成功
        """
        try:
            # 将.json扩展名改为.txt
            filepath = Path(filepath)
            if filepath.suffix == '.json':
                filepath = filepath.with_suffix('.txt')
            
            content = "".join(state_list)
            return self.write_text_file(content, filepath)
            
        except Exception as e:
            error_msg = f"写入循环状态失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "write_loop_state")
    
    def extract_crossname_from_netfile(self, path: Union[str, Path]) -> str:
        """从网络文件路径提取路口名称
        
        Args:
            path: 网络文件路径
            
        Returns:
            str: 路口名称
        """
        try:
            # 使用 ntpath.basename 来处理 Windows 路径
            filename = ntpath.basename(str(path))
            # 分割文件名和扩展名
            name_parts = filename.split('.')
            # 返回第一个部分（基本文件名）
            return name_parts[0]
        except Exception as e:
            self.logger.warning(f"提取路口名称失败: {e}")
            return "unknown"
    
    def extract_crossname_from_evalfile(self, filename: str) -> str:
        """从评估文件名提取路口名称
        
        Args:
            filename: 评估文件名
            
        Returns:
            str: 路口名称，如果提取失败返回None
        """
        try:
            # 使用正则表达式匹配文件名模式
            match = re.match(r'(.*?)-eval-', filename)
            if match:
                return match.group(1)
            else:
                self.logger.warning(f"无法从文件名提取路口名称: {filename}")
                return None
        except Exception as e:
            self.logger.warning(f"提取路口名称失败: {e}")
            return None
    
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
    
    def delete_file(self, filepath: Union[str, Path]) -> bool:
        """删除文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 删除是否成功
            
        Raises:
            FileOperationError: 删除失败时抛出
        """
        try:
            filepath = Path(filepath)
            if filepath.exists():
                filepath.unlink()
                self.logger.info(f"文件已删除: {filepath}")
                return True
            else:
                self.logger.warning(f"文件不存在，无法删除: {filepath}")
                return False
                
        except Exception as e:
            error_msg = f"删除文件失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "delete")
    
    def file_exists(self, filepath: Union[str, Path]) -> bool:
        """检查文件是否存在
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 文件是否存在
        """
        return Path(filepath).exists()
    
    def get_file_size(self, filepath: Union[str, Path]) -> int:
        """获取文件大小
        
        Args:
            filepath: 文件路径
            
        Returns:
            int: 文件大小（字节）
            
        Raises:
            FileOperationError: 获取失败时抛出
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileOperationError(f"文件不存在", str(filepath), "get_size")
            
            return filepath.stat().st_size
            
        except Exception as e:
            error_msg = f"获取文件大小失败"
            self.logger.error(f"{error_msg}: {e}")
            raise FileOperationError(error_msg, str(filepath), "get_size")


# 全局文件管理器实例
file_manager = FileManager()


# 便捷函数
def ensure_dir(directory: Union[str, Path]) -> str:
    """确保目录存在的便捷函数"""
    return file_manager.ensure_dir(directory)


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """保存JSON文件的便捷函数"""
    return file_manager.save_json(data, filepath)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """加载JSON文件的便捷函数"""
    return file_manager.load_json(filepath)


def write_evaluation(results: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """写入评估结果的便捷函数"""
    return file_manager.write_evaluation_results(results, filepath)


def write_eval_result(mean: float, std: float, filepath: Union[str, Path]) -> bool:
    """写入评估结果的便捷函数"""
    return file_manager.write_eval_result_with_timestamp(mean, std, filepath)


def write_predict_result(data: List[Dict], filepath: Union[str, Path], 
                        print_to_console: bool = False) -> bool:
    """写入预测结果的便捷函数"""
    return file_manager.write_predict_results(data, filepath, print_to_console)


def write_loop_state(state_list: List[str], filepath: Union[str, Path]) -> bool:
    """写入循环状态的便捷函数"""
    return file_manager.write_loop_state(state_list, filepath)


def extract_crossname_from_netfile(path: Union[str, Path]) -> str:
    """从网络文件提取路口名称的便捷函数"""
    return file_manager.extract_crossname_from_netfile(path)


def extract_crossname_from_evalfile(filename: str) -> str:
    """从评估文件提取路口名称的便捷函数"""
    return file_manager.extract_crossname_from_evalfile(filename)