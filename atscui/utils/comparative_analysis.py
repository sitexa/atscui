#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比分析工具 - 集成到ATSCUI系统
基于visualization_tab.py的设计模式，提供PPO与固定周期的对比分析功能
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import traceback
import glob
import re

# 处理模块导入问题 - 当作为独立脚本运行时
try:
    from atscui.logging_manager import get_logger
    from atscui.utils.file_utils import FileManager
    from atscui.exceptions import ValidationError, FileOperationError
except ImportError:
    # 当作为独立脚本运行时，添加项目根目录到Python路径
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from atscui.logging_manager import get_logger
        from atscui.utils.file_utils import FileManager
        from atscui.exceptions import ValidationError, FileOperationError
    except ImportError:
        # 如果仍然无法导入，使用简化版本
        import logging
        
        def get_logger(name):
            logger = logging.getLogger(name)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            return logger
        
        class FileManager:
            @staticmethod
            def file_exists(path):
                return os.path.exists(path)
        
        class ValidationError(Exception):
            pass
        
        class FileOperationError(Exception):
            pass

# 设置中文字体 - 兼容多操作系统
def setup_chinese_fonts():
    """设置中文字体，智能适配不同操作系统"""
    import matplotlib.font_manager as fm
    import platform
    
    # 获取操作系统信息
    system = platform.system().lower()
    
    # 根据操作系统定义字体优先级
    if system == 'darwin':  # macOS
        font_list = [
            'Arial Unicode MS',  # macOS最佳中文支持
            'PingFang SC',       # macOS系统中文字体
            'Hiragino Sans GB',  # macOS中文字体
            'STHeiti',           # macOS中文字体
            'Arial',             # 英文字体
            'Helvetica',         # macOS英文字体
            'sans-serif'         # 系统默认
        ]
    elif system == 'linux':  # Ubuntu/Linux
        font_list = [
            'Noto Sans CJK SC',      # Linux最佳中文支持
            'WenQuanYi Micro Hei',   # Linux中文字体
            'WenQuanYi Zen Hei',     # Linux中文字体
            'Droid Sans Fallback',   # Android/Linux字体
            'Liberation Sans',       # Linux英文字体
            'DejaVu Sans',          # Linux英文字体
            'Ubuntu',               # Ubuntu字体
            'Noto Sans',            # 通用字体
            'Arial',                # 通用英文字体
            'sans-serif'            # 系统默认
        ]
    else:  # Windows或其他系统
        font_list = [
            'Microsoft YaHei',   # Windows中文字体
            'SimHei',           # Windows中文字体
            'Arial Unicode MS', # Windows中文支持
            'Arial',            # 英文字体
            'Helvetica',        # 英文字体
            'sans-serif'        # 系统默认
        ]
    
    # 检查可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 找到可用的字体
    selected_fonts = []
    for font in font_list:
        if font in available_fonts:
            selected_fonts.append(font)
    
    # 如果没有找到任何字体，添加通用备选
    if not selected_fonts:
        selected_fonts = ['DejaVu Sans', 'Arial', 'sans-serif']
        print(f"警告: 未找到系统推荐字体，使用通用字体: {selected_fonts[0]}")
        if system == 'linux':
            print("建议安装中文字体包: sudo apt-get install fonts-noto-cjk")
    else:
        print(f"检测到操作系统: {system.upper()}")
        print(f"使用字体: {selected_fonts[0]} (共找到 {len(selected_fonts)} 个可用字体)")
    
    # 设置matplotlib字体配置
    plt.rcParams['font.sans-serif'] = selected_fonts
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.family'] = ['sans-serif']  # 设置字体族
    
    # 额外的渲染设置，提高兼容性
    plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学文本字体
    
    return selected_fonts[0] if selected_fonts else 'sans-serif'

# 初始化中文字体设置
setup_chinese_fonts()

class ComparativeAnalyzer:
    """对比分析器 - 用于PPO与固定周期的性能对比"""
    
    def __init__(self):
        self.logger = get_logger('comparative_analyzer')
        self.file_manager = FileManager()
        
    def analyze_training_comparison(self, sample_file_path: str) -> Tuple[Optional[str], str]:
        """
        基于样本文件路径分析同路口多算法训练日志的对比
        
        Args:
            sample_file_path: 样本文件路径（如zfdx-PPO_conn0_ep1.csv）
            
        Returns:
            Tuple[Optional[str], str]: (分析结果图片路径, 输出信息)
        """
        try:
            # 1. 验证输入文件
            if not sample_file_path:
                return None, "❌ 请提供样本文件路径"
                
            if not self.file_manager.file_exists(sample_file_path):
                return None, f"❌ 样本文件不存在: {sample_file_path}"
            
            # 2. 解析路口名称和目录
            base_dir = os.path.dirname(sample_file_path)
            filename = os.path.basename(sample_file_path)
            
            # 提取路口名称（如zfdx）
            match = re.match(r'([^-]+)-', filename)
            if not match:
                return None, f"❌ 无法从文件名提取路口名称: {filename}"
            
            intersection_name = match.group(1)
            self.logger.info(f"检测到路口名称: {intersection_name}")
            
            # 3. 查找所有相关文件
            file_groups = self._find_algorithm_files(base_dir, intersection_name)
            
            if not any(file_groups.values()):
                return None, f"❌ 未找到路口 {intersection_name} 的任何算法日志文件"
            
            # 4. 加载和统计数据
            self.logger.info("开始加载和统计训练数据...")
            algorithm_stats = self._load_and_aggregate_data(file_groups)
            
            # 5. 执行对比分析
            analysis_results = self._perform_multi_algorithm_analysis(algorithm_stats, intersection_name)
            
            # 6. 生成可视化图表
            plot_path = os.path.join(base_dir, f"{intersection_name}_multi_algorithm_analysis.png")
            self._create_multi_algorithm_plots(algorithm_stats, analysis_results, plot_path)
            
            # 7. 生成分析报告
            report_path = os.path.join(base_dir, f"{intersection_name}_multi_algorithm_report.md")
            self._generate_multi_algorithm_report(algorithm_stats, analysis_results, report_path)
            
            success_msg = f"✅ 多算法对比分析完成\n" \
                         f"🚦 路口: {intersection_name}\n" \
                         f"📊 可视化图表: {plot_path}\n" \
                         f"📄 分析报告: {report_path}\n" \
                         f"🔍 分析算法: {', '.join(algorithm_stats.keys())}"
            
            self.logger.info("多算法对比分析成功完成")
            return plot_path, success_msg
            
        except ValidationError as e:
            error_msg = f"❌ 输入验证失败: {e}"
            self.logger.warning(error_msg)
            return None, error_msg
            
        except FileOperationError as e:
            error_msg = f"❌ 文件操作失败: {e}"
            self.logger.error(error_msg)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"❌ 对比分析时发生错误: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None, error_msg
    
    def _find_algorithm_files(self, base_dir: str, intersection_name: str) -> Dict[str, List[str]]:
        """查找指定路口的所有算法日志文件"""
        # 定义支持的算法类型
        rl_algorithms = ['PPO', 'DQN', 'A2C', 'SAC']
        fixtime_algorithms = ['FIXTIME-curriculum', 'FIXTIME-static']
        
        file_groups = {}
        
        # 初始化文件组
        for alg in rl_algorithms + fixtime_algorithms:
            file_groups[alg] = []
        
        # 强化学习算法文件模式: 路口-算法_conn0_ep*.csv
        for algorithm in rl_algorithms:
            pattern = os.path.join(base_dir, f"{intersection_name}-{algorithm}_conn0_ep*.csv")
            file_groups[algorithm] = glob.glob(pattern)
        
        # fixtime算法文件模式: 路口-fixtime-类型-*.csv
        for algorithm in fixtime_algorithms:
            pattern = os.path.join(base_dir, f"{intersection_name}-{algorithm}-*.csv")
            file_groups[algorithm] = glob.glob(pattern)
        
        # 记录找到的文件数量
        found_info = []
        for alg, files in file_groups.items():
            if files:
                found_info.append(f"{alg}({len(files)})")
        
        self.logger.info(f"找到文件: {', '.join(found_info) if found_info else '无'}")
        
        return file_groups
    
    def _load_and_aggregate_data(self, file_groups: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """加载并聚合各算法的数据"""
        algorithm_stats = {}
        
        for algorithm, files in file_groups.items():
            if not files:
                continue
                
            all_data = []
            for file_path in files:
                try:
                    data = pd.read_csv(file_path)
                    # 添加文件标识
                    data['source_file'] = os.path.basename(file_path)
                    all_data.append(data)
                    self.logger.info(f"加载 {algorithm} 文件: {os.path.basename(file_path)} ({len(data)}条记录)")
                except Exception as e:
                    self.logger.warning(f"加载文件失败 {file_path}: {e}")
                    continue
            
            if all_data:
                # 合并所有数据
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # 计算统计指标
                stats = self._calculate_algorithm_statistics(combined_data, algorithm)
                algorithm_stats[algorithm] = stats
                
                self.logger.info(f"{algorithm} 总计: {len(combined_data)}条记录, {len(all_data)}个文件")
        
        return algorithm_stats
    
    def _calculate_algorithm_statistics(self, data: pd.DataFrame, algorithm: str) -> Dict:
        """计算单个算法的统计指标"""
        stats = {
            'raw_data': data,
            'total_records': len(data),
            'file_count': data['source_file'].nunique(),
            'metrics': {}
        }
        
        # 定义关键指标
        key_metrics = [
            'system_mean_waiting_time',
            'system_mean_speed', 
            'system_total_throughput',
            'system_mean_travel_time',
            'system_total_fuel_consumption',
            'system_total_co2_emission'
        ]
        
        # 对于强化学习算法，使用后期阶段的平均值
        rl_algorithms = ['PPO', 'DQN', 'A2C', 'SAC']
        if algorithm in rl_algorithms and len(data) > 100:
            # 使用最后1/4的数据作为稳定期性能
            stable_data = data.iloc[-len(data)//4:]
        else:
            stable_data = data
        
        # 计算各指标的统计值
        for metric in key_metrics:
            if metric in stable_data.columns:
                if metric == 'system_total_throughput':
                    # 对于总通行量，计算平均通行量（假设每条记录代表一个时间步长）
                    # 假设每个时间步长为1秒，转换为每小时通行量
                    # 如果数据中有时间信息，可以更精确计算
                    avg_throughput_per_step = stable_data[metric].mean()
                    # 假设每步为1秒，转换为每小时（3600秒）
                    hourly_throughput = avg_throughput_per_step * 3600 / len(stable_data) if len(stable_data) > 0 else 0
                    
                    stats['metrics'][metric] = {
                        'mean': hourly_throughput,  # 使用每小时平均通行量
                        'std': stable_data[metric].std() * 3600 / len(stable_data) if len(stable_data) > 0 else 0,
                        'min': stable_data[metric].min(),
                        'max': stable_data[metric].max(),
                        'median': stable_data[metric].median()
                    }
                    # 同时保存原始总通行量用于参考
                    stats['metrics']['system_total_throughput_raw'] = {
                        'mean': stable_data[metric].mean(),
                        'std': stable_data[metric].std(),
                        'min': stable_data[metric].min(),
                        'max': stable_data[metric].max(),
                        'median': stable_data[metric].median()
                    }
                else:
                    stats['metrics'][metric] = {
                        'mean': stable_data[metric].mean(),
                        'std': stable_data[metric].std(),
                        'min': stable_data[metric].min(),
                        'max': stable_data[metric].max(),
                        'median': stable_data[metric].median()
                    }
        
        # 兼容旧格式的列名映射
        legacy_mapping = {
            'avg_waiting_time': 'system_mean_waiting_time',
            'avg_speed': 'system_mean_speed',
            'total_throughput': 'system_total_throughput',
            'avg_travel_time': 'system_mean_travel_time',
            'total_fuel_consumption': 'system_total_fuel_consumption',
            'total_co2_emission': 'system_total_co2_emission'
        }
        
        for legacy_col, standard_col in legacy_mapping.items():
            if legacy_col in stable_data.columns and standard_col not in stats['metrics']:
                stats['metrics'][standard_col] = {
                    'mean': stable_data[legacy_col].mean(),
                    'std': stable_data[legacy_col].std(),
                    'min': stable_data[legacy_col].min(),
                    'max': stable_data[legacy_col].max(),
                    'median': stable_data[legacy_col].median()
                }
        
        return stats
    
    def _perform_multi_algorithm_analysis(self, algorithm_stats: Dict, intersection_name: str) -> Dict:
        """执行多算法对比分析"""
        analysis_results = {
            'intersection_name': intersection_name,
            'algorithms': list(algorithm_stats.keys()),
            'comparison_matrix': {},
            'best_performers': {},
            'improvement_analysis': {}
        }
        
        # 定义指标和优化方向（True表示越大越好，False表示越小越好）
        metrics_direction = {
            'system_mean_waiting_time': False,  # 等待时间越小越好
            'system_mean_speed': True,          # 速度越大越好
            'system_total_throughput': True,    # 平均通行量越大越好
            'system_mean_travel_time': False,   # 行程时间越小越好
            'system_total_fuel_consumption': False,  # 燃油消耗越小越好
            'system_total_co2_emission': False       # CO2排放越小越好
        }
        
        # 构建对比矩阵
        for metric, is_higher_better in metrics_direction.items():
            metric_comparison = {}
            metric_values = {}
            
            # 收集各算法在该指标上的表现
            for algorithm, stats in algorithm_stats.items():
                if metric in stats['metrics']:
                    metric_values[algorithm] = stats['metrics'][metric]['mean']
            
            if len(metric_values) > 1:
                # 找出最佳表现
                if is_higher_better:
                    best_algorithm = max(metric_values, key=metric_values.get)
                    best_value = max(metric_values.values())
                else:
                    best_algorithm = min(metric_values, key=metric_values.get)
                    best_value = min(metric_values.values())
                
                analysis_results['best_performers'][metric] = {
                    'algorithm': best_algorithm,
                    'value': best_value
                }
                
                # 计算相对改善率
                for algorithm, value in metric_values.items():
                    if algorithm != best_algorithm:
                        if is_higher_better:
                            improvement = (best_value - value) / value * 100
                        else:
                            improvement = (value - best_value) / value * 100
                        
                        metric_comparison[f"{best_algorithm}_vs_{algorithm}"] = {
                            'improvement_rate': improvement,
                            'best_value': best_value,
                            'compared_value': value
                        }
            
            analysis_results['comparison_matrix'][metric] = {
                'values': metric_values,
                'comparisons': metric_comparison
            }
        
        # 计算综合评分
        self._calculate_overall_scores(analysis_results, algorithm_stats)
        
        return analysis_results
    
    def _calculate_overall_scores(self, analysis_results: Dict, algorithm_stats: Dict):
        """计算综合评分"""
        algorithms = analysis_results['algorithms']
        overall_scores = {alg: 0 for alg in algorithms}
        
        # 为每个指标分配权重
        metric_weights = {
            'system_mean_waiting_time': 0.25,    # 平均等待时间
            'system_mean_speed': 0.20,           # 平均速度
            'system_total_throughput': 0.20,     # 平均通行量（每小时）
            'system_mean_travel_time': 0.15,     # 平均行程时间
            'system_total_fuel_consumption': 0.10, # 总燃油消耗
            'system_total_co2_emission': 0.10    # 总CO2排放
        }
        
        for metric, weight in metric_weights.items():
            if metric in analysis_results['best_performers']:
                best_alg = analysis_results['best_performers'][metric]['algorithm']
                overall_scores[best_alg] += weight
        
        analysis_results['overall_scores'] = overall_scores
        analysis_results['best_overall'] = max(overall_scores, key=overall_scores.get)
    
    def _perform_comparative_analysis(self, ppo_data: pd.DataFrame, fixtime_data: pd.DataFrame) -> dict:
        """执行对比分析计算"""
        # 获取PPO最终性能（后期阶段平均值）
        ppo_final = ppo_data.iloc[-len(ppo_data)//4:]  # 最后1/4数据
        fixtime_result = fixtime_data.iloc[0]  # 固定周期结果
        
        # 定义对比指标映射
        comparison_metrics = {
            '平均等待时间': ('avg_waiting_time', 'system_mean_waiting_time'),
            '平均速度': ('avg_speed', 'system_mean_speed'),
            '总通行量': ('total_throughput', 'system_total_throughput'),
            '平均行程时间': ('avg_travel_time', 'system_mean_travel_time'),
            '总燃油消耗': ('total_fuel_consumption', 'system_total_fuel_consumption'),
            '总CO2排放': ('total_co2_emission', 'system_total_co2_emission')
        }
        
        improvements = {}
        comparison_data = {}
        
        for metric_name, (fixtime_col, ppo_col) in comparison_metrics.items():
            if fixtime_col in fixtime_data.columns and ppo_col in ppo_final.columns:
                fixtime_val = fixtime_result[fixtime_col]
                ppo_val = ppo_final[ppo_col].mean()
                
                # 计算改善率
                if metric_name in ['平均等待时间', '平均行程时间', '总燃油消耗', '总CO2排放']:
                    improvement = (fixtime_val - ppo_val) / fixtime_val * 100
                else:
                    improvement = (ppo_val - fixtime_val) / fixtime_val * 100
                
                improvements[metric_name] = improvement
                comparison_data[metric_name] = {
                    'fixtime': fixtime_val,
                    'ppo': ppo_val,
                    'improvement': improvement
                }
        
        # 分析PPO训练过程
        total_steps = len(ppo_data)
        training_phases = {
            '初期': ppo_data.iloc[:total_steps//3],
            '中期': ppo_data.iloc[total_steps//3:2*total_steps//3],
            '后期': ppo_data.iloc[2*total_steps//3:]
        }
        
        phase_analysis = {}
        for phase_name, phase_data in training_phases.items():
            phase_analysis[phase_name] = {
                'avg_waiting': phase_data['system_mean_waiting_time'].mean(),
                'avg_speed': phase_data['system_mean_speed'].mean(),
                'avg_throughput': phase_data['system_total_throughput'].mean()
            }
        
        return {
            'improvements': improvements,
            'comparison_data': comparison_data,
            'phase_analysis': phase_analysis,
            'avg_improvement': np.mean(list(improvements.values())),
            'ppo_stats': {
                'total_records': len(ppo_data),
                'total_steps': ppo_data['step'].max(),
                'step_interval': ppo_data['step'].iloc[1] - ppo_data['step'].iloc[0]
            },
            'fixtime_stats': {
                'total_records': len(fixtime_data),
                'total_steps': fixtime_data['total_steps'].iloc[0] if 'total_steps' in fixtime_data.columns else 'N/A'
            }
        }
    
    def _create_multi_algorithm_plots(self, algorithm_stats: Dict, analysis_results: Dict, output_path: str):
        """创建多算法对比分析图表"""
        algorithms = list(algorithm_stats.keys())
        n_algorithms = len(algorithms)
        
        if n_algorithms == 0:
            self.logger.warning("没有算法数据可供绘图")
            return
        
        # 设置图表布局
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # 定义颜色映射
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        algorithm_colors = {alg: colors[i % len(colors)] for i, alg in enumerate(algorithms)}
        
        # 1. 关键指标对比柱状图 (上排)
        key_metrics = [
            ('system_mean_waiting_time', '平均等待时间(秒)'),
            ('system_mean_speed', '平均速度(m/s)'),
            ('system_total_throughput', '平均通行量(辆/小时)')
        ]
        
        for i, (metric, title) in enumerate(key_metrics):
            ax = fig.add_subplot(gs[0, i])
            
            values = []
            labels = []
            bar_colors = []
            
            for algorithm in algorithms:
                if metric in algorithm_stats[algorithm]['metrics']:
                    values.append(algorithm_stats[algorithm]['metrics'][metric]['mean'])
                    labels.append(algorithm)
                    bar_colors.append(algorithm_colors[algorithm])
            
            if values:
                bars = ax.bar(labels, values, color=bar_colors, alpha=0.7)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
                
                # 标记最佳表现
                if metric in analysis_results['best_performers']:
                    best_alg = analysis_results['best_performers'][metric]['algorithm']
                    best_idx = labels.index(best_alg) if best_alg in labels else -1
                    if best_idx >= 0:
                        bars[best_idx].set_edgecolor('gold')
                        bars[best_idx].set_linewidth(3)
                
                ax.set_title(f'{title}对比', fontweight='bold', fontsize=12)
                ax.set_ylabel(title)
                ax.tick_params(axis='x', rotation=45)
        
        # 2. 综合评分雷达图 (中排左)
        if len(algorithms) > 1:
            ax = fig.add_subplot(gs[1, 0])
            overall_scores = analysis_results['overall_scores']
            
            algorithms_list = list(overall_scores.keys())
            scores = list(overall_scores.values())
            
            bars = ax.bar(algorithms_list, scores, 
                         color=[algorithm_colors[alg] for alg in algorithms_list], alpha=0.7)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('综合评分对比', fontweight='bold', fontsize=12)
            ax.set_ylabel('综合评分')
            ax.tick_params(axis='x', rotation=45)
        
        # 3. 各指标最佳表现者 (中排中)
        ax = fig.add_subplot(gs[1, 1])
        ax.axis('off')
        
        best_performers_text = "各指标最佳表现:\n\n"
        for metric, performer in analysis_results['best_performers'].items():
            metric_name = {
                'system_mean_waiting_time': '等待时间',
                'system_mean_speed': '平均速度',
                'system_total_throughput': '平均通行量',
                'system_mean_travel_time': '行程时间',
                'system_total_fuel_consumption': '燃油消耗',
                'system_total_co2_emission': 'CO2排放'
            }.get(metric, metric)
            
            best_performers_text += f"🏆 {metric_name}: {performer['algorithm']}\n"
        
        ax.text(0.1, 0.9, best_performers_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        
        # 4. 数据统计信息 (中排右)
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')
        
        stats_text = "数据统计信息:\n\n"
        for algorithm, stats in algorithm_stats.items():
            stats_text += f"{algorithm}:\n"
            stats_text += f"  文件数: {stats['file_count']}\n"
            stats_text += f"  记录数: {stats['total_records']}\n\n"
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 5. 改善率对比 (下排)
        improvement_metrics = ['system_mean_waiting_time', 'system_mean_speed', 'system_total_throughput']
        
        for i, metric in enumerate(improvement_metrics):
            if i >= 3:  # 最多显示3个指标
                break
                
            ax = fig.add_subplot(gs[2, i])
            
            if metric in analysis_results['comparison_matrix']:
                comparisons = analysis_results['comparison_matrix'][metric]['comparisons']
                
                if comparisons:
                    comparison_names = []
                    improvement_rates = []
                    
                    for comp_name, comp_data in comparisons.items():
                        comparison_names.append(comp_name.replace('_vs_', ' vs '))
                        improvement_rates.append(comp_data['improvement_rate'])
                    
                    colors_list = ['green' if rate > 0 else 'red' for rate in improvement_rates]
                    bars = ax.barh(comparison_names, improvement_rates, color=colors_list, alpha=0.7)
                    
                    for i, (bar, rate) in enumerate(zip(bars, improvement_rates)):
                        # 避免使用+号格式化，防止Ubuntu系统中的符号乱码
                        rate_text = f'{rate:.1f}%' if rate >= 0 else f'{rate:.1f}%'
                        if rate > 0:
                            rate_text = f'+{rate:.1f}%'
                        ax.text(rate + (2 if rate > 0 else -2), i, rate_text, 
                               va='center', ha='left' if rate > 0 else 'right')
                    
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    
                    metric_name = {
                        'system_mean_waiting_time': '等待时间',
                        'system_mean_speed': '平均速度',
                        'system_total_throughput': '平均通行量'
                    }.get(metric, metric)
                    
                    ax.set_title(f'{metric_name}改善率', fontweight='bold', fontsize=12)
                    ax.set_xlabel('改善率 (%)')
        
        plt.suptitle(f'{analysis_results["intersection_name"]}路口 - 多算法性能对比分析', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"多算法对比分析图表已保存: {output_path}")
    
    def _create_comparison_plots(self, ppo_data: pd.DataFrame, fixtime_data: pd.DataFrame, 
                               analysis_results: dict, output_path: str):
        """创建对比分析图表"""
        fig = plt.figure(figsize=(20, 16))
        
        # 创建网格布局
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # 1. PPO训练过程曲线 (上排)
        training_metrics = [
            ('system_mean_waiting_time', '平均等待时间(秒)', 'red'),
            ('system_mean_speed', '平均速度(m/s)', 'blue'),
            ('system_total_throughput', '总通行量(辆)', 'green')
        ]
        
        for i, (metric, title, color) in enumerate(training_metrics):
            ax = fig.add_subplot(gs[0, i])
            if metric in ppo_data.columns:
                # 原始数据
                ax.plot(ppo_data['step'], ppo_data[metric], alpha=0.3, color=color, linewidth=0.5)
                
                # 移动平均
                window = max(1, len(ppo_data) // 20)
                moving_avg = ppo_data[metric].rolling(window=window, center=True).mean()
                ax.plot(ppo_data['step'], moving_avg, color=color, linewidth=2, label='PPO训练曲线')
                
                # 固定周期基准线
                fixtime_mapping = {
                    'system_mean_waiting_time': 'avg_waiting_time',
                    'system_mean_speed': 'avg_speed',
                    'system_total_throughput': 'total_throughput'
                }
                
                if metric in fixtime_mapping and fixtime_mapping[metric] in fixtime_data.columns:
                    fixtime_val = fixtime_data[fixtime_mapping[metric]].iloc[0]
                    ax.axhline(y=fixtime_val, color='red', linestyle='--', alpha=0.7, label='固定周期基准')
                
                ax.set_title(f'PPO训练过程 - {title}', fontweight='bold', fontsize=12)
                ax.set_xlabel('训练步数')
                ax.set_ylabel(title)
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # 2. 性能对比柱状图 (中排)
        comparison_metrics = ['平均等待时间', '平均速度', '总通行量']
        for i, metric in enumerate(comparison_metrics):
            if metric in analysis_results['comparison_data']:
                ax = fig.add_subplot(gs[1, i])
                data = analysis_results['comparison_data'][metric]
                
                bars = ax.bar(['固定周期', 'PPO算法'], [data['fixtime'], data['ppo']], 
                             color=['orange', 'skyblue'], alpha=0.7)
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom')
                
                # 添加改善率标注
                improvement = data['improvement']
                ax.text(0.5, max(data['fixtime'], data['ppo']) * 1.1, 
                       f'改善率: {improvement:+.1f}%', 
                       ha='center', va='bottom', fontweight='bold',
                       color='green' if improvement > 0 else 'red')
                
                ax.set_title(f'{metric}对比', fontweight='bold', fontsize=12)
                ax.set_ylabel(metric)
        
        # 3. 训练阶段分析 (下排左)
        ax = fig.add_subplot(gs[2, 0])
        phases = list(analysis_results['phase_analysis'].keys())
        waiting_times = [analysis_results['phase_analysis'][p]['avg_waiting'] for p in phases]
        
        bars = ax.bar(phases, waiting_times, color=['lightcoral', 'gold', 'lightgreen'], alpha=0.7)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
        
        ax.set_title('PPO训练阶段 - 等待时间变化', fontweight='bold', fontsize=12)
        ax.set_ylabel('平均等待时间(秒)')
        
        # 4. 改善率雷达图 (下排中)
        ax = fig.add_subplot(gs[2, 1])
        metrics = list(analysis_results['improvements'].keys())
        improvements = list(analysis_results['improvements'].values())
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.barh(metrics, improvements, color=colors, alpha=0.7)
        
        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax.text(imp + (5 if imp > 0 else -5), i, f'{imp:+.1f}%', 
                   va='center', ha='left' if imp > 0 else 'right')
        
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('各指标改善率对比', fontweight='bold', fontsize=12)
        ax.set_xlabel('改善率 (%)')
        
        # 5. 综合评价 (下排右)
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        
        # 统计信息
        positive_count = sum(1 for imp in improvements if imp > 0)
        total_count = len(improvements)
        avg_improvement = analysis_results['avg_improvement']
        
        summary_text = f"""
综合评价结果

总体改善率: {avg_improvement:+.1f}%

PPO优势指标: {positive_count}/{total_count}

训练数据量: {analysis_results['ppo_stats']['total_records']}条
训练总步数: {analysis_results['ppo_stats']['total_steps']:.0f}步

结论: {'PPO算法显著优于固定周期' if avg_improvement > 20 else 'PPO算法略优于固定周期' if avg_improvement > 0 else '固定周期略优于PPO算法'}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('PPO算法 vs 固定周期信号控制 - 综合对比分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"对比分析图表已保存: {output_path}")
    
    def _generate_multi_algorithm_report(self, algorithm_stats: Dict, analysis_results: Dict, output_path: str):
        """生成多算法详细分析报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {analysis_results['intersection_name']}路口多算法性能对比分析报告\n\n")
            
            # 1. 概览
            f.write("## 1. 分析概览\n\n")
            f.write(f"- 路口名称: {analysis_results['intersection_name']}\n")
            f.write(f"- 分析算法: {', '.join(analysis_results['algorithms'])}\n")
            f.write(f"- 综合最佳算法: {analysis_results['best_overall']}\n\n")
            
            # 2. 数据统计
            f.write("## 2. 数据统计\n\n")
            f.write("| 算法 | 文件数 | 记录数 | 数据来源 |\n")
            f.write("|------|--------|--------|----------|\n")
            
            for algorithm, stats in algorithm_stats.items():
                source_files = stats['raw_data']['source_file'].unique()
                f.write(f"| {algorithm} | {stats['file_count']} | {stats['total_records']} | {', '.join(source_files[:3])}{'...' if len(source_files) > 3 else ''} |\n")
            
            f.write("\n")
            
            # 3. 关键指标对比
            f.write("## 3. 关键指标对比\n\n")
            
            metrics_info = {
                'system_mean_waiting_time': ('平均等待时间', '秒', False),
                'system_mean_speed': ('平均速度', 'm/s', True),
                'system_total_throughput': ('平均通行量', '辆/小时', True),
                'system_mean_travel_time': ('平均行程时间', '秒', False),
                'system_total_fuel_consumption': ('总燃油消耗', 'L', False),
                'system_total_co2_emission': ('总CO2排放', 'g', False)
            }
            
            for metric, (name, unit, is_higher_better) in metrics_info.items():
                if metric in analysis_results['comparison_matrix']:
                    f.write(f"### {name}\n\n")
                    
                    values = analysis_results['comparison_matrix'][metric]['values']
                    if values:
                        f.write("| 算法 | 数值 | 排名 |\n")
                        f.write("|------|------|------|\n")
                        
                        # 排序
                        sorted_values = sorted(values.items(), 
                                             key=lambda x: x[1], 
                                             reverse=is_higher_better)
                        
                        for rank, (algorithm, value) in enumerate(sorted_values, 1):
                            rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
                            f.write(f"| {algorithm} | {value:.2f} {unit} | {rank_emoji} |\n")
                        
                        f.write("\n")
                        
                        # 改善率分析
                        comparisons = analysis_results['comparison_matrix'][metric]['comparisons']
                        if comparisons:
                            f.write("**改善率分析:**\n\n")
                            for comp_name, comp_data in comparisons.items():
                                best_alg, compared_alg = comp_name.split('_vs_')
                                improvement = comp_data['improvement_rate']
                                f.write(f"- {best_alg} 相比 {compared_alg}: {improvement:+.1f}%\n")
                            f.write("\n")
            
            # 4. 综合评分
            f.write("## 4. 综合评分\n\n")
            f.write("基于加权评分系统的综合性能排名:\n\n")
            f.write("| 排名 | 算法 | 综合评分 | 优势指标 |\n")
            f.write("|------|------|----------|----------|\n")
            
            sorted_scores = sorted(analysis_results['overall_scores'].items(), 
                                 key=lambda x: x[1], reverse=True)
            
            for rank, (algorithm, score) in enumerate(sorted_scores, 1):
                # 找出该算法的优势指标
                advantages = []
                for metric, performer in analysis_results['best_performers'].items():
                    if performer['algorithm'] == algorithm:
                        metric_name = metrics_info.get(metric, (metric, '', True))[0]
                        advantages.append(metric_name)
                
                rank_emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}"
                f.write(f"| {rank_emoji} | {algorithm} | {score:.3f} | {', '.join(advantages) if advantages else '无'} |\n")
            
            f.write("\n")
            
            # 5. 详细统计信息
            f.write("## 5. 详细统计信息\n\n")
            
            for algorithm, stats in algorithm_stats.items():
                f.write(f"### {algorithm}算法\n\n")
                f.write(f"- 数据文件: {stats['file_count']}个\n")
                f.write(f"- 总记录数: {stats['total_records']}条\n\n")
                
                f.write("**指标统计:**\n\n")
                f.write("| 指标 | 均值 | 标准差 | 最小值 | 最大值 | 中位数 |\n")
                f.write("|------|------|--------|--------|--------|--------|\n")
                
                for metric, metric_stats in stats['metrics'].items():
                    metric_name = metrics_info.get(metric, (metric, '', True))[0]
                    f.write(f"| {metric_name} | {metric_stats['mean']:.2f} | {metric_stats['std']:.2f} | {metric_stats['min']:.2f} | {metric_stats['max']:.2f} | {metric_stats['median']:.2f} |\n")
                
                f.write("\n")
            
            # 6. 结论与建议
            f.write("## 6. 结论与建议\n\n")
            
            best_algorithm = analysis_results['best_overall']
            best_score = analysis_results['overall_scores'][best_algorithm]
            
            if best_score > 0.6:
                conclusion = f"{best_algorithm}算法在综合评估中表现卓越，建议优先采用。"
            elif best_score > 0.4:
                conclusion = f"{best_algorithm}算法在综合评估中表现良好，可以考虑采用。"
            else:
                conclusion = "各算法表现相近，建议根据具体场景需求选择。"
            
            f.write(f"**主要结论:** {conclusion}\n\n")
            
            # 具体建议
            f.write("**具体建议:**\n\n")
            for metric, performer in analysis_results['best_performers'].items():
                metric_name = metrics_info.get(metric, (metric, '', True))[0]
                f.write(f"- 若优先考虑{metric_name}，推荐使用{performer['algorithm']}算法\n")
            
            f.write("\n")
            f.write("**注意事项:**\n\n")
            f.write("- 本分析基于历史训练数据，实际部署效果可能因环境变化而有所差异\n")
            f.write("- 建议在实际应用前进行小规模试点验证\n")
            f.write("- 定期监控和评估算法性能，必要时进行调优\n")
        
        self.logger.info(f"多算法分析报告已保存: {output_path}")
    
    def analyze_training_comparison_legacy(self, ppo_file: str, fixtime_file: str) -> Tuple[Optional[str], str]:
        """
        旧版本的训练对比分析方法（保持向后兼容性）
        
        Args:
            ppo_file: PPO训练日志文件路径
            fixtime_file: 固定周期结果文件路径
            
        Returns:
            Tuple[Optional[str], str]: (分析结果图片路径, 输出信息)
        """
        try:
            # 验证输入文件
            if not ppo_file or not fixtime_file:
                return None, "❌ 请提供PPO和固定周期文件路径"
                
            if not self.file_manager.file_exists(ppo_file):
                return None, f"❌ PPO文件不存在: {ppo_file}"
                
            if not self.file_manager.file_exists(fixtime_file):
                return None, f"❌ 固定周期文件不存在: {fixtime_file}"
            
            # 加载数据
            self.logger.info("开始加载训练数据...")
            ppo_data = pd.read_csv(ppo_file)
            fixtime_data = pd.read_csv(fixtime_file)
            
            # 执行对比分析
            analysis_results = self._perform_comparative_analysis(ppo_data, fixtime_data)
            
            # 生成可视化图表
            output_dir = os.path.dirname(ppo_file)
            plot_path = os.path.join(output_dir, "comparative_analysis.png")
            self._create_comparison_plots(ppo_data, fixtime_data, analysis_results, plot_path)
            
            # 生成分析报告
            report_path = os.path.join(output_dir, "comparative_analysis_report.md")
            self._generate_analysis_report(analysis_results, report_path)
            
            # 输出结果摘要
            success_msg = f"✅ 对比分析完成\n" \
                         f"📊 PPO训练记录: {analysis_results['ppo_stats']['total_records']}条\n" \
                         f"📊 固定周期记录: {analysis_results['fixtime_stats']['total_records']}条\n" \
                         f"📈 平均改善率: {analysis_results['avg_improvement']:+.1f}%\n" \
                         f"🖼️ 可视化图表: {plot_path}\n" \
                         f"📄 分析报告: {report_path}"
            
            self.logger.info("对比分析成功完成")
            return plot_path, success_msg
            
        except ValidationError as e:
            error_msg = f"❌ 输入验证失败: {e}"
            self.logger.warning(error_msg)
            return None, error_msg
            
        except FileOperationError as e:
            error_msg = f"❌ 文件操作失败: {e}"
            self.logger.error(error_msg)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"❌ 对比分析时发生错误: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None, error_msg
    
    def _generate_analysis_report(self, analysis_results: dict, output_path: str):
        """生成详细分析报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# PPO算法与固定周期信号控制对比分析报告\n\n")
            
            # 数据概览
            f.write("## 1. 数据概览\n\n")
            f.write(f"- PPO训练数据: {analysis_results['ppo_stats']['total_records']}条记录\n")
            f.write(f"- 训练总步数: {analysis_results['ppo_stats']['total_steps']:.0f}步\n")
            f.write(f"- 固定周期数据: {analysis_results['fixtime_stats']['total_records']}条记录\n\n")
            
            # 性能对比
            f.write("## 2. 性能对比结果\n\n")
            f.write("| 指标 | 固定周期 | PPO算法 | 改善率 |\n")
            f.write("|------|----------|---------|--------|\n")
            
            for metric, data in analysis_results['comparison_data'].items():
                f.write(f"| {metric} | {data['fixtime']:.2f} | {data['ppo']:.2f} | {data['improvement']:+.1f}% |\n")
            
            f.write(f"\n**平均改善率**: {analysis_results['avg_improvement']:+.1f}%\n\n")
            
            # 训练过程分析
            f.write("## 3. PPO训练过程分析\n\n")
            for phase, data in analysis_results['phase_analysis'].items():
                f.write(f"### {phase}阶段\n")
                f.write(f"- 平均等待时间: {data['avg_waiting']:.2f}秒\n")
                f.write(f"- 平均速度: {data['avg_speed']:.2f}m/s\n")
                f.write(f"- 平均通行量: {data['avg_throughput']:.2f}辆\n\n")
            
            # 结论
            avg_imp = analysis_results['avg_improvement']
            if avg_imp > 20:
                conclusion = "PPO算法在多数指标上显著优于固定周期控制，建议优先采用。"
            elif avg_imp > 0:
                conclusion = "PPO算法整体优于固定周期控制，但优势不够明显，需要结合具体场景选择。"
            else:
                conclusion = "固定周期控制在当前场景下表现更好，建议保持现有方案。"
            
            f.write(f"## 4. 结论与建议\n\n{conclusion}\n")
        
        self.logger.info(f"分析报告已保存: {output_path}")

# 使用示例函数
def analyze_training_files(sample_file_path: str) -> Tuple[Optional[str], str]:
    """
    便捷函数：基于样本文件分析多算法训练对比
    
    Args:
        sample_file_path: 样本文件路径（如zfdx-PPO_conn0_ep1.csv）
        
    Returns:
        Tuple[Optional[str], str]: (图片路径, 结果信息)
    """
    analyzer = ComparativeAnalyzer()
    return analyzer.analyze_training_comparison(sample_file_path)

def analyze_training_files_legacy(ppo_file: str, fixtime_file: str) -> Tuple[Optional[str], str]:
    """
    便捷函数：分析训练文件对比（兼容旧版本）
    
    Args:
        ppo_file: PPO训练日志文件路径
        fixtime_file: 固定周期结果文件路径
        
    Returns:
        Tuple[Optional[str], str]: (图片路径, 结果信息)
    """
    analyzer = ComparativeAnalyzer()
    return analyzer.analyze_training_comparison_legacy(ppo_file, fixtime_file)

if __name__ == "__main__":
    import sys
    import argparse
    
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='多算法对比分析工具')
    parser.add_argument('directory', help='包含多种算法结果文件的目录路径')
    parser.add_argument('--sample-file', help='样本文件路径（可选，用于指定特定的样本文件）')
    
    # 如果没有提供参数，使用默认测试
    if len(sys.argv) == 1:
        # 测试用例 - 新的多算法分析
        sample_file = "/Users/xnpeng/sumoptis/atscui/outs/train/zfdx-PPO_conn0_ep1.csv"
        
        result_path, message = analyze_training_files(sample_file)
        print(message)
        if result_path:
            print(f"多算法分析完成，结果保存在: {result_path}")
    else:
        # 解析命令行参数
        args = parser.parse_args()
        
        try:
            if args.sample_file:
                # 使用指定的样本文件
                result_path, message = analyze_training_files(args.sample_file)
            else:
                # 在指定目录中查找样本文件
                import glob
                pattern = os.path.join(args.directory, "*-PPO_conn0_ep*.csv")
                sample_files = glob.glob(pattern)
                
                if not sample_files:
                    # 尝试其他算法模式
                    patterns = [
                        "*-DQN_conn0_ep*.csv",
                        "*-A2C_conn0_ep*.csv",
                        "*-SAC_conn0_ep*.csv"
                    ]
                    for pattern_template in patterns:
                        pattern = os.path.join(args.directory, pattern_template)
                        sample_files = glob.glob(pattern)
                        if sample_files:
                            break
                
                if sample_files:
                    # 使用找到的第一个样本文件
                    sample_file = sample_files[0]
                    result_path, message = analyze_training_files(sample_file)
                else:
                    message = f"❌ 在目录 {args.directory} 中未找到合适的算法结果文件"
                    result_path = None
            
            print(message)
            if result_path:
                print(f"多算法分析完成，结果保存在: {result_path}")
                
        except Exception as e:
            print(f"❌ 执行分析时发生错误: {e}")
            sys.exit(1)