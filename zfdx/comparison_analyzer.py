#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能体结果对比分析器
专门用于处理固定周期与智能体控制方案的对比分析

作者: ATSCUI系统
日期: 2024
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob

# 可选依赖
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  警告: seaborn 未安装，将使用基础的 matplotlib 绘图")


class ComparisonAnalyzer:
    """智能体结果对比分析器 - 专注于固定周期与智能体控制方案的对比分析"""
    
    def __init__(self, output_dir="evaluation_results"):
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📊 对比分析器初始化完成")
        print(f"📁 输出目录: {output_dir}")
    
    def find_agent_results(self):
        """查找现有的智能体结果文件"""
        agent_files = []
        
        # 扩大搜索范围，查找项目中的智能体结果文件
        search_dirs = [
            self.output_dir,
            "../outs",
            ".."
        ]
        
        csv_patterns = [
            "*DQN*.csv", 
            "*PPO*.csv",
            "*A2C*.csv",
            "*SAC*.csv"
        ]
        
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for pattern in csv_patterns:
                    full_pattern = os.path.join(search_dir, pattern)
                    files = glob.glob(full_pattern)
                    agent_files.extend(files)
        
        # 去重并过滤掉固定周期结果文件
        agent_files = list(set(agent_files))
        agent_files = [f for f in agent_files if 'fixed' not in os.path.basename(f).lower()]
        
        if agent_files:
            print(f"📊 找到 {len(agent_files)} 个智能体结果文件:")
            for file in agent_files:
                print(f"   - {file}")
        else:
            print("⚠️  未找到智能体结果文件，将只进行固定周期仿真")
            print("💡 提示：请确保智能体训练/评估结果已保存为CSV格式")
        
        return agent_files
    
    def compare_with_agent_results(self, fixed_results, agent_files):
        """与智能体结果进行对比分析"""
        if not agent_files:
            print("⚠️  没有智能体结果文件可供对比")
            print("📊 将只显示固定周期仿真结果")
            self._show_fixed_timing_summary(fixed_results)
            return
        
        print("\n📊 开始对比分析...")
        
        # 加载智能体结果
        agent_data_list = []
        for file in agent_files:
            try:
                df = pd.read_csv(file)
                # 添加文件来源信息
                algorithm = self._extract_algorithm_from_filename(file)
                df['algorithm'] = algorithm
                df['source_file'] = os.path.basename(file)
                agent_data_list.append(df)
                print(f"✅ 成功加载: {file} (算法: {algorithm})")
            except Exception as e:
                print(f"❌ 加载失败 {file}: {e}")
        
        if not agent_data_list:
            print("❌ 没有成功加载的智能体结果文件")
            self._show_fixed_timing_summary(fixed_results)
            return
        
        # 合并智能体数据
        agent_data = pd.concat(agent_data_list, ignore_index=True)
        
        # 准备对比数据
        fixed_df = pd.DataFrame(fixed_results)
        fixed_df['control_type'] = 'Fixed Timing'
        fixed_df['algorithm'] = 'Fixed Timing'
        
        agent_df = agent_data.copy()
        agent_df['control_type'] = 'Agent Control'
        
        # 标准化列名
        fixed_df = self._standardize_columns(fixed_df)
        agent_df = self._standardize_columns(agent_df)
        
        # 找到共同的指标列
        metric_columns = ['avg_waiting_time', 'avg_queue_length', 'avg_speed', 
                         'total_throughput', 'avg_travel_time']
        available_metrics = [col for col in metric_columns if col in fixed_df.columns and col in agent_df.columns]
        
        if len(available_metrics) < 2:
            print("❌ 数据列不匹配，无法进行对比")
            print(f"固定周期数据列: {list(fixed_df.columns)}")
            print(f"智能体数据列: {list(agent_df.columns)}")
            self._show_fixed_timing_summary(fixed_results)
            return
        
        # 选择用于对比的列
        comparison_columns = available_metrics + ['control_type', 'algorithm']
        fixed_subset = fixed_df[comparison_columns]
        agent_subset = agent_df[comparison_columns]
        
        # 合并数据
        comparison_data = pd.concat([fixed_subset, agent_subset], ignore_index=True)
        
        # 生成对比报告
        self._generate_comparison_report(comparison_data, available_metrics)
        
        # 生成对比图表
        self._generate_comparison_plots(comparison_data, available_metrics)
        
        # 保存对比数据
        comparison_file = f"{self.output_dir}/comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_data.to_csv(comparison_file, index=False)
        print(f"💾 对比数据已保存: {comparison_file}")
        
        return comparison_data
    
    def _extract_algorithm_from_filename(self, filename):
        """从文件名中提取算法名称"""
        filename_lower = os.path.basename(filename).lower()
        if 'dqn' in filename_lower:
            return 'DQN'
        elif 'ppo' in filename_lower:
            return 'PPO'
        elif 'a2c' in filename_lower:
            return 'A2C'
        elif 'sac' in filename_lower:
            return 'SAC'
        elif 'fixtime' in filename_lower:
            return 'FixTime'
        else:
            return 'Unknown'
    
    def _show_fixed_timing_summary(self, fixed_results):
        """显示固定周期仿真结果摘要"""
        if not fixed_results:
            return
            
        df = pd.DataFrame(fixed_results)
        print("\n" + "="*60)
        print("📊 固定周期交通信号仿真结果摘要")
        print("="*60)
        
        metrics = ['avg_waiting_time', 'avg_queue_length', 'avg_speed', 
                  'total_throughput', 'avg_travel_time']
        
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                std_val = df[metric].std()
                print(f"{metric:20s}: {mean_val:8.2f} ± {std_val:6.2f}")
        
        print("="*60)
    
    def _standardize_columns(self, df):
        """标准化列名"""
        column_mapping = {
            'waiting_time': 'avg_waiting_time',
            'queue_length': 'avg_queue_length', 
            'speed': 'avg_speed',
            'throughput': 'total_throughput',
            'travel_time': 'avg_travel_time',
            'mean_waiting_time': 'avg_waiting_time',
            'mean_queue_length': 'avg_queue_length',
            'mean_speed': 'avg_speed',
            'mean_travel_time': 'avg_travel_time',
            # 添加更多可能的列名映射
            'system_mean_waiting_time': 'avg_waiting_time',
            'system_mean_speed': 'avg_speed',
            'system_total_stopped': 'avg_queue_length'
        }
        
        df_renamed = df.rename(columns=column_mapping)
        return df_renamed
    
    def _generate_comparison_report(self, comparison_data, metrics):
        """生成对比分析报告"""
        print("\n" + "="*80)
        print("📊 固定周期 vs 智能体控制 - 性能比较报告")
        print("="*80)
        
        # 按控制类型分组统计
        grouped = comparison_data.groupby('control_type')[metrics].agg(['mean', 'std'])
        
        print("\n📈 关键性能指标对比:")
        print("-" * 80)
        print(f"{'指标':<20} {'固定周期':<15} {'智能体控制':<15} {'改善幅度':<15}")
        print("-" * 80)
        
        improvements = {}
        
        for metric in metrics:
            if metric in grouped.columns.get_level_values(0):
                fixed_mean = grouped.loc['Fixed Timing', (metric, 'mean')]
                agent_mean = grouped.loc['Agent Control', (metric, 'mean')] if 'Agent Control' in grouped.index else None
                
                if agent_mean is not None:
                    # 计算改善幅度（对于等待时间、排队长度等，减少是好的）
                    if 'waiting' in metric or 'queue' in metric or 'travel' in metric:
                        improvement = (fixed_mean - agent_mean) / fixed_mean * 100
                    else:  # 对于速度、吞吐量等，增加是好的
                        improvement = (agent_mean - fixed_mean) / fixed_mean * 100
                    
                    improvements[metric] = improvement
                    
                    print(f"{metric:<20} {fixed_mean:<15.2f} {agent_mean:<15.2f} {improvement:+.1f}%")
                else:
                    print(f"{metric:<20} {fixed_mean:<15.2f} {'N/A':<15} {'N/A':<15}")
        
        # 按算法详细对比
        if 'algorithm' in comparison_data.columns:
            print("\n🔍 按算法详细对比:")
            print("-" * 80)
            
            algo_grouped = comparison_data.groupby('algorithm')[metrics].agg(['mean', 'std'])
            
            for metric in metrics:
                if metric in algo_grouped.columns.get_level_values(0):
                    print(f"\n{metric}:")
                    for algo in algo_grouped.index:
                        mean_val = algo_grouped.loc[algo, (metric, 'mean')]
                        std_val = algo_grouped.loc[algo, (metric, 'std')]
                        print(f"  {algo:<15}: {mean_val:8.2f} ± {std_val:6.2f}")
        
        # 总结
        print("\n🎯 总结:")
        if improvements:
            avg_improvement = np.mean(list(improvements.values()))
            print(f"✅ 平均性能改善: {avg_improvement:+.1f}%")
            
            best_metric = max(improvements.items(), key=lambda x: x[1])
            print(f"🏆 最佳改善指标: {best_metric[0]} ({best_metric[1]:+.1f}%)")
            
            # 显示具体改善情况
            for metric, improvement in improvements.items():
                if improvement > 0:
                    print(f"✅ {metric}: 改善 {improvement:.1f}%")
                else:
                    print(f"❌ {metric}: 下降 {abs(improvement):.1f}%")
        else:
            print("⚠️  无法计算改善幅度")
        
        print("="*80)
    
    def _generate_comparison_plots(self, comparison_data, metrics):
        """生成对比图表"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 箱线图对比
            n_metrics = min(len(metrics), 4)  # 最多显示4个指标
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('固定周期 vs 智能体控制 - 性能对比 (箱线图)', fontsize=16, fontweight='bold')
            
            axes_flat = axes.flatten()
            
            for i, metric in enumerate(metrics[:4]):
                if i < len(axes_flat):
                    ax = axes_flat[i]
                    
                    # 创建箱线图
                    if HAS_SEABORN:
                        sns.boxplot(data=comparison_data, x='control_type', y=metric, ax=ax)
                    else:
                        # 使用 matplotlib 创建箱线图
                        control_types = comparison_data['control_type'].unique()
                        data_by_type = [comparison_data[comparison_data['control_type'] == ct][metric].values 
                                      for ct in control_types]
                        ax.boxplot(data_by_type, labels=control_types)
                    ax.set_title(f'{metric.replace("_", " ").title()}')
                    ax.set_xlabel('控制方式')
                    ax.set_ylabel('数值')
                    ax.tick_params(axis='x', rotation=45)
            
            # 隐藏多余的子图
            for i in range(n_metrics, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            plt.tight_layout()
            plot_file1 = f"{self.output_dir}/comparison_boxplots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
            print(f"📊 箱线图已保存: {plot_file1}")
            plt.close()
            
            # 2. 算法对比柱状图（如果有多个算法）
            if 'algorithm' in comparison_data.columns and len(comparison_data['algorithm'].unique()) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle('不同算法性能对比 (柱状图)', fontsize=16, fontweight='bold')
                
                axes_flat = axes.flatten()
                
                for i, metric in enumerate(metrics[:4]):
                    if i < len(axes_flat):
                        ax = axes_flat[i]
                        
                        # 计算各算法的平均值
                        algo_means = comparison_data.groupby('algorithm')[metric].mean()
                        
                        # 创建柱状图
                        bars = ax.bar(algo_means.index, algo_means.values)
                        ax.set_title(f'{metric.replace("_", " ").title()}')
                        ax.set_xlabel('算法')
                        ax.set_ylabel('平均值')
                        ax.tick_params(axis='x', rotation=45)
                        
                        # 在柱子上显示数值
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.2f}', ha='center', va='bottom')
                
                # 隐藏多余的子图
                for i in range(n_metrics, len(axes_flat)):
                    axes_flat[i].set_visible(False)
                
                plt.tight_layout()
                plot_file2 = f"{self.output_dir}/algorithm_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
                print(f"📊 算法对比图已保存: {plot_file2}")
                plt.close()
            
        except Exception as e:
            print(f"⚠️  生成图表时出错: {e}")
            print("💡 提示: 可能需要安装中文字体或调整matplotlib配置")