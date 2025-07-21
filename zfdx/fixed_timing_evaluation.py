#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定周期交通信号仿真评估脚本
专注于运行zfdx路口的固定周期信号控制仿真，保存结果数据，并与现有智能体控制方案进行比较分析

作者: ATSCUI系统
日期: 2024
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sumo_core.envs.sumo_env import SumoEnv

class FixedTimingEvaluator:
    """固定周期交通信号评估器 - 专注于固定周期仿真和结果对比分析"""
    
    def __init__(self, net_file, route_file, output_dir="evaluation_results"):
        self.net_file = net_file
        self.route_file = route_file
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚦 固定周期交通信号评估器初始化完成")
        print(f"📁 网络文件: {net_file}")
        print(f"📁 路由文件: {route_file}")
        print(f"📁 输出目录: {output_dir}")
    
    def create_fixed_timing_env(self, episode_length=3600, delta_time=5):
        """创建固定周期仿真环境"""
        env = SumoEnv(
            net_file=self.net_file,
            route_file=self.route_file,
            out_csv_name=f"{self.output_dir}/fixed_timing",
            use_gui=False,
            num_seconds=episode_length,
            delta_time=delta_time,
            yellow_time=3,
            min_green=10,
            max_green=60,
            fixed_ts=True,  # 关键参数：启用固定周期
            sumo_seed=42,
            sumo_warnings=False
        )
        return env
    
    def run_fixed_timing_simulation(self, num_episodes=5, episode_length=3600, delta_time=5):
        """运行固定周期仿真"""
        print(f"\n🔄 开始运行固定周期仿真...")
        print(f"📊 仿真参数: {num_episodes}轮 × {episode_length}秒/轮")
        
        all_results = []
        
        for episode in range(num_episodes):
            print(f"\n🔄 运行第 {episode + 1}/{num_episodes} 轮仿真...")
            
            try:
                # 创建环境
                env = self.create_fixed_timing_env(episode_length, delta_time)
                
                # 重置环境
                obs = env.reset()
                done = False
                step = 0
                
                episode_metrics = {
                    'episode': episode + 1,
                    'total_steps': 0,
                    'avg_waiting_time': 0,
                    'avg_queue_length': 0,
                    'avg_speed': 0,
                    'total_throughput': 0,
                    'avg_travel_time': 0,
                    'total_fuel_consumption': 0,
                    'total_co2_emission': 0
                }
                
                # 运行仿真
                while not done:
                    # 固定周期模式下不需要动作，环境会自动按照固定周期运行
                    obs, reward, done, info = env.step({})
                    step += 1
                    
                    # 每100步输出一次进度
                    if step % 100 == 0:
                        progress = (step * delta_time) / episode_length * 100
                        print(f"  进度: {progress:.1f}% ({step * delta_time}/{episode_length}秒)")
                
                # 提取最终指标
                episode_metrics.update(self._extract_final_metrics(info, episode, step))
                episode_metrics['total_steps'] = step
                
                all_results.append(episode_metrics)
                
                print(f"✅ 第 {episode + 1} 轮完成 - 等待时间: {episode_metrics['avg_waiting_time']:.2f}s")
                
                # 关闭环境
                env.close()
                
            except Exception as e:
                print(f"❌ 第 {episode + 1} 轮仿真失败: {e}")
                continue
        
        if all_results:
            print(f"\n✅ 固定周期仿真完成！共成功运行 {len(all_results)} 轮")
        else:
            print(f"\n❌ 所有仿真轮次都失败了")
        
        return all_results
    
    def _extract_final_metrics(self, info, episode, step):
        """从仿真信息中提取最终指标"""
        metrics = {
            'avg_waiting_time': 0,
            'avg_queue_length': 0,
            'avg_speed': 0,
            'total_throughput': 0,
            'avg_travel_time': 0,
            'total_fuel_consumption': 0,
            'total_co2_emission': 0
        }
        
        try:
            # 从info中提取系统级指标
            if isinstance(info, dict):
                # 尝试不同的键名
                waiting_keys = ['system_mean_waiting_time', 'mean_waiting_time', 'avg_waiting_time']
                for key in waiting_keys:
                    if key in info:
                        metrics['avg_waiting_time'] = float(info[key])
                        break
                
                speed_keys = ['system_mean_speed', 'mean_speed', 'avg_speed']
                for key in speed_keys:
                    if key in info:
                        metrics['avg_speed'] = float(info[key])
                        break
                
                queue_keys = ['system_total_stopped', 'total_stopped', 'avg_queue_length']
                for key in queue_keys:
                    if key in info:
                        metrics['avg_queue_length'] = float(info[key])
                        break
                
                throughput_keys = ['system_total_arrived', 'total_arrived', 'total_throughput']
                for key in throughput_keys:
                    if key in info:
                        metrics['total_throughput'] = float(info[key])
                        break
                
                # 其他指标
                if 'system_mean_travel_time' in info:
                    metrics['avg_travel_time'] = float(info['system_mean_travel_time'])
                if 'system_total_fuel_consumption' in info:
                    metrics['total_fuel_consumption'] = float(info['system_total_fuel_consumption'])
                if 'system_total_co2_emission' in info:
                    metrics['total_co2_emission'] = float(info['system_total_co2_emission'])
        
        except Exception as e:
            print(f"⚠️  提取指标时出错: {e}")
        
        return metrics
    
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
        elif 'agent' in filename_lower:
            return 'Agent'
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
                    sns.boxplot(data=comparison_data, x='control_type', y=metric, ax=ax)
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


def main():
    """主函数 - 运行固定周期评估和对比分析"""
    print("🚦 开始ZFDX路口固定周期交通信号评估")
    print("=" * 60)
    
    # 配置文件路径
    net_file = "net/zfdx.net.xml"
    route_file = "net/zfdx-perhour.rou.xml"
    
    # 检查文件是否存在
    if not os.path.exists(net_file):
        print(f"❌ 网络文件不存在: {net_file}")
        print("💡 请确保在zfdx目录下运行此脚本")
        return
    
    if not os.path.exists(route_file):
        print(f"❌ 路由文件不存在: {route_file}")
        print("💡 请确保在zfdx目录下运行此脚本")
        return
    
    # 创建评估器
    evaluator = FixedTimingEvaluator(net_file, route_file)
    
    try:
        # 运行固定周期仿真
        print("\n🔄 开始固定周期仿真...")
        print("⏱️  预计需要几分钟时间，请耐心等待...")
        
        fixed_results = evaluator.run_fixed_timing_simulation(
            num_episodes=10,      # 运行5轮仿真
            episode_length=3600, # 每轮1小时
            delta_time=5         # 5秒步长
        )
        
        if not fixed_results:
            print("❌ 固定周期仿真失败")
            return
        
        print(f"✅ 固定周期仿真完成，共 {len(fixed_results)} 轮")
        
        # 保存固定周期结果
        fixed_df = pd.DataFrame(fixed_results)
        fixed_file = f"{evaluator.output_dir}/fixed_timing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        fixed_df.to_csv(fixed_file, index=False)
        print(f"💾 固定周期结果已保存: {fixed_file}")
        
        # 查找智能体结果
        print("\n🔍 查找现有的智能体结果文件...")
        agent_files = evaluator.find_agent_results()
        
        # 进行对比分析
        comparison_data = evaluator.compare_with_agent_results(fixed_results, agent_files)
        
        print("\n🎉 评估完成！")
        print(f"📁 所有结果保存在: {evaluator.output_dir}")
        print("\n📋 使用说明:")
        print("  1. 固定周期仿真结果已保存为CSV格式")
        print("  2. 如有智能体结果，对比分析图表已生成")
        print("  3. 可以将智能体训练/评估的CSV结果文件放入evaluation_results目录进行对比")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断了仿真")
    except Exception as e:
        print(f"\n❌ 仿真过程中出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 故障排除提示:")
        print("  1. 确保SUMO_HOME环境变量已设置")
        print("  2. 确保在zfdx目录下运行脚本")
        print("  3. 检查网络和路由文件是否存在")

if __name__ == "__main__":
    main()