import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from atscui.logging_manager import get_logger

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  


class Visualizer:
    def __init__(self):
        self.logger = get_logger('visualizer')

    def plot_all_training_logs(self, specific_log_file_path: str, original_log_directory: Optional[str] = None) -> Optional[str]:
        """
        分析并绘制指定训练日志文件系列的所有指标图。

        Args:
            specific_log_file_path: 用户选择的单个日志文件的完整路径（例如 outs/zfdx-PPO_conn2_ep1.csv）。
            original_log_directory: 可选。日志文件实际所在的原始目录（例如 './outs'）。如果提供，将使用此目录来查找所有相关日志文件。

        Returns:
            str: 生成的图片路径，如果失败则返回 None。
        """
        if original_log_directory:
            log_directory = original_log_directory
        else:
            log_directory = os.path.dirname(specific_log_file_path)
        base_filename = os.path.basename(specific_log_file_path).strip()

        run_identifier = None
        try:
            ep_idx = base_filename.rfind('_ep')
            if ep_idx == -1:
                ep_idx = base_filename.rfind('-ep') # Fallback to handle '-ep'
            
            if ep_idx == -1:
                raise ValueError("'_ep' or '-ep' not found in filename.")

            csv_idx = base_filename.rfind('.csv')
            if csv_idx == -1 or csv_idx < ep_idx:
                raise ValueError("'.csv' not found or misplaced in filename.")

            run_identifier = base_filename[:ep_idx]
            episode_num_str = base_filename[ep_idx + len('_ep'):csv_idx]

            if not episode_num_str.isdigit():
                raise ValueError("Episode number is not purely numeric.")

        except ValueError as e:
            return None

        # 构建正则表达式以匹配相关文件
        file_regex_pattern = rf"^{re.escape(run_identifier)}[_-]ep\d+\.csv$"

        try:
            all_files_in_dir = os.listdir(log_directory)
        except FileNotFoundError:
            return None

        all_log_files = []
        for f in all_files_in_dir:
            # 跳过目录
            if os.path.isdir(os.path.join(log_directory, f)):
                continue
            
            stripped_f = f.strip()
            match = re.match(file_regex_pattern, stripped_f)
            if match:
                all_log_files.append(os.path.join(log_directory, f))

        if not all_log_files:
            return None

        # 重新编译正则表达式以提取 episode
        episode_pattern = re.compile(rf"^{re.escape(run_identifier)}[_-]ep(\d+)\.csv$")

        run_data = []
        for file_path in all_log_files:
            match_ep = episode_pattern.match(os.path.basename(file_path).strip())
            if match_ep:
                episode = int(match_ep.group(1))
                try:
                    df = pd.read_csv(file_path)
                    df['episode'] = episode
                    run_data.append(df)
                except Exception as e:
                    pass

        if not run_data:
            return None

        combined_df = pd.concat(run_data).sort_values(by='episode')

        metrics = [
            'system_total_stopped',
            'system_mean_waiting_time',
            'system_mean_speed',
            'system_total_throughput',
            'system_mean_travel_time',
            'agents_total_stopped',
            'agents_total_accumulated_waiting_time',
            'agents_total_throughput',
            'agents_mean_travel_time'
        ]

        plot_output_dir = os.path.join(log_directory, "plots")
        os.makedirs(plot_output_dir, exist_ok=True)

        num_metrics = len(metrics)
        num_cols = 3
        num_rows = (num_metrics + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 4))
        axes = axes.flatten()

        metrics_plotted = 0
        for i, metric in enumerate(metrics):
            if metric in combined_df.columns:
                ax = axes[metrics_plotted]
                episode_summary = combined_df.groupby('episode')[metric].mean().reset_index()

                if len(episode_summary) > 1:
                    ax.plot(episode_summary['episode'], episode_summary[metric], marker='o', linestyle='-')
                else:
                    ax.plot(episode_summary['episode'], episode_summary[metric], marker='o')

                ax.set_title(f'{metric} (Mean per Episode)')
                ax.set_xlabel('Episode')
                ax.set_ylabel(metric)
                ax.grid(True)
                metrics_plotted += 1

        for j in range(metrics_plotted, len(axes)):
            fig.delaxes(axes[j])

        if metrics_plotted == 0:
            plt.close(fig)
            return None

        plt.tight_layout()
        plot_filename = os.path.join(plot_output_dir, f'{run_identifier}_all_metrics.png')
        plt.savefig(plot_filename)
        plt.close(fig)
        return plot_filename

    @staticmethod
    def plot_training_process(csv_path: str, save_path: str = None):
        """Plot training metrics from CSV file"""
        df = pd.read_csv(csv_path)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot rewards
        ax1.plot(df['step'], df['reward'], label='Reward')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Reward')
        ax1.legend()

        # Plot loss
        if 'loss' in df.columns:
            ax2.plot(df['step'], df['loss'], label='Loss')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
            ax2.legend()

        if save_path:
            plt.savefig(save_path)
        return fig

    @staticmethod
    def plot_evaluation_results(eval_path: str, save_path: str = None):
        """Plot evaluation results"""
        with open(eval_path, 'r') as f:
            lines = f.readlines()

        metrics = {}
        for line in lines:
            key, value = line.strip().split(':')
            metrics[key.strip()] = float(value.strip())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(metrics.keys(), metrics.values())
        ax.set_ylabel('Value')
        plt.xticks(rotation=45)

        if save_path:
            plt.savefig(save_path)
        return fig

    @staticmethod
    def plot_process(train_out_file, folder_name, file_name):
        process_fig = Visualizer.replace_extension(file_name, "png")
        output_path = os.path.join(folder_name, process_fig)
        # 加载数据
        df = pd.read_csv(train_out_file)

        # 数据预处理
        df['step'] = pd.to_numeric(df['step'])
        # 创建2x2的子图布局，调整整体图表大小
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        # fig.suptitle('交通系统指标随时间变化\n{train_out_file}', fontsize=14)
        plt.suptitle(f'Traffic System Metrics Over Time\n{output_path}', fontsize=14)

        # 绘制系统平均速度
        axs[0, 0].plot(df['step'], df['system_mean_speed'], 'b-')
        axs[0, 0].set_title('Average Speed', fontsize=10)
        axs[0, 0].set_ylabel('speed', fontsize=8)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=6)
        axs[0, 0].grid(True)

        # 绘制系统停止的总车辆数
        axs[0, 1].plot(df['step'], df['system_total_stopped'], 'r-')
        axs[0, 1].set_title('System total stopped', fontsize=10)
        axs[0, 1].set_ylabel('vehicles', fontsize=8)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=6)
        axs[0, 1].grid(True)

        # 绘制系统总等待时间
        axs[1, 0].plot(df['step'], df['system_total_waiting_time'], 'g-')
        axs[1, 0].set_title('System total waiting time', fontsize=10)
        axs[1, 0].set_xlabel('timestep', fontsize=8)
        axs[1, 0].set_ylabel('waiting time', fontsize=8)
        axs[1, 0].tick_params(axis='both', which='major', labelsize=6)
        axs[1, 0].grid(True)

        # 绘制代理总停止数
        axs[1, 1].plot(df['step'], df['agents_total_stopped'], 'm-')
        axs[1, 1].set_title('agents total stopped', fontsize=10)
        axs[1, 1].set_xlabel('timestep', fontsize=8)
        axs[1, 1].set_ylabel('agent number', fontsize=8)
        axs[1, 1].tick_params(axis='both', which='major', labelsize=6)
        axs[1, 1].grid(True)

        # 调整子图间距
        plt.tight_layout()

        # 写入文件中
        plt.savefig(output_path)
        plt.close()
        print(f"图形已保存为{output_path}")
        return output_path

    @staticmethod
    def plot_predict(predict_file, folder_name, file_name):
        predict_fig = Visualizer.replace_extension(file_name, "png")
        output_path = os.path.join(folder_name, predict_fig)

        # 读取JSON数据
        with open(predict_file, 'r') as file:
            data = json.load(file)

        # 将数据转换为DataFrame，不包括iteration
        df = pd.DataFrame(data)

        # 设置图形大小
        plt.figure(figsize=(20, 10))
        plt.suptitle(f'Traffic System Metrics Over Time\n{output_path}', fontsize=14)

        # 绘制各项指标的图形
        metrics = [
            'system_total_stopped', 'system_total_waiting_time', 'system_mean_waiting_time',
            'system_mean_speed', 'tl_1_stopped', 'tl_1_accumulated_waiting_time',
            'tl_1_average_speed', 'agents_total_stopped', 'agents_total_accumulated_waiting_time'
        ]

        # 计算需要的列数
        num_cols = (len(metrics) + 1) // 2  # 向上取整，确保有足够的列

        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, num_cols, i)
            plt.plot(df['step'], df[metric], marker='o')
            plt.title(metric.replace('_', ' ').title(), fontsize=10)
            plt.xlabel('Step', fontsize=8)
            plt.ylabel('Value', fontsize=8)
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.grid(True)

        # 调整子图布局
        plt.tight_layout()

        # 保存图形
        plt.savefig(output_path)
        plt.close()

        print(f"图形已保存为{output_path}")
        return output_path

    @staticmethod
    def plot_evaluation(eval_folder, cross_name="my-intersection"):
        output_file = os.path.join(eval_folder, cross_name + "-eval.png")

        plt.figure(figsize=(12, 6))
        max_evaluations = 0

        for filename in os.listdir(eval_folder):
            if filename.startswith(cross_name + "-eval-") and filename.endswith(".txt"):
                file_path = os.path.join(eval_folder, filename)
                algorithm = filename.split('-')[-1].split('.')[0]  # 提取算法名称

                # 读取评估结果文件
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # 解析数据
                mean_rewards = []
                std_rewards = []

                for line in lines:
                    _, mean_reward, std_reward = line.strip().split(', ')
                    mean_rewards.append(float(mean_reward))
                    std_rewards.append(float(std_reward))

                # 使用评估次序作为x轴
                x = range(1, len(mean_rewards) + 1)
                max_evaluations = max(max_evaluations, len(mean_rewards))

                # 绘制数据
                plt.errorbar(x, mean_rewards, yerr=std_rewards, fmt='o-', capsize=5, label=algorithm)

        plt.title(f'Evaluation Results Comparison\n{output_file}')
        plt.xlabel('Evaluation Sequence')
        plt.ylabel('Mean Reward')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        # 设置x轴刻度为整数
        if max_evaluations > 0:
            plt.xticks(range(1, max_evaluations + 1))

        plt.tight_layout()

        # 设置x轴刻度为整数
        plt.xticks(range(1, max(plt.xticks()[0]) + 1))

        # 保存图形

        plt.savefig(output_file)
        plt.close()

        print(f"评估结果比较图形已保存为 {output_file}")

        return output_file

    @staticmethod
    def replace_extension(pathname, new_extension):
        # 确保新扩展名以点开头
        if not new_extension.startswith('.'):
            new_extension = '.' + new_extension
        # 分离路径和文件名
        directory, filename = os.path.split(pathname)
        # 分离文件名和扩展名
        name, _ = os.path.splitext(filename)
        # 组合新的文件名
        new_filename = name + new_extension
        # 组合新的完整路径
        new_pathname = os.path.join(directory, new_filename)
        return new_pathname


# 创建全局visualizer实例
visualizer = Visualizer()

# 便捷函数
def plot_process(train_out_file, folder_name, file_name):
    return visualizer.plot_process(train_out_file, folder_name, file_name)

def plot_predict(predict_file, folder_name, file_name):
    return visualizer.plot_predict(predict_file, folder_name, file_name)
