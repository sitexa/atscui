import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob as py_glob # 避免与 default_api.glob 混淆

def analyze_and_plot_training_logs(intersection_name, algorithm_name, connection_id, output_dir="outs"):
    """
    分析训练日志文件并绘制指标变化曲线图。

    Args:
        intersection_name (str): 路口名称，例如 'zfdx'
        algorithm_name (str): 算法名称，例如 'DQN'
        connection_id (str): 连接ID，例如 '1'
        output_dir (str): 日志文件所在的目录，默认为 'outs'
    """
    all_data = []
    pattern = re.compile(rf"{intersection_name}-{algorithm_name}_conn{connection_id}_ep(\d+)\.csv")

    # 使用 glob 查找文件，并禁用 .gitignore
    import glob
    search_pattern = os.path.join(output_dir, f"{intersection_name}-{algorithm_name}_conn{connection_id}_ep*.csv")
    
    # 获取当前工作目录，以便 glob 可以正确解析相对路径
    current_working_directory = os.getcwd()
    absolute_search_pattern = os.path.join(current_working_directory, search_pattern)

    # 使用 glob 查找文件，并禁用 .gitignore
    # 注意：这里无法直接调用 default_api.glob，因为这是在 Python 脚本内部。
    # 假设 os.listdir 能够找到文件，���者用户已经处理了 .gitignore 问题。
    # 为了在脚本中处理 .gitignore，需要手动过滤，或者依赖于外部工具（如 default_api.glob）
    # 在这里，我将继续使用 os.listdir，并假设文件是可见的。
    
    # 改进：直接使用 default_api.glob 的结果，而不是 os.listdir
    # 但由于 analyze_logs.py 是一个独立脚本，它无法直接调用 default_api.glob。
    # 因此，我将保留 os.listdir 的方法，并提醒用户确保文件可见。
    
    # 考虑到之前 glob 失败是因为 .gitignore，这里需要确保脚本能够访问这些文件。
    # 最直接的方法是让用户在运行脚本前手动处理 .gitignore，或者将文件复制到非忽略目录。
    # 另一种方法是在脚本中尝试读取所有文件，并处理 FileNotFoundError。
    
    # 鉴于当前环境的限制，我将继续使用 os.listdir，并假设文件是可访问的。
    # 如果文件仍然无法访问，用户需要手动调整 .gitignore 或文件位置。

    # 重新考虑：为了让脚本更健壮，我应该尝试使用 os.walk 或 glob.glob
    # 但 glob.glob 同样会受到 .gitignore 的影响，除非手动实现忽略逻辑。
    # 最好的方法是让用户提供一个文件列表，或者确保文件在可访问的目录中。

    # 鉴于用户要求将功能放在 atscui/ui/components/visualization_tab.py 上，
    # 那么在 visualization_tab.py 中调用 analyze_and_plot_training_logs 时，
    # 我们可以利用 default_api.glob 来获取文件列表，然后传递给这个函数。
    # 所以，analyze_and_plot_training_logs 函数本身不需要处理 .gitignore。
    # 它只需要一个文件路径列表。

    # 重新设计 analyze_and_plot_training_logs 函数，使其接受文件路径列表
    # 而不是自己去查找文件。这样，在 UI 中调用时，可以先用 default_api.glob 找到文件，
    # 然后将文件路径列表传递给这个函数。

    # 暂时先这样写，后续在集成到 UI 时再调整。
    # 为了让这个独立脚本能够运行，我将使用 glob.glob 并提醒用户 .gitignore 的问题。
    
    import glob as py_glob # 避免与 default_api.glob 混淆

    # 查找所有匹配的文件，不考虑 .gitignore
    # 注意：py_glob.glob 默认不尊重 .gitignore，但如果文件系统权限或路径问题，仍然可能失败。
    # 这里的目的是为了在独立脚本中模拟查找过程。
    file_paths = py_glob.glob(absolute_search_pattern)
    
    if not file_paths:
        print(f"No files found matching pattern: {absolute_search_pattern}. Please check the path and .gitignore settings.")
        return

    for filepath in file_paths:
        filename = os.path.basename(filepath)
        match = pattern.match(filename)
        if match:
            episode_num = int(match.group(1))
            try:
                df = pd.read_csv(filepath)
                df['episode'] = episode_num
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    if not all_data:
        print(f"No valid data found in matching log files for {intersection_name}-{algorithm_name}_conn{connection_id}.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # 计算每个回合的平均指标
    episode_summary = combined_df.groupby('episode').agg(
        mean_waiting_time=('system_mean_waiting_time', 'mean'),
        mean_speed=('system_mean_speed', 'mean'),
        total_stopped=('system_total_stopped', 'sum'),
        total_accumulated_waiting_time=('agents_total_accumulated_waiting_time', 'sum')
    ).reset_index()

    # 绘制曲线图
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(episode_summary['episode'], episode_summary['mean_waiting_time'], marker='o')
    plt.title('Mean Waiting Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Waiting Time')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(episode_summary['episode'], episode_summary['mean_speed'], marker='o', color='green')
    plt.title('Mean Speed per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Mean Speed')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(episode_summary['episode'], episode_summary['total_stopped'], marker='o', color='red')
    plt.title('Total Stopped Vehicles per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Stopped Vehicles')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(episode_summary['episode'], episode_summary['total_accumulated_waiting_time'], marker='o', color='purple')
    plt.title('Total Accumulated Waiting Time per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Accumulated Waiting Time')
    plt.grid(True)

    plt.tight_layout()
    plt.suptitle(f"Training Metrics for {intersection_name}-{algorithm_name}_conn{connection_id}", y=1.02)
    plt.show()

# 示例调用 (在实际集成时，这些参数将从UI获取)
# if __name__ == "__main__":
#     # 为了测试，假设存在这些文件
#     # analyze_and_plot_training_logs('zfdx', 'DQN', '1')
#     pass
