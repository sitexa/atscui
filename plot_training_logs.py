import pandas as pd
import matplotlib.pyplot as plt
import os
import re


def analyze_and_plot_from_specific_file(specific_log_file_path, log_directory="outs"):
    # Extract the base filename from the full path and strip any whitespace
    base_filename = os.path.basename(specific_log_file_path).strip()
    
    run_identifier = None
    episode_num_str = None

    # Robustly extract run_identifier and episode number using string manipulation
    try:
        # Find the last occurrence of '_ep'
        ep_idx = base_filename.rfind('_ep')
        if ep_idx == -1:
            raise ValueError("'_ep' not found in filename.")
        
        # Find the last occurrence of '.csv'
        csv_idx = base_filename.rfind('.csv')
        if csv_idx == -1 or csv_idx < ep_idx: # .csv must be after _ep
            raise ValueError("'.csv' not found or misplaced in filename.")
        
        # Extract the run_identifier (everything before '_ep')
        run_identifier = base_filename[:ep_idx]
        
        # Extract the episode number string (between '_ep' and '.csv')
        episode_num_str = base_filename[ep_idx + len('_ep'):csv_idx]
        
        # Validate that the extracted episode number is indeed digits
        if not episode_num_str.isdigit():
            raise ValueError("Episode number is not purely numeric.")

    except ValueError as e:
        return
    except Exception as e:
        return

    # Construct the specific file pattern for globbing all episodes of this run
    # re.escape is still good practice for the run_identifier part
    file_regex_pattern = rf"^{re.escape(run_identifier)}_ep(\d+)\.csv$"
    
    all_files_in_dir = []
    try:
        all_files_in_dir = os.listdir(log_directory)
    except FileNotFoundError:
        return

    all_files = []
    for f in all_files_in_dir:
        # Skip directories like 'plots'
        if os.path.isdir(os.path.join(log_directory, f)):
            continue
        
        # Strip whitespace/newlines from filename before matching
        stripped_f = f.strip()
        match = re.match(file_regex_pattern, stripped_f) # Use stripped filename for matching
        if match:
            all_files.append(os.path.join(log_directory, f)) # Add original filename to list

    if not all_files:
        return

    # Group files by episode
    run_data = []
    for file_path in all_files:
        # Re-match to extract episode number reliably (using original filename for path, but stripped for match)
        stripped_basename = os.path.basename(file_path).strip()
        match_ep = re.match(file_regex_pattern, stripped_basename)
        if match_ep:
            episode = int(match_ep.group(1))
            try:
                df = pd.read_csv(file_path)
                df['episode'] = episode
                run_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not run_data:
        return

    # Concatenate all episodes for the current run
    combined_df = pd.concat(run_data).sort_values(by='episode')

    # Define metrics to plot
    metrics = [
        'system_total_stopped',
        'system_mean_waiting_time',
        'system_mean_speed',
        'agents_total_stopped',
        'agents_total_accumulated_waiting_time'
    ]

    # Create plots directory if it's not exist
    plot_output_dir = os.path.join(log_directory, "plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    # Determine number of subplots needed
    num_metrics = len(metrics)
    num_cols = 2
    num_rows = (num_metrics + num_cols - 1) // num_cols # Ceiling division

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten() # Flatten to easily iterate

    for i, metric in enumerate(metrics):
        if metric in combined_df.columns:
            ax = axes[i]
            episode_summary = combined_df.groupby('episode')[metric].mean().reset_index()
            ax.plot(episode_summary['episode'], episode_summary[metric], marker='o')
            ax.set_title(f'{metric} (Mean per Episode)')
            ax.set_xlabel('Episode')
            ax.set_ylabel(metric)
            ax.grid(True)
        else:
            print(f"Metric '{metric}' not found in data for {run_identifier}.")
            fig.delaxes(axes[i]) # Remove empty subplot if metric not found

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        if j < len(axes):
            fig.delaxes(axes[j])

    plt.tight_layout()
    plot_filename = os.path.join(plot_output_dir, f'{run_identifier}_all_metrics.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Generated combined plot: {plot_filename}")

# Call the function with the desired specific log file path
analyze_and_plot_from_specific_file("outs/zfdx-PPO_conn2_ep1.csv")
