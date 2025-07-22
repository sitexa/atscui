from sumo_core.envs.sumo_env import ContinuousSumoEnv, SumoEnv
from atscui.utils.flow_generator import generate_curriculum_flow
from pathlib import Path


def createEnv(config):
    # 延迟导入stable_baselines3模块
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    # --- 动态流量文件生成 ---
    if config.use_curriculum_learning:
        print("=====Using Curriculum Learning: Generating dynamic flow file...=====")
        # 定义课程阶段
        stage_definitions = [
            {'name': 'low', 'duration_ratio': 0.3, 'flow_rate_multiplier': 0.5},
            {'name': 'medium', 'duration_ratio': 0.4, 'flow_rate_multiplier': 1.0},
            {'name': 'high', 'duration_ratio': 0.3, 'flow_rate_multiplier': 2.0}
        ]
        
        # 定义各路线的基础流量
        # This part can be made more sophisticated, e.g., by reading from config
        route_distribution = {
            'route_we': config.base_flow_rate,
            'route_ew': config.base_flow_rate,
            'route_ns': config.base_flow_rate * 0.8,
            'route_sn': config.base_flow_rate * 0.8,
        }

        # 定义输出的临时流量文件��径
        # It's good practice to save it in the same directory as the original route file
        generated_rou_file = str(Path(config.rou_file).parent / "curriculum.rou.xml")
        print(f"Generated curriculum file will be saved to: {generated_rou_file}")

        # 调用生成器
        static_phase_duration = generate_curriculum_flow(
            base_route_file=config.base_template_rou_file,
            output_file=generated_rou_file,
            total_sim_seconds=int(config.num_seconds * config.static_phase_ratio),
            stage_definitions=stage_definitions,
            route_distribution=route_distribution
        )
        
        # 更新配置以使用新生成的文件和参数
        final_rou_file = generated_rou_file
        use_dynamic_flows = True
        dynamic_start_time = static_phase_duration
        print(f"Static phases will run for {static_phase_duration} seconds, then switch to dynamic flows.")

    else:
        # 保持原有逻辑，使用固定的流量文件
        print("=====Using Static Flow File=====")
        final_rou_file = config.rou_file
        use_dynamic_flows = False
        dynamic_start_time = 999999  # 一个很大的数，确保不触发

    if config.algo_name == "SAC":
        print("=====create ContinuousEnv for SAC=====")
        env = ContinuousSumoEnv(
            net_file=config.net_file,
            route_file=final_rou_file,
            out_csv_name=config.csv_path,
            single_agent=config.single_agent,
            use_gui=config.gui,
            num_seconds=config.num_seconds,
            render_mode=config.render_mode,
            use_dynamic_flows=use_dynamic_flows,
            dynamic_start_time=dynamic_start_time,
            flows_rate=config.dynamic_flows_rate
        )
    else:
        # 为FIXTIME算法设置固定配时参数
        fixed_ts = config.algo_name.upper() == "FIXTIME"
        
        env = SumoEnv(
            net_file=config.net_file,
            route_file=final_rou_file,
            out_csv_name=config.csv_path,
            single_agent=config.single_agent,
            use_gui=config.gui,
            num_seconds=config.num_seconds,
            render_mode=config.render_mode,
            use_dynamic_flows=use_dynamic_flows,
            dynamic_start_time=dynamic_start_time,
            flows_rate=config.dynamic_flows_rate,
            fixed_ts=fixed_ts  # 为FIXTIME算法启用固定配时
        )

    print("=====env:action_space:", env.action_space)
    env = Monitor(env, "monitor/SumoEnv-v0")
    env = DummyVecEnv([lambda: env])

    return env
