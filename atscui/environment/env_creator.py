from sumo_core.envs.sumo_env import ContinuousSumoEnv, SumoEnv
from atscui.utils.flow_generator import generate_curriculum_flow, extract_routes_from_template
from pathlib import Path
from datetime import datetime

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
        
        # 从流量模板文件中提取路线信息
        try:
            available_routes = extract_routes_from_template(config.rou_file)
            print(f"从模板文件中提取到 {len(available_routes)} 条路线: {list(available_routes.keys())}")
            
            # 根据提取的路线动态构建流量分布
            route_distribution = {}
            for route_id in available_routes.keys():
                # 为东西向路线设置基础流量，南北向路线设置为基础流量的0.8倍
                if 'we' in route_id.lower() or 'ew' in route_id.lower():
                    route_distribution[route_id] = config.curriculum_base_flow
                elif 'ns' in route_id.lower() or 'sn' in route_id.lower():
                    route_distribution[route_id] = config.curriculum_base_flow * 0.8
                else:
                    # 其他路线（如左转等）设置为基础流量的0.6倍
                    route_distribution[route_id] = config.curriculum_base_flow * 0.6
                    
        except Exception as e:
            print(f"从模板文件提取路线失败: {e}")
            print("使用默认的硬编码路线分布")
            # 回退到硬编码的路线分布
            route_distribution = {
                'route_we': config.curriculum_base_flow,
                'route_ew': config.curriculum_base_flow,
                'route_ns': config.curriculum_base_flow * 0.8,
                'route_sn': config.curriculum_base_flow * 0.8,
            }

        # 定义输出的流量文件路径
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        generated_rou_file = config.config_dir / f"{config.cross_name}-{config.algo_name}-{ts}.rou.xml"
        print(f"Generated curriculum file will be saved to: {generated_rou_file}")

        # 调用生成器
        static_phase_duration = generate_curriculum_flow(
            base_route_file=config.rou_file,
            output_file=generated_rou_file,
            total_sim_seconds=int(config.num_seconds * config.curriculum_static_ratio),
            stage_definitions=stage_definitions,
            route_distribution=route_distribution
        )
        
        # 更新配置以使用新生成的文件和参数
        final_rou_file = str(generated_rou_file)  # 确保路径是字符串格式
        use_dynamic_flows = True # 当使用课程学习时，还要启用动态流量，即在课程流量结束后，动态生成一部分流量
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
            flows_rate=config.curriculum_dynamic_rate
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
            flows_rate=config.curriculum_dynamic_rate,
            fixed_ts=fixed_ts  # 为FIXTIME算法启用固定配时
        )

    print("=====env:action_space:", env.action_space)
    env = Monitor(env, "monitor/SumoEnv-v0")
    env = DummyVecEnv([lambda: env])

    return env
