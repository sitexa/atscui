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
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sumo_core.envs.sumo_env import SumoEnv
from comparison_analyzer import ComparisonAnalyzer
from atscui.utils.flow_generator import generate_curriculum_flow, extract_routes_from_template
import xml.etree.ElementTree as ET

class FixedTimingEvaluator:
    """固定周期交通信号评估器 - 专注于固定周期仿真和结果对比分析"""
    
    def __init__(self, net_file, route_file, output_dir="evaluation_results", 
                 use_curriculum_learning=False, base_template_rou_file=None):
        self.net_file = net_file
        self.route_file = route_file
        self.output_dir = output_dir
        self.use_curriculum_learning = use_curriculum_learning
        self.base_template_rou_file = base_template_rou_file or route_file
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化对比分析器
        self.comparison_analyzer = ComparisonAnalyzer(output_dir)
        
        print(f"🚦 固定周期交通信号评估器初始化完成")
        print(f"📁 网络文件: {net_file}")
        print(f"📁 路由文件: {route_file}")
        print(f"📁 输出目录: {output_dir}")
        print(f"🎓 课程学习: {'启用' if use_curriculum_learning else '禁用'}")
        if use_curriculum_learning:
            print(f"📚 模板文件: {self.base_template_rou_file}")
    
    def _calculate_static_flow_rate(self, route_file):
        """计算静态流量文件的总流量率"""
        try:
            tree = ET.parse(route_file)
            root = tree.getroot()
            
            total_flow_rate = 0
            flow_count = 0
            
            # 查找所有flow元素
            for flow in root.findall('flow'):
                veh_per_hour = flow.get('vehsPerHour')
                if veh_per_hour:
                    total_flow_rate += float(veh_per_hour)
                    flow_count += 1
            
            print(f"📊 静态流量文件分析: {flow_count}个流量定义，总流量率: {total_flow_rate} veh/h")
            return total_flow_rate
            
        except Exception as e:
            print(f"⚠️  计算静态流量率失败: {e}")
            return 0
    
    def _calculate_dynamic_average_flow_rate(self, route_file, stage_definitions):
        """计算动态流量文件的平均流量率（考虑时间权重）"""
        try:
            tree = ET.parse(route_file)
            root = tree.getroot()
            
            # 按阶段分组流量
            stage_flows = {}
            for flow in root.findall('flow'):
                flow_id = flow.get('id')
                veh_per_hour = float(flow.get('vehsPerHour', 0))
                
                # 提取阶段名称（low_, medium_, high_）
                stage_name = None
                for stage in stage_definitions:
                    if flow_id.startswith(stage['name'] + '_'):
                        stage_name = stage['name']
                        break
                
                if stage_name:
                    if stage_name not in stage_flows:
                        stage_flows[stage_name] = 0
                    stage_flows[stage_name] += veh_per_hour
            
            # 计算时间加权平均流量率
            total_weighted_flow = 0
            for stage in stage_definitions:
                stage_name = stage['name']
                duration_ratio = stage['duration_ratio']
                stage_flow = stage_flows.get(stage_name, 0)
                
                weighted_flow = stage_flow * duration_ratio
                total_weighted_flow += weighted_flow
                print(f"📊 {stage_name}阶段: {stage_flow} veh/h × {duration_ratio:.1%} = {weighted_flow:.1f}")
            
            print(f"📊 动态流量时间加权平均: {total_weighted_flow:.1f} veh/h")
            return total_weighted_flow
            
        except Exception as e:
            print(f"⚠️  计算动态平均流量率失败: {e}")
            return 0
    
    def create_fixed_timing_env(self, episode_length=3600, delta_time=4):
        """创建固定周期仿真环境"""
        # 处理课程学习路由文件
        final_route_file = self.route_file
        use_dynamic_flows = False
        dynamic_start_time = 999999
        
        if self.use_curriculum_learning:
            final_route_file = self._generate_curriculum_route_file(episode_length)
            use_dynamic_flows = True
            dynamic_start_time = int(episode_length * 0.3)  # 30%时间为静态阶段
            print(f"📚 使用课程学习路由文件: {final_route_file}")
            print(f"⏰ 动态流量开始时间: {dynamic_start_time}秒")
        
        env = SumoEnv(
            net_file=self.net_file,
            route_file=final_route_file,
            out_csv_name=f"{self.output_dir}/fixed_timing",
            use_gui=False,
            num_seconds=episode_length,
            delta_time=delta_time,
            fixed_ts=True,  # 让SUMO完全按配置文件运行
            single_agent=True,
            sumo_seed=42,
            sumo_warnings=False,  # 关闭警告以减少输出
            use_dynamic_flows=use_dynamic_flows,
            dynamic_start_time=dynamic_start_time,
            flows_rate=1.0  # 默认流量倍率
        )
        return env
    
    def _generate_curriculum_route_file(self, episode_length):
        """生成课程学习路由文件"""
        try:
            # 计算静态流量文件的总流量率用于标准化
            static_total_flow = self._calculate_static_flow_rate(self.route_file)
            
            # 定义课程阶段
            stage_definitions = [
                {'name': 'low', 'duration_ratio': 0.3, 'flow_rate_multiplier': 0.5},
                {'name': 'medium', 'duration_ratio': 0.4, 'flow_rate_multiplier': 1.0},
                {'name': 'high', 'duration_ratio': 0.3, 'flow_rate_multiplier': 2.0}
            ]
            
            # 计算动态流量的平均倍数，用于标准化
            avg_multiplier = sum(stage['duration_ratio'] * stage['flow_rate_multiplier'] 
                               for stage in stage_definitions)
            print(f"📊 动态流量平均倍数: {avg_multiplier:.2f}")
            
            # 从流量模板文件中提取路线信息
            try:
                available_routes = extract_routes_from_template(self.base_template_rou_file)
                print(f"📚 从模板文件中提取到 {len(available_routes)} 条路线: {list(available_routes.keys())}")
                
                # 根据提取的路线动态构建流量分布，使用静态流量总量进行标准化
                route_distribution = {}
                if static_total_flow > 0:
                    # 基于静态流量总量和平均倍数进行标准化计算
                    # 目标：动态流量平均总量 = 静态流量总量
                    # 公式：base_flow_rate * avg_multiplier * route_count = static_total_flow
                    base_flow_rate = static_total_flow / (avg_multiplier * len(available_routes))
                    print(f"📊 流量标准化计算:")
                    print(f"   静态流量总量: {static_total_flow} veh/h")
                    print(f"   动态平均倍数: {avg_multiplier:.2f}")
                    print(f"   路线数量: {len(available_routes)}")
                    print(f"   标准化基础流量率: {base_flow_rate:.1f} veh/h")
                else:
                    # 回退到固定值
                    base_flow_rate = 100
                    print(f"📊 使用默认基础流量率: {base_flow_rate} veh/h")
                
                for route_id in available_routes.keys():
                    # 为东西向路线设置基础流量，南北向路线设置为基础流量的0.8倍
                    if 'we' in route_id.lower() or 'ew' in route_id.lower():
                        route_distribution[route_id] = base_flow_rate
                    elif 'ns' in route_id.lower() or 'sn' in route_id.lower():
                        route_distribution[route_id] = base_flow_rate * 0.8
                    else:
                        # 其他路线（如左转等）设置为基础流量的0.6倍
                        route_distribution[route_id] = base_flow_rate * 0.6
                        
            except Exception as e:
                print(f"⚠️  从模板文件提取路线失败: {e}")
                print("使用默认的硬编码路线分布")
                # 回退到硬编码的路线分布
                if static_total_flow > 0:
                    # 基于静态流量总量和平均倍数进行标准化计算
                    # 计算标准化基础流量率
                    route_count = 4  # 硬编码路线数量
                    base_flow_rate = static_total_flow / (avg_multiplier * route_count)
                    print(f"📊 硬编码路线流量标准化:")
                    print(f"   静态流量总量: {static_total_flow} veh/h")
                    print(f"   动态平均倍数: {avg_multiplier:.2f}")
                    print(f"   标准化基础流量率: {base_flow_rate:.1f} veh/h")
                    
                    # 按比例分配流量（东西向较高，南北向较低）
                    route_distribution = {
                        'route_we': base_flow_rate * 1.25,  # 东西向增加25%
                        'route_ew': base_flow_rate * 1.25,  # 东西向增加25%
                        'route_ns': base_flow_rate * 0.75,  # 南北向减少25%
                        'route_sn': base_flow_rate * 0.75,  # 南北向减少25%
                    }
                else:
                    route_distribution = {
                        'route_we': 100,
                        'route_ew': 100,
                        'route_ns': 80,
                        'route_sn': 80,
                    }
            
            # 定义输出的临时流量文件路径
            generated_rou_file = str(Path(self.output_dir) / "curriculum.rou.xml")
            print(f"📚 课程学习文件将保存到: {generated_rou_file}")
            
            # 调用生成器
            static_phase_duration = generate_curriculum_flow(
                base_route_file=self.base_template_rou_file,
                output_file=generated_rou_file,
                total_sim_seconds=int(episode_length * 0.3),  # 30%时间为静态阶段
                stage_definitions=stage_definitions,
                route_distribution=route_distribution
            )
            
            # 验证生成的动态流量文件
            dynamic_average_flow = self._calculate_dynamic_average_flow_rate(generated_rou_file, stage_definitions)
            if static_total_flow > 0:
                flow_ratio = dynamic_average_flow / static_total_flow
                print(f"📊 流量标准化验证:")
                print(f"   静态流量总量: {static_total_flow:.1f} veh/h")
                print(f"   动态流量时间加权平均: {dynamic_average_flow:.1f} veh/h")
                print(f"   流量比率: {flow_ratio:.3f} (目标: ~1.000)")
                if abs(flow_ratio - 1.0) < 0.1:
                    print(f"✅ 流量标准化成功！")
                else:
                    print(f"⚠️  流量标准化偏差较大，可能需要调整")
            
            print(f"✅ 课程学习路由文件生成成功: {generated_rou_file}")
            return generated_rou_file
            
        except Exception as e:
            print(f"❌ 生成课程学习路由文件失败: {e}")
            print(f"🔄 回退使用原始路由文件: {self.route_file}")
            return self.route_file
    
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
                    # 固定周期模式下传递空动作，让信号灯按原始配时运行
                    step_result = env.step({})
                    if len(step_result) == 5:
                        # 新版本 Gymnasium 格式: obs, reward, terminated, truncated, info
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        # 旧版本格式: obs, reward, done, info
                        obs, reward, done, info = step_result
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
                import traceback
                print(f"❌ 第 {episode + 1} 轮仿真失败: {e}")
                print(f"详细错误信息: {traceback.format_exc()}")
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
            # 打印info内容以便调试
            print(f"📊 Info内容: {info}")
            
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
                
                throughput_keys = ['system_total_throughput', 'system_total_arrived', 'total_arrived', 'total_throughput']
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
        return self.comparison_analyzer.find_agent_results()
    
    def compare_with_agent_results(self, fixed_results, agent_files):
        """与智能体结果进行对比分析"""
        return self.comparison_analyzer.compare_with_agent_results(fixed_results, agent_files)

def test_curriculum_learning():
    """测试课程学习功能"""
    print("\n🧪 开始测试课程学习功能...")
    print("=" * 50)
    
    # 配置文件路径
    net_file = "./zfdx/net/zfdx.net.xml"
    route_file = "./zfdx/net/zfdx-perhour.rou.xml"
    template_file = "./zfdx/net/zfdx.rou.template.xml"  # 模板文件
    output_dir = "./zfdx/evaluation_results_curriculum_test"
    
    # 检查文件是否存在
    files_to_check = [
        (net_file, "网络文件"),
        (route_file, "路由文件"),
        (template_file, "模板文件")
    ]
    
    for file_path, file_desc in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ {file_desc}不存在: {file_path}")
            print(f"💡 跳过课程学习测试，使用普通路由文件: {route_file}")
            template_file = route_file  # 回退到普通路由文件
            break
    
    try:
        # 创建启用课程学习的评估器
        evaluator = FixedTimingEvaluator(
            net_file=net_file,
            route_file=route_file,
            output_dir=output_dir,
            use_curriculum_learning=True,
            base_template_rou_file=template_file
        )
        
        print("\n🔄 运行课程学习测试仿真...")
        test_results = evaluator.run_fixed_timing_simulation(
            num_episodes=1,
            episode_length=1800,  # 30分钟测试
            delta_time=5
        )
        
        if test_results:
            print("\n✅ 课程学习测试成功！")
            print(f"📊 测试结果: 等待时间={test_results[0]['avg_waiting_time']:.2f}s")
            
            # 保存测试结果
            test_df = pd.DataFrame(test_results)
            test_file = f"{output_dir}/curriculum_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            test_df.to_csv(test_file, index=False)
            print(f"💾 测试结果已保存: {test_file}")
        else:
            print("❌ 课程学习测试失败")
            
    except Exception as e:
        print(f"❌ 课程学习测试出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数 - 运行固定周期评估和对比分析"""
    print("🚦 开始ZFDX路口固定周期交通信号评估")
    print("=" * 60)
    
    # 配置文件路径
    net_file = "./net/zfdx.net.xml"
    route_file = "./net/zfdx-perhour.rou.xml"
    template_file = "./net/zfdx.rou.template.xml"  # 课程学习模板文件
    output_dir = "./evaluation_results"
    
    # 课程学习开关 - 可以通过环境变量或命令行参数控制
    use_curriculum = os.getenv('USE_CURRICULUM_LEARNING', 'false').lower() == 'true'
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if '--curriculum' in sys.argv or '-c' in sys.argv:
            use_curriculum = True
            print("🎓 通过命令行参数启用课程学习")
        elif '--test-curriculum' in sys.argv or '-tc' in sys.argv:
            test_curriculum_learning()
            return
        elif '--help' in sys.argv or '-h' in sys.argv:
            print("\n📋 使用说明:")
            print("  python fixed_timing_evaluation.py              # 普通固定周期仿真")
            print("  python fixed_timing_evaluation.py --curriculum # 启用课程学习")
            print("  python fixed_timing_evaluation.py --test-curriculum # 测试课程学习功能")
            print("  python fixed_timing_evaluation.py --help       # 显示帮助信息")
            print("\n🌍 环境变量:")
            print("  USE_CURRICULUM_LEARNING=true                   # 启用课程学习")
            return

    # 检查文件是否存在
    if not os.path.exists(net_file):
        print(f"❌ 网络文件不存在: {net_file}")
        print("💡 请确保在zfdx目录下运行此脚本")
        return
    
    if not os.path.exists(route_file):
        print(f"❌ 路由文件不存在: {route_file}")
        print("💡 请确保在zfdx目录下运行此脚本")
        return
    
    # 检查课程学习模板文件
    if use_curriculum and not os.path.exists(template_file):
        print(f"⚠️  课程学习模板文件不存在: {template_file}")
        print(f"🔄 回退使用普通路由文件进行课程学习")
        template_file = route_file
    
    # 创建评估器
    evaluator = FixedTimingEvaluator(
        net_file=net_file,
        route_file=route_file,
        output_dir=output_dir,
        use_curriculum_learning=use_curriculum,
        base_template_rou_file=template_file if use_curriculum else None
    )
    
    try:
        # 运行固定周期仿真
        print("\n🔄 开始固定周期仿真...")
        print("⏱️  预计需要几分钟时间，请耐心等待...")
        
        fixed_results = evaluator.run_fixed_timing_simulation(
            num_episodes=1,      # 运行1轮仿真
            episode_length=3600, # 延长至2小时以获得更多车辆到达数据
            delta_time=5         
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
        print(f"🎓 课程学习状态: {'启用' if use_curriculum else '禁用'}")
        print("\n📋 使用说明:")
        print("  1. 固定周期仿真结果已保存为CSV格式")
        print("  2. 如有智能体结果，对比分析图表已生成")
        print("  3. 可以将智能体训练/评估的CSV结果文件放入evaluation_results目录进行对比")
        if use_curriculum:
            print("  4. 课程学习已启用，仿真包含动态流量变化")
        print("\n💡 提示: 使用 --help 查看更多选项")
        
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