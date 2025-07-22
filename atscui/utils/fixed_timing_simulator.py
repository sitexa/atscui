#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定配时仿真模块
参考fixed_timing_evaluation.py的成功实现，提供模块化的固定配时仿真功能

作者: ATSCUI系统
日期: 2024
"""

import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Iterator, Tuple

from sumo_core.envs.sumo_env import SumoEnv
from atscui.utils.flow_generator import generate_curriculum_flow, extract_routes_from_template
from atscui.logging_manager import get_logger

class FixedTimingSimulator:
    """固定配时仿真器 - 提供模块化的固定配时仿真功能"""
    
    def __init__(self, config, logger=None):
        """
        初始化固定配时仿真器
        
        Args:
            config: 配置对象，包含网络文件、路由文件等信息
            logger: 日志记录器，如果为None则创建新的
        """
        self.config = config
        self.logger = logger or get_logger('fixed_timing_simulator')
        
        # 仿真参数
        self.episode_length = getattr(config, 'num_seconds', 3600)
        self.delta_time = 5
        self.num_episodes = 1  # 固定配时通常只需要运行一次
        
        # 流量文件路径
        self.route_file_path = config.rou_file
        self.use_dynamic_flows = False
        self.dynamic_start_time = 0
        self.flows_rate = getattr(config, 'dynamic_flows_rate', 10)
        
        self.logger.info(f"🚦 固定配时仿真器初始化完成")
        self.logger.info(f"📁 网络文件: {config.net_file}")
        self.logger.info(f"📁 路由文件: {config.rou_file}")
        self.logger.info(f"⏱️  仿真时长: {self.episode_length}秒")
    
    def prepare_traffic_files(self) -> Iterator[Tuple[int, str]]:
        """
        准备流量文件，如果启用课程学习则生成动态流量文件
        
        Yields:
            Tuple[int, str]: (进度百分比, 状态消息)
        """
        yield 5, "准备流量文件..."
        
        if self.config.use_curriculum_learning and hasattr(self.config, 'base_template_rou_file') and self.config.base_template_rou_file:
            yield 10, "启用课程学习，正在生成动态流量文件..."
            self.logger.info("=====使用课程学习: 生成动态流量文件=====")
            
            try:
                # 定义课程阶段（参考env_creator.py和fixed_timing_evaluation.py）
                stage_definitions = [
                    {'name': 'low', 'duration_ratio': 0.3, 'flow_rate_multiplier': 0.5},
                    {'name': 'medium', 'duration_ratio': 0.4, 'flow_rate_multiplier': 1.0},
                    {'name': 'high', 'duration_ratio': 0.3, 'flow_rate_multiplier': 2.0}
                ]
                
                # 动态提取路线信息或使用默认配置
                try:
                    available_routes = extract_routes_from_template(self.config.base_template_rou_file)
                    self.logger.info(f"从模板文件提取到 {len(available_routes)} 条路线: {list(available_routes.keys())}")
                    
                    # 根据提取的路线动态生成路线分布
                    base_flow = getattr(self.config, 'base_flow_rate', 300)
                    route_distribution = {}
                    for route_id in available_routes.keys():
                        # 为南北向路线设置较低的基础流量（80%）
                        if 'ns' in route_id.lower() or 'sn' in route_id.lower():
                            route_distribution[route_id] = base_flow * 0.8
                        else:
                            route_distribution[route_id] = base_flow
                            
                except Exception as e:
                    self.logger.warning(f"无法从模板文件提取路线信息，使用默认配置: {e}")
                    # 使用默认配置
                    base_flow = getattr(self.config, 'base_flow_rate', 300)
                    route_distribution = {
                        'route_we': base_flow,
                        'route_ew': base_flow,
                        'route_ns': base_flow * 0.8,
                        'route_sn': base_flow * 0.8,
                    }
                
                self.logger.info(f"使用基础流量: {base_flow}, 路线分布: {route_distribution}")
                
                # 定义输出的流量文件路径
                generated_rou_file = str(Path(self.config.rou_file).parent / "curriculum.rou.xml")
                self.logger.info(f"生成的课程文件将保存到: {generated_rou_file}")
                
                yield 15, "正在生成课程学习流量文件..."
                
                # 调用生成器
                static_phase_duration = generate_curriculum_flow(
                    base_route_file=self.config.base_template_rou_file,
                    output_file=generated_rou_file,
                    total_sim_seconds=int(self.episode_length * getattr(self.config, 'static_phase_ratio', 0.8)),
                    stage_definitions=stage_definitions,
                    route_distribution=route_distribution
                )
                
                # 更新配置以使用新生成的文件和参数
                self.route_file_path = generated_rou_file
                self.use_dynamic_flows = True
                self.dynamic_start_time = static_phase_duration
                
                self.logger.info(f"静态阶段将运行 {static_phase_duration} 秒，然后切换到动态流量")
                self.logger.info(f"课程学习流量文件已生成: {self.route_file_path}")
                
                yield 20, f"课程学习流量文件生成完成: {os.path.basename(generated_rou_file)}"
                
            except Exception as e:
                self.logger.error(f"生成课程学习流量文件失败: {e}")
                yield 20, f"生成流量文件失败，使用原始文件: {e}"
                # 回退到原始文件
                self.route_file_path = self.config.rou_file
                self.use_dynamic_flows = False
                self.dynamic_start_time = 0
        
        elif os.path.isdir(self.config.rou_file):
            # 在目录中查找.rou.xml文件
            rou_files = [f for f in os.listdir(self.config.rou_file) if f.endswith('.rou.xml')]
            if rou_files:
                self.route_file_path = os.path.join(self.config.rou_file, rou_files[0])
                self.logger.info(f"在目录中找到路由文件: {self.route_file_path}")
            else:
                raise FileNotFoundError(f"在目录 {self.config.rou_file} 中未找到.rou.xml文件")
        
        yield 25, "流量文件准备完成"
    
    def create_fixed_timing_env(self):
        """
        创建固定配时仿真环境
        参考fixed_timing_evaluation.py的成功实现
        """
        env = SumoEnv(
            net_file=self.config.net_file,
            route_file=self.route_file_path,
            out_csv_name=None,  # 不使用内置CSV输出
            use_gui=getattr(self.config, 'gui', False),
            num_seconds=self.episode_length,
            delta_time=self.delta_time,
            yellow_time=3,
            min_green=10,
            max_green=60,
            fixed_ts=True,  
            single_agent=True, 
            sumo_seed=42,
            sumo_warnings=False,
            # 动态流量参数（如果启用课程学习）
            use_dynamic_flows=self.use_dynamic_flows,
            dynamic_start_time=self.dynamic_start_time,
            flows_rate=self.flows_rate
        )
        return env
    
    def run_simulation(self) -> Iterator[Tuple[int, str]]:
        """
        运行固定配时仿真
        
        Yields:
            Tuple[int, str]: (进度百分比, 状态消息)
        """
        # 准备流量文件
        yield from self.prepare_traffic_files()
        
        yield 30, "开始固定配时仿真..."
        self.logger.info(f"开始固定配时仿真，仿真时长: {self.episode_length}秒")
        
        all_results = []
        
        try:
            # 创建环境
            env = self.create_fixed_timing_env()
            
            # 重置环境
            obs = env.reset()
            done = False
            step = 0
            
            episode_metrics = {
                'episode': 1,
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
            total_steps = self.episode_length // self.delta_time
            
            while not done and step < total_steps:
                # 固定周期模式下不需要动作，环境会自动按照固定周期运行
                obs, reward, terminated, truncated, info = env.step({})
                done = terminated or truncated  # 正确处理done条件
                step += 1
                
                # 更新进度
                if step % 100 == 0:
                    progress = 30 + int((step / total_steps) * 60)  # 30-90%的进度范围
                    elapsed_time = step * self.delta_time
                    yield progress, f"仿真进行中... {elapsed_time}/{self.episode_length}秒 ({progress-30:.1f}%)"
            
            # 更新实际运行的步数
            episode_metrics['total_steps'] = step
            
            # 提取最终指标
            episode_metrics.update(self._extract_final_metrics(info))
            episode_metrics['total_steps'] = step
            
            all_results.append(episode_metrics)
            
            # 保存结果到实例变量，供外部访问
            self._last_results = episode_metrics
            
            # 保存结果到CSV文件
            try:
                csv_path = self.save_results(all_results)
                yield 95, f"结果已保存到: {os.path.basename(csv_path)}"
            except Exception as e:
                self.logger.warning(f"保存结果失败: {e}")
            
            yield 100, f"仿真完成 - 等待时间: {episode_metrics['avg_waiting_time']:.2f}s"
            self.logger.info(f"✅ 固定配时仿真完成 - 等待时间: {episode_metrics['avg_waiting_time']:.2f}s")
            
            # 关闭环境
            env.close()
            
        except Exception as e:
            self.logger.error(f"❌ 固定配时仿真失败: {e}")
            yield 100, f"仿真失败: {e}"
            raise
    
    def get_last_results(self) -> Dict[str, float]:
        """
        获取最后一次仿真的结果
        
        Returns:
            Dict[str, float]: 仿真结果指标
        """
        return getattr(self, '_last_results', {})
    
    def _extract_final_metrics(self, info) -> Dict[str, float]:
        """
        从仿真信息中提取最终指标
        参考fixed_timing_evaluation.py的实现
        """
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
                # 使用正确的键名提取指标
                if 'system_mean_waiting_time' in info:
                    metrics['avg_waiting_time'] = float(info['system_mean_waiting_time'])
                
                if 'system_mean_speed' in info:
                    metrics['avg_speed'] = float(info['system_mean_speed'])
                
                if 'system_total_stopped' in info:
                    metrics['avg_queue_length'] = float(info['system_total_stopped'])
                
                if 'system_total_throughput' in info:
                    metrics['total_throughput'] = float(info['system_total_throughput'])
                
                if 'system_mean_travel_time' in info:
                    metrics['avg_travel_time'] = float(info['system_mean_travel_time'])
                
                # 燃料消耗和CO2排放（如果可用）
                if 'system_total_fuel_consumption' in info:
                    metrics['total_fuel_consumption'] = float(info['system_total_fuel_consumption'])
                if 'system_total_co2_emission' in info:
                    metrics['total_co2_emission'] = float(info['system_total_co2_emission'])
        
        except Exception as e:
            self.logger.warning(f"⚠️  提取指标时出错: {e}")
        
        return metrics
    
    def save_results(self, results: list, output_dir: str = None, simulation_name: str = None) -> str:
        """
        保存仿真结果到CSV文件
        
        Args:
            results: 仿真结果列表
            output_dir: 输出目录，如果为None则使用配置中的csv_path目录
            simulation_name: 仿真名称，用于生成文件名
            
        Returns:
            str: 保存的文件路径
        """
        if not results:
            raise ValueError("没有仿真结果可保存")
        
        # 确定输出目录
        if output_dir is None:
            output_dir = os.path.dirname(self.config.csv_path) if hasattr(self.config, 'csv_path') else "./outs"
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if simulation_name:
            filename = f"fixtime_{simulation_name}_{timestamp}.csv"
        else:
            filename = f"fixtime_simulation_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # 保存结果
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"💾 固定配时仿真结果已保存: {filepath}")
        return filepath