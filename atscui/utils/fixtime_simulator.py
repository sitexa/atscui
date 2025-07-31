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
import xml.etree.ElementTree as ET

from sumo_core.envs.sumo_env import SumoEnv
from atscui.utils.flow_generator import generate_curriculum_flow, extract_routes_from_template
from atscui.logging_manager import get_logger
from atscui.config import BaseConfig

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
        self.logger = get_logger('fixed_timing_simulator')
        
        # 仿真参数
        self.episode_length = getattr(config, 'num_seconds', 3600)
        self.delta_time = getattr(config, 'delta_time', 5)
        self.num_episodes = 5  # fixtime 只要进行一次仿真
        
        # 流量文件路径
        self.route_file_path = config.rou_file
        
        self.logger.info(f"固定配时仿真器初始化完成，仿真时长: {self.episode_length}秒")
    
    def _calculate_static_flow_rate(self, route_file):
        """计算静态流量文件的总流量率"""
        try:
            # 检查路径是否为文件
            if not os.path.isfile(route_file):
                self.logger.warning(f"路径不是有效的文件: {route_file}")
                return 0
                
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
            
            # 静态流量文件分析完成
            return total_flow_rate
            
        except Exception as e:
            self.logger.warning(f"计算静态流量率失败: {e}")
            return 0
    
    def _calculate_dynamic_average_flow_rate(self, route_file, stage_definitions):
        """计算动态流量文件的平均流量率（考虑时间权重）"""
        try:
            # 检查路径是否为文件
            if not os.path.isfile(route_file):
                self.logger.warning(f"路径不是有效的文件: {route_file}")
                return 0
                
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
                # 计算阶段流量权重
            return total_weighted_flow
            
        except Exception as e:
            self.logger.warning(f"计算动态平均流量率失败: {e}")
            return 0
    
    def prepare_traffic_files(self) -> Iterator[Tuple[int, str]]:
        """
        准备流量文件，如果启用课程学习则生成动态流量文件
        
        Yields:
            Tuple[int, str]: (进度百分比, 状态消息)
        """
        yield 5, "准备流量文件..."
        
        if self.config.use_curriculum_learning:
            yield 10, "启用课程学习，正在生成动态流量文件..."
            self.logger.info("使用课程学习: 生成动态流量文件")
            
            try:
                # 在课程学习模式下，使用模板文件计算静态流量率用于标准化
                # 如果模板文件不存在，则跳过静态流量计算
                static_total_flow = 0
                if os.path.isfile(self.config.rou_file):
                    static_total_flow = self._calculate_static_flow_rate(self.config.rou_file)
                else:
                    self.logger.warning(f"路由文件不存在，跳过静态流量计算: {self.config.rou_file}")
                    static_total_flow = 1000  # 使用默认值
                
                # 定义课程阶段（参考env_creator.py和fixed_timing_evaluation.py）
                stage_definitions = [
                    {'name': 'low', 'duration_ratio': 0.3, 'flow_rate_multiplier': 0.5},
                    {'name': 'medium', 'duration_ratio': 0.4, 'flow_rate_multiplier': 1.0},
                    {'name': 'high', 'duration_ratio': 0.3, 'flow_rate_multiplier': 2.0}
                ]
                
                # 计算动态流量的平均倍数，用于标准化
                avg_multiplier = sum(stage['duration_ratio'] * stage['flow_rate_multiplier'] for stage in stage_definitions)
                
                # 动态提取路线信息或使用默认配置
                try:
                    available_routes = extract_routes_from_template(self.config.rou_file)
                    
                    # 根据提取的路线动态构建流量分布，使用静态流量总量进行标准化
                    route_distribution = {}
                    if static_total_flow > 0:
                        # 基于静态流量总量和平均倍数进行标准化计算
                        base_flow_rate = static_total_flow / (avg_multiplier * len(available_routes))
                    else:
                        # 回退到固定值
                        base_flow_rate = 100
                    
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
                    self.logger.warning(f"从模板文件提取路线失败: {e}")
                    # 回退到硬编码的路线分布
                    if static_total_flow > 0:
                        route_count = 4  # 硬编码路线数量
                        base_flow_rate = static_total_flow / (avg_multiplier * route_count)
                        
                        # 按比例分配流量（东西向较高，南北向较低）
                        route_distribution = {
                            'route_we': base_flow_rate * 1.25,
                            'route_ew': base_flow_rate * 1.25,
                            'route_ns': base_flow_rate * 0.75,
                            'route_sn': base_flow_rate * 0.75,
                        }
                    else:
                        route_distribution = {
                            'route_we': 100,
                            'route_ew': 100,
                            'route_ns': 80,
                            'route_sn': 80,
                        }
                
                # 定义输出的流量文件路径，使用"路口-算法-datestamp"格式
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                
                # 生成课程文件名
                curriculum_filename = f"{self.config.cross_name}-fixtime-{timestamp}.rou.xml"
                generated_rou_file = str(self.config.config_dir / curriculum_filename)

                yield 15, "正在生成课程学习流量文件..."
                
                # 调用生成器
                static_phase_duration = generate_curriculum_flow(
                    base_route_file=self.config.rou_file,
                    output_file=generated_rou_file,
                    total_sim_seconds=int(self.episode_length * getattr(self.config, 'curriculum_static_ratio', 0.8)),
                    stage_definitions=stage_definitions,
                    route_distribution=route_distribution
                )
                
                # 验证文件是否成功生成
                if not os.path.exists(generated_rou_file):
                    self.logger.error(f"课程文件生成失败，文件不存在: {generated_rou_file}")
                    raise FileNotFoundError(f"课程文件生成失败: {generated_rou_file}")
                
                # 验证生成的动态流量文件
                dynamic_average_flow = self._calculate_dynamic_average_flow_rate(generated_rou_file, stage_definitions)
                if static_total_flow > 0:
                    flow_ratio = dynamic_average_flow / static_total_flow
                    if abs(flow_ratio - 1.0) >= 0.1:
                        self.logger.warning(f"流量标准化偏差较大: {flow_ratio:.3f}")
                
                # 更新配置以使用新生成的文件和参数
                self.route_file_path = generated_rou_file
                self.dynamic_start_time = static_phase_duration
                
                self.logger.info(f"l247===================课程学习路由文件生成成功: {self.route_file_path}")
                
                yield 20, f"课程学习流量文件生成完成: {os.path.basename(generated_rou_file)}"
                
            except Exception as e:
                self.logger.error(f"生成课程学习流量文件失败: {e}")
                yield 20, f"生成流量文件失败，使用原始文件: {e}"
                # 回退到原始文件
                self.route_file_path = self.config.rou_file
                self.dynamic_start_time = 0
        else:
            # 不使用课程学习，使用原始路由文件
            self.logger.info("未启用课程学习，使用原始路由文件")
            # 验证路由文件路径
            if not os.path.isfile(self.config.rou_file):
                raise FileNotFoundError(f"路由文件不存在: {self.config.rou_file}")
            self.route_file_path = self.config.rou_file
            self.dynamic_start_time = 0
        
        yield 25, "流量文件准备完成"
    
    def create_fixed_timing_env(self):
        """
        创建固定配时仿真环境
        参考fixed_timing_evaluation.py的成功实现
        """
        # 处理课程学习相关参数
        use_curriculum_learning = getattr(self.config, 'use_curriculum_learning', False)
        dynamic_start_time = getattr(self, 'dynamic_start_time', 999999)
        
        if use_curriculum_learning:
            self.logger.info(f"使用课程学习路由文件: {self.route_file_path}")
        
        env = SumoEnv(
            net_file=self.config.net_file,
            route_file=self.route_file_path,
            out_csv_name=self.config.csv_path,
            use_gui=getattr(self.config, 'gui', False),
            num_seconds=self.episode_length,
            delta_time=self.delta_time,
            fixed_ts=True,  # 让SUMO完全按配置文件运行
            single_agent=True, 
            sumo_seed=42,
            sumo_warnings=False,  # 关闭警告以减少输出
            use_dynamic_flows=use_curriculum_learning,
            dynamic_start_time=dynamic_start_time,
            flows_rate=1.0  # 默认流量倍率
        )
        return env
    
    def run_simulation(self, num_episodes=None) -> Iterator[Tuple[int, str]]:
        """
        运行固定配时仿真
        
        Args:
            num_episodes: 仿真轮数，如果为None则使用配置中的默认值
            
        Yields:
            Tuple[int, str]: (进度百分比, 状态消息)
        """
        # 准备流量文件
        yield from self.prepare_traffic_files()
        
        # 确定仿真轮数
        if num_episodes is None:
            num_episodes = self.num_episodes
        
        yield 30, f"开始固定配时仿真... ({num_episodes}轮)"
        self.logger.info(f"开始运行固定周期仿真")
        self.logger.info(f"仿真参数: {num_episodes}轮 × {self.episode_length}秒/轮")
        
        all_results = []
        
        for episode in range(num_episodes):
            episode_progress_start = 30 + (episode * 60) // num_episodes
            episode_progress_end = 30 + ((episode + 1) * 60) // num_episodes
            
            yield episode_progress_start, f"运行第 {episode + 1}/{num_episodes} 轮仿真..."
            self.logger.info(f"运行第 {episode + 1}/{num_episodes} 轮仿真")
            
            try:
                # 创建环境
                env = self.create_fixed_timing_env()
                
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
                        elapsed_time = step * self.delta_time
                        episode_progress = episode_progress_start + int((elapsed_time / self.episode_length) * (episode_progress_end - episode_progress_start))
                        yield episode_progress, f"第{episode + 1}轮: {elapsed_time}/{self.episode_length}秒 ({(elapsed_time/self.episode_length)*100:.1f}%)"
                
                # 提取最终指标
                episode_metrics.update(self._extract_final_metrics(info, episode + 1, step))
                episode_metrics['total_steps'] = step
                
                all_results.append(episode_metrics)
                
                self.logger.info(f"第 {episode + 1} 轮完成 - 等待时间: {episode_metrics['avg_waiting_time']:.2f}s")
                
                # 关闭环境
                env.close()
                
            except Exception as e:
                self.logger.error(f"第 {episode + 1} 轮仿真失败: {e}")
                continue
        
        try:
            if all_results:
                # 保存结果到实例变量，供外部访问
                # 始终保存最后一轮的结果作为字典，确保get方法可用
                self._last_results = all_results[-1]
                
                # 保存结果到CSV文件
                csv_path = self.save_results(all_results)
                yield 95, f"结果已保存到: {os.path.basename(csv_path)}"
                
                # 计算平均指标
                avg_waiting_time = sum(r['avg_waiting_time'] for r in all_results) / len(all_results)
                
                yield 100, f"仿真完成！共{len(all_results)}轮，平均等待时间: {avg_waiting_time:.2f}s"
                self.logger.info(f"固定周期仿真完成，共成功运行 {len(all_results)} 轮")
            else:
                yield 100, "所有仿真轮次都失败了"
                self.logger.error(f"所有仿真轮次都失败了")
                
        except Exception as e:
            self.logger.warning(f"保存结果失败: {e}")
            yield 100, f"仿真完成但保存失败: {e}"
    
    def get_last_results(self) -> Dict[str, float]:
        """
        获取最后一次仿真的结果
        
        Returns:
            Dict[str, float]: 仿真结果指标
        """
        return getattr(self, '_last_results', {})
    
    def _extract_final_metrics(self, info, episode=None, step=None) -> Dict[str, float]:
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
            # 检查info的类型并进行相应处理
            if isinstance(info, dict):
                # 从字典中提取系统级指标
                
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
                
                if 'system_mean_travel_time' in info:
                    metrics['avg_travel_time'] = float(info['system_mean_travel_time'])
                
                # 燃料消耗和CO2排放（如果可用）
                fuel_keys = ['system_total_fuel_consumption', 'total_fuel_consumption', 'fuel_consumption']
                for key in fuel_keys:
                    if key in info:
                        metrics['total_fuel_consumption'] = float(info[key])
                        break
                        
                co2_keys = ['system_total_co2_emission', 'total_co2_emission', 'co2_emission']
                for key in co2_keys:
                    if key in info:
                        metrics['total_co2_emission'] = float(info[key])
                        break
                        
            elif isinstance(info, list):
                # 如果info是列表，尝试从列表中的字典元素提取指标
                self.logger.debug(f"Info是列表类型，长度: {len(info)}")
                for item in info:
                    if isinstance(item, dict):
                        # 递归调用自身处理字典项
                        item_metrics = self._extract_final_metrics(item, episode, step)
                        # 合并非零指标
                        for key, value in item_metrics.items():
                            if value > 0:
                                metrics[key] = max(metrics[key], value)
                        break
            else:
                self.logger.debug(f"Info类型未知: {type(info)}")
        
        except Exception as e:
            self.logger.warning(f"提取指标时出错: {e}")
        
        return metrics
    
    def save_results(self, results: list, output_dir: str = None) -> str:
        """
        保存仿真结果到CSV文件
        
        Args:
            results: 仿真结果列表
            output_dir: 输出目录，如果为None则使用配置中的csv_path目录
            
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
        
        # 生成文件名：路口-算法-流量方案-timestamp格式
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 从配置中提取路口名称
        intersection_name = "unknown"
        if hasattr(self.config, 'net_file'):
            net_file_name = os.path.basename(self.config.net_file)
            if net_file_name.endswith('.net.xml'):
                intersection_name = net_file_name.replace('.net.xml', '')
        
        # 确定流量方案类型
        flow_type = "static"  # 默认为静态流量
        if hasattr(self.config, 'use_curriculum_learning') and self.config.use_curriculum_learning:
            flow_type = "curriculum"
        
        # 生成标准化文件名：路口-算法-流量方案-timestamp
        filename = f"{intersection_name}-FIXTIME-{flow_type}-{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # 保存结果
        df = pd.DataFrame(results)
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"固定配时仿真结果已保存: {filepath}")
        return filepath