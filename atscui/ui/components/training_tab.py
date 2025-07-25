import os
import shlex
from pathlib import Path

import gradio as gr
import json
import traceback
import numpy as np
from typing import Dict, Any, Optional, Tuple, Generator, Iterator

from atscui.config.base_config import AlgorithmConfig, RunningConfig, DQNConfig, PPOConfig, A2CConfig, SACConfig, BaseConfig
from atscui.environment.env_creator import createEnv
from atscui.models.agent_creator import AgentFactory, createAgent
from atscui.utils.file_utils import file_manager, extract_crossname_from_netfile, ensure_dir
from atscui.utils.utils import common_utils, utility_manager
from atscui.utils.visualization import plot_process, plot_predict
from atscui.test_run_model import run_model
from atscui.exceptions import (
    TrainingError, EnvironmentError, ModelError, 
    FileOperationError, ValidationError, ConfigurationError
)
from atscui.logging_manager import get_logger, log_manager
from atscui.config.config_manager import ConfigManager
from atscui.utils.fixtime_simulator import FixedTimingSimulator


class ParameterParser:
    """参数解析器
    
    负责解析和验证训练参数，创建配置对象。
    """
    
    def __init__(self):
        self.logger = get_logger('parameter_parser')
        self.config_manager = ConfigManager()
    
    def parse_and_validate(self, **kwargs) -> BaseConfig:
        """解析并验证参数
        
        Args:
            **kwargs: 训练参数
            
        Returns:
            TrainingConfig: 验证后的配置对象
            
        Raises:
            ValidationError: 参数验证失败时抛出
            ConfigurationError: 配置创建失败时抛出
        """
        try:
            # 验证必需参数
            self._validate_required_params(kwargs)
            
            # 验证文件路径
            self._validate_file_paths(kwargs)
            
            # 验证数值参数
            self._validate_numeric_params(kwargs)
            
            # 创建配置对象
            config = self._create_config(kwargs)
            
            # 使用配置管理器验证
            self.config_manager.validate_config(config)
            
            return config
            
        except (ValidationError, ConfigurationError):
            raise
        except Exception as e:
            error_msg = f"参数解析失败: {e}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def _validate_required_params(self, params: Dict[str, Any]) -> None:
        """验证必需参数"""
        required_params = ['net_file', 'algo_name', 'operation']
        
        for param in required_params:
            if not params.get(param):
                raise ValidationError(f"必需参数缺失: {param}")
        
        # 根据是否使用课程学习来验证rou_file或curriculum_template_file
        if params.get('use_curriculum_learning'):
            if not params.get('curriculum_template_file'):
                raise ValidationError("使用课程学习时，路线模板文件 (curriculum_template_file) 不能为空")
        else:
            if not params.get('rou_file'):
                raise ValidationError("必需参数缺失: rou_file")
    
    def _validate_file_paths(self, params: Dict[str, Any]) -> None:
        """验证文件路径"""
        # 验证网络文件
        if params.get('net_file'):
            utility_manager.validate_file_path(params['net_file'], must_exist=True)
        
        # 根据是否使用课程学习来验证路由文件或模板文件
        if params.get('use_curriculum_learning'):
            if params.get('curriculum_template_file'):
                utility_manager.validate_file_path(params['curriculum_template_file'], must_exist=True)
            else:
                raise ValidationError("使用课程学习时，路线模板文件不能为空")
        else:
            # 验证路由文件 (非课程学习模式)
            if params.get('rou_file'):
                utility_manager.validate_file_path(params['rou_file'], must_exist=True)
    
    def _validate_numeric_params(self, params: Dict[str, Any]) -> None:
        """验证数值参数"""
        # 验证时间步数
        total_timesteps = params.get('total_timesteps', 0)
        if total_timesteps <= 0:
            raise ValidationError("总时间步数必须大于0")
        
        # 验证评估回合数
        n_eval_episodes = params.get('n_eval_episodes', 0)
        if n_eval_episodes <= 0:
            raise ValidationError("评估回合数必须大于0")
        
        # 验证仿真秒数
        num_seconds = params.get('num_seconds', 0)
        if num_seconds <= 0:
            raise ValidationError("仿真秒数必须大于0")
    
    def _create_config(self, params: Dict[str, Any]) -> BaseConfig:
        """创建配置对象"""
        try:
            net_file = params.get('net_file')
            algo_name = params.get('algo_name', 'DQN')  # 默认算法模型
            
            _cross_name = extract_crossname_from_netfile(net_file)
            # 在文件名中包含phase_control信息
            cvs_file = _cross_name + "-" + algo_name
            csv_path = os.path.join(ensure_dir("outs"), cvs_file)
            model_file = _cross_name + "-model-" + algo_name + ".zip"
            model_path = os.path.join(ensure_dir("models"), model_file)
            predict_file = _cross_name + "-predict-" + algo_name + ".json"
            predict_path = os.path.join(ensure_dir("predicts"), predict_file)
            eval_file = _cross_name + "-eval-" + algo_name + ".txt"
            eval_path = os.path.join(ensure_dir("evals"), eval_file)
            tensorboard_logpath = ensure_dir(params.get('tensorboard_logs', 'logs'))
            
            # 根据算法名称选择相应的配置类
            config_class = self._get_config_class(algo_name)
            
            # 创建基础配置参数
            base_config_params = {
                'net_file': net_file,
                'rou_file': params.get('rou_file'),
                'csv_path': csv_path,
                'model_path': model_path,
                'predict_path': predict_path,
                'eval_path': eval_path,
                'single_agent': params.get('single_agent', True),
                'gui': params.get('gui', True),
                'render_mode': params.get('render_mode', None),
                'operation': params.get('operation', 'TRAIN'),
                'algo_name': algo_name,
                'tensorboard_logs': tensorboard_logpath,
                'use_curriculum_learning': params.get('use_curriculum_learning', False),
                'base_template_rou_file': params.get('curriculum_template_file'),
                'static_phase_ratio': params.get('curriculum_static_ratio'),
                'base_flow_rate': params.get('curriculum_base_flow'),
                'dynamic_flows_rate': params.get('curriculum_dynamic_rate')
            }
            
            # 如果用户提供了参数，则覆盖默认值
            if 'total_timesteps' in params:
                base_config_params['total_timesteps'] = params['total_timesteps']
            if 'num_seconds' in params:
                base_config_params['num_seconds'] = params['num_seconds']
            if 'n_steps' in params:
                base_config_params['n_steps'] = params['n_steps']
            if 'n_eval_episodes' in params:
                base_config_params['n_eval_episodes'] = params['n_eval_episodes']
            if 'prediction_steps' in params:
                base_config_params['prediction_steps'] = params['prediction_steps']
            
            config = config_class(**base_config_params)
            return config
            
        except Exception as e:
            raise ConfigurationError(f"创建配置对象失败: {e}")
    
    def _get_config_class(self, algo_name: str):
        """根据算法名称获取相应的配置类"""
        config_mapping = {
            'DQN': DQNConfig,
            'PPO': PPOConfig,
            'A2C': A2CConfig,
            'SAC': SACConfig,
            'FIXTIME': AlgorithmConfig
        }
        
        # 如果算法名称不在映射中，使用默认的AlgorithmConfig
        return config_mapping.get(algo_name, AlgorithmConfig)


# 全局参数解析器实例
parameter_parser = ParameterParser()


def parseParams(net_file,  # 网络模型
                rou_file,  # 交通需求 (现在是rou文件生成的目录)
                algo_name="DQN",  # 算法名称
                operation="TRAIN",  # 操作名称
                tensorboard_logs="logs",  # tensorboard_logs folder
                single_agent=True,  # 单智能体
                num_seconds=10000,  # 每回合episode仿真步(时长)
                n_eval_episodes=10,  # 评估回合数
                n_steps=1024,  # A2C价值网络更新间隔时间步
                total_timesteps=864_000,  # 总训练时间步（1天)
                gui=True,  # 图形界面
                render_mode=None,  # 渲染模式
                prediction_steps=100,
                use_curriculum_learning=False, # 是否使用课程学习
                curriculum_template_file=None, # 课程学习的rou模板文件
                curriculum_total_seconds=None, # 课程学习的总仿真秒数
                curriculum_static_ratio=None, # 课程学习的静态阶段时长占比
                curriculum_base_flow=None, # 课程学习的基础流率
                curriculum_dynamic_rate=None, # 课程学习的动态阶段生成速率
                ) -> BaseConfig:
    """解析参数的便捷函数
    
    保持向后兼容性的包装函数。
    """
    return parameter_parser.parse_and_validate(
        net_file=net_file,
        rou_file=rou_file,
        algo_name=algo_name,
        operation=operation,
        tensorboard_logs=tensorboard_logs,
        single_agent=single_agent,
        num_seconds=num_seconds,
        n_eval_episodes=n_eval_episodes,
        n_steps=n_steps,
        total_timesteps=total_timesteps,
        gui=gui,
        render_mode=render_mode,
        prediction_steps=prediction_steps,
        use_curriculum_learning=use_curriculum_learning,
        curriculum_template_file=curriculum_template_file,
        curriculum_total_seconds=curriculum_total_seconds,
        curriculum_static_ratio=curriculum_static_ratio,
        curriculum_base_flow=curriculum_base_flow,
        curriculum_dynamic_rate=curriculum_dynamic_rate
    )


class TrainingTab:
    """训练标签页
    
    提供模型训练、评估和预测的用户界面。
    """
    
    def __init__(self):
        self.logger = get_logger('training_tab')
        self.agent_factory = AgentFactory()
        self.current_training_logger = None
        self.is_training = False
        
        # UI组件引用
        self.network_file = None
        self.demand_file = None
        self.progress = None
        self.output_msg = None
    
    def render(self) -> None:
        """渲染训练标签页UI"""
        try:
            with gr.Row():
                with gr.Column(scale=2):
                    network_file = gr.File(
                        label="路网核心文件 (.net.xml)", 
                        value="zfdx/net/zfdx.net.xml", 
                        file_types=[".xml", ".net.xml"]
                    )
                    with gr.Row():
                        algorithm = gr.Dropdown(
                            choices=self.agent_factory.get_supported_algorithms(),
                            value="PPO", 
                            label="算法模型"
                        )
                        operation = gr.Dropdown(
                            ["TRAIN", "EVAL", "PREDICT", "ALL"], 
                            value="TRAIN", 
                            label="运行功能"
                        )
            
            with gr.Accordion("流量配置参数 (Traffic Configuration Parameters)", open=True):
                with gr.Row():
                    # 静态/动态流量开关
                    traffic_mode = gr.Radio(
                        choices=["静态流量", "动态流量"],
                        value="动态流量",
                        label="流量模式"
                    )
                with gr.Row():
                    route_file = gr.File(
                        label="路线流量文件 (.rou.xml)", 
                        value="zfdx/net/zfdx-perhour.rou.xml", 
                        file_types=[".xml", ".rou.xml"]
                    )
                    total_simulation_seconds = gr.Number(
                        value=3600, 
                        label="仿真总秒数 (Total Seconds)"
                    )
                # 动态流量参数（仅在动态流量模式下显示）
                with gr.Row(visible=False) as dynamic_params_row:
                    static_phase_ratio = gr.Slider(
                        0.1, 1.0, 
                        value=0.7, 
                        step=0.1, 
                        label="静态阶段时长占比 (Static Phase Ratio)"
                    )
                    base_flow_rate = gr.Number(
                        value=300, 
                        label="基础流率 (Base Flow Rate)"
                    )
                    dynamic_phase_rate = gr.Number(
                        value=10, 
                        label="动态阶段生成速率 (Dynamic Phase Rate)"
                    )

            with gr.Accordion("训练与仿真参数 (Training & Simulation Parameters)", open=True):
                with gr.Row():
                    total_timesteps = gr.Slider(
                        5000, 5_000_000, 
                        value=1_000_000, 
                        step=5000, 
                        label="总训练步数"
                    )
                    prediction_steps = gr.Slider(
                        50, 500, 
                        value=100, 
                        step=10, 
                        label="预测步数"
                    )
            
            gui_checkbox = gr.Checkbox(label="显示GUI界面", value=False)
            
            with gr.Row():
                run_button = gr.Button("开始运行", variant="primary")
                stop_button = gr.Button("停止训练", variant="secondary")
            
            progress = gr.Slider(
                minimum=0, maximum=100, value=0, 
                label="进度", interactive=False
            )
            output_msg = gr.Textbox(label="输出信息", lines=8, max_lines=20)
            
            # 存储组件引用
            self.progress = progress
            self.output_msg = output_msg
            
            # 流量模式切换事件
            def toggle_dynamic_params(mode):
                return gr.update(visible=(mode == "动态流量"))
            
            traffic_mode.change(
                toggle_dynamic_params,
                inputs=[traffic_mode],
                outputs=[dynamic_params_row]
            )
            
            # 绑定事件
            run_button.click(
                self.run_training,
                inputs=[
                    network_file, route_file, algorithm, operation,
                    total_timesteps, total_simulation_seconds, gui_checkbox, prediction_steps,
                    traffic_mode, static_phase_ratio, base_flow_rate, dynamic_phase_rate
                ],
                outputs=[progress, output_msg]
            )
            
            stop_button.click(
                self.stop_training,
                outputs=[output_msg]
            )
            
        except Exception as e:
            self.logger.error(f"渲染训练标签页失败: {e}")
            raise

    def run_training(self,
                     network_file,
                     route_file,
                     algorithm,
                     operation,
                     total_timesteps,
                     total_simulation_seconds,
                     gui_checkbox,
                     prediction_steps,
                     traffic_mode,
                     static_phase_ratio,
                     base_flow_rate,
                     dynamic_phase_rate) -> Iterator[Tuple[int, str]]:
        """运行训练任务
        
        Args:
            network_file: 网络文件
            route_file: 路线流量文件（静态模式）或路线模板文件（动态模式）
            algorithm: 算法名称
            operation: 操作类型
            total_timesteps: 总训练步数
            total_simulation_seconds: 仿真总秒数
            gui_checkbox: 是否使用GUI
            prediction_steps: 预测步数
            traffic_mode: 流量模式（"静态流量" 或 "动态流量"）
            static_phase_ratio: 静态阶段时长占比（仅动态模式使用）
            base_flow_rate: 基础流率（仅动态模式使用）
            dynamic_phase_rate: 动态阶段生成速率（仅动态模式使用）
            
        Yields:
            Tuple[int, str]: (进度百分比, 输出信息)
        """
        try:
            # 设置训练状态
            self.is_training = True
            
            # 创建训练专用日志记录器
            self.current_training_logger = log_manager.create_training_logger(
                algorithm, operation
            )
            
            # 初始验证
            yield from self._validate_inputs(
                network_file, route_file, total_timesteps, total_simulation_seconds
            )
            
            # 解析参数
            yield 5, "正在解析训练参数..."
            config = self._parse_training_config(
                network_file, route_file, algorithm, operation,
                total_timesteps, total_simulation_seconds, gui_checkbox, prediction_steps,
                traffic_mode, static_phase_ratio, base_flow_rate, dynamic_phase_rate
            )

            yield 10, f"配置解析完成，开始{operation}操作..."
            
            # 执行训练仿真
            yield from self.run_simulation(config)
            
        except (ValidationError, ConfigurationError, TrainingError) as e:
            error_msg = f"训练失败: {e}"
            self.logger.error(error_msg)
            if self.current_training_logger:
                self.current_training_logger.error(error_msg)
            yield 0, error_msg
            
        except Exception as e:
            error_msg = f"训练过程中发生未知错误: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            if self.current_training_logger:
                self.current_training_logger.error(error_msg)
            yield 0, error_msg
            
        finally:
            self.is_training = False
            # 训练任务结束
    
    def _validate_inputs(self, network_file, route_file, 
                        total_timesteps, total_simulation_seconds) -> Iterator[Tuple[int, str]]:
        """验证输入参数"""
        yield 1, "正在验证输入参数..."
        
        if not network_file or not route_file:
            raise ValidationError("请上传路网模型和路线流量文件")
        
        if not isinstance(total_timesteps, int) or total_timesteps <= 0:
            raise ValidationError("训练步数必须是正整数")
            
        if not isinstance(total_simulation_seconds, (int, float)) or total_simulation_seconds <= 0:
            raise ValidationError("仿真总秒数必须是正数")
        
        # 验证文件路径
        try:
            utility_manager.validate_file_path(network_file.name, must_exist=True)
            utility_manager.validate_file_path(route_file.name, must_exist=True)
        except ValidationError as e:
            raise ValidationError(f"文件验证失败: {e}")
        
        yield 3, "输入参数验证完成"
    
    def _parse_training_config(self, network_file, route_file, algorithm, 
                              operation, total_timesteps, total_simulation_seconds, 
                              gui_checkbox, prediction_steps, traffic_mode,
                              static_phase_ratio, base_flow_rate, dynamic_phase_rate) -> BaseConfig:
        """解析训练配置"""
        try:
            import shlex
            network_path = shlex.quote(network_file.name)
            route_path = shlex.quote(route_file.name)
            use_gui = bool(gui_checkbox)
            
            # 根据流量模式决定配置参数
            if traffic_mode == "静态流量":
                # 静态流量模式：直接使用上传的流量文件
                config = parseParams(
                    net_file=network_path,
                    rou_file=route_path,  # 直接使用流量文件
                    algo_name=algorithm,
                    operation=operation,
                    tensorboard_logs="logs",
                    total_timesteps=total_timesteps,
                    num_seconds=total_simulation_seconds,
                    gui=use_gui,
                    prediction_steps=prediction_steps,
                    use_curriculum_learning=False  # 静态流量不使用课程学习
                )
            else:
                # 动态流量模式：使用课程学习，将路线文件作为模板
                rou_file_dir = os.path.dirname(route_file.name)
                rou_file_placeholder = shlex.quote(rou_file_dir)
                
                config = parseParams(
                    net_file=network_path,
                    rou_file=rou_file_placeholder,  # 实际的rou文件将动态生成
                    algo_name=algorithm,
                    operation=operation,
                    tensorboard_logs="logs",
                    total_timesteps=total_timesteps,
                    num_seconds=total_simulation_seconds,
                    gui=use_gui,
                    prediction_steps=prediction_steps,
                    use_curriculum_learning=True,  # 动态流量使用课程学习
                    curriculum_template_file=route_path,  # 将路线文件作为模板
                    curriculum_total_seconds=total_simulation_seconds,
                    curriculum_static_ratio=static_phase_ratio,
                    curriculum_base_flow=base_flow_rate,
                    curriculum_dynamic_rate=dynamic_phase_rate
                )
            
            return config
            
        except Exception as e:
            raise ConfigurationError(f"配置解析失败: {e}")
    
    def stop_training(self) -> str:
        """停止训练"""
        try:
            if self.is_training:
                self.is_training = False
                return "训练已停止"
            else:
                return "当前没有正在进行的训练任务"
        except Exception as e:
            error_msg = f"停止训练失败: {e}"
            self.logger.error(error_msg)
            return error_msg

    def run_simulation(self, config: BaseConfig) -> Iterator[Tuple[int, str]]:
        """运行仿真训练
        
        Args:
            config: 训练配置
            
        Yields:
            Tuple[int, str]: (进度百分比, 输出信息)
        """
        env = None
        model = None
        agent = None
        
        try:
            if config.algo_name.upper() == "FIXTIME":
                yield 15, "正在创建FIXTIME仿真环境..."
                yield from self._run_fixtime_simulation(config, 30, 90)
                return  
            
            yield 15, "正在创建仿真环境..."
            env = createEnv(config)
            if not env:
                raise EnvironmentError("环境创建失败")
            
            
            yield 20, "正在创建智能体模型..."
            agent = self.agent_factory.create_agent(env, config)
            model = agent.model
            
            # 检查是否加载已有模型
            model_path = Path(config.model_path)
            if model_path.exists():
                yield 25, "发现已有模型，正在加载..."
                model.load(model_path)
            
            # 根据操作类型执行相应任务
            if config.operation == "EVAL":
                yield from self._run_evaluation(model, config)
            elif config.operation == "TRAIN":
                yield from self._run_learning(model, config)
            elif config.operation == "PREDICT":
                yield from self._run_prediction(model, config)
            elif config.operation == "ALL":
                yield from self._run_all_operations(model, config)
            else:
                raise ValidationError(f"不支持的操作类型: {config.operation}")
            
            yield 100, f"{config.operation}操作完成！"
            
        except (EnvironmentError, TrainingError, ValidationError) as e:
            error_msg = f"仿真执行失败: {e}"
            self.logger.error(error_msg)
            if self.current_training_logger:
                self.current_training_logger.error(error_msg)
            yield 0, error_msg
            
        except Exception as e:
            error_msg = f"仿真过程中发生未知错误: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            if self.current_training_logger:
                self.current_training_logger.error(error_msg)
            yield 0, error_msg
            
        finally:
            # 清理资源
            try:
                if env:
                    env.close()
                if agent:
                    agent.cleanup()
                # 显式删除模型引用
                del model
                pass  # 仿真资源已清理
            except Exception as e:
                self.logger.warning(f"清理资源时出现警告: {e}")
    
    def _run_fixtime_simulation(self, config: BaseConfig, 
                                start_progress: int = 30, 
                                end_progress: int = 90) -> Iterator[Tuple[int, str]]:
        """运行固定配时仿真
        
        Args:
            model: FixTimeAgent对象（实际不使用，直接创建环境）
            config: 配置对象
            start_progress: 起始进度
            end_progress: 结束进度
        """
        
        yield start_progress, "初始化固定配时仿真器..."
        
        try:
            # 创建固定配时仿真器
            simulator = FixedTimingSimulator(config, self.current_training_logger)
            
            # 运行仿真
            results = None
            for progress, message in simulator.run_simulation():
                # 检查是否被用户停止
                if not self.is_training:
                    yield start_progress + 20, "固定配时仿真被用户停止"
                    return
                
                # 将仿真器的进度映射到总体进度范围
                mapped_progress = start_progress + int((progress / 100) * (end_progress - start_progress))
                yield mapped_progress, message
                
                # 保存最终结果
                if hasattr(simulator, '_last_results'):
                    results = simulator._last_results
            
            # 获取仿真结果
            results = simulator.get_last_results()
            if results:
                yield end_progress, f"固定配时仿真完成！平均等待时间: {results.get('avg_waiting_time', 0):.2f}s"
            else:
                yield end_progress, "固定配时仿真完成"
                
        except Exception as e:
            error_msg = f"❌ 固定配时仿真失败: {str(e)}"
            self.current_training_logger.error(error_msg)
            yield end_progress, error_msg
    
    def _run_evaluation(self, model, config: BaseConfig) -> Iterator[Tuple[int, str]]:
        """运行评估"""
        try:
            yield 30, "开始评估模型性能..."
            
            # 延迟导入evaluate_policy
            from stable_baselines3.common.evaluation import evaluate_policy
            
            mean_reward, std_reward = evaluate_policy(
                model, model.get_env(), 
                n_eval_episodes=config.n_eval_episodes
            )
            
            # 保存评估结果
            file_manager.write_eval_result_with_timestamp(
                mean_reward, std_reward, config.eval_path
            )
            
            result_msg = f"评估完成\n平均奖励: {mean_reward:.4f}\n标准差: {std_reward:.4f}"
            yield 90, result_msg
            
        except Exception as e:
            raise TrainingError(f"评估失败: {e}")
    
    def _run_learning(self, model, config: BaseConfig) -> Iterator[Tuple[int, str]]:
        """运行训练"""
        try:
            yield 30, "开始模型训练..."
            
            # 创建自定义回调
            from stable_baselines3.common.callbacks import BaseCallback

            class StopTrainingCallback(BaseCallback):
                def __init__(self, training_tab, verbose=0):
                    super(StopTrainingCallback, self).__init__(verbose)
                    self.training_tab = training_tab

                def _on_step(self) -> bool:
                    return self.training_tab.is_training

            # 执行训练
            stop_callback = StopTrainingCallback(self)
            model.learn(total_timesteps=config.total_timesteps, progress_bar=True, callback=stop_callback)
            
            yield 70, "训练完成，正在保存模型..."
            
            # 保存模型
            file_manager.ensure_dir(Path(config.model_path).parent)
            model.save(config.model_path)
            
            yield 80, "模型保存完成，开始训练后评估..."
            
            # 延迟导入evaluate_policy
            from stable_baselines3.common.evaluation import evaluate_policy
            
            # 训练后评估
            mean_reward, std_reward = evaluate_policy(
                model, model.get_env(), 
                n_eval_episodes=config.n_eval_episodes
            )
            
            file_manager.write_eval_result_with_timestamp(
                mean_reward, std_reward, config.eval_path
            )
            
            result_msg = f"训练完成\n训练后评估 - 平均奖励: {mean_reward:.4f}, 标准差: {std_reward:.4f}"
            yield 95, result_msg
            
        except Exception as e:
            raise TrainingError(f"训练失败: {e}")
    
    def _run_prediction(self, model, config: BaseConfig) -> Iterator[Tuple[int, str]]:
        """运行预测"""
        try:
            yield 30, "开始模型预测..."
            
            env = model.get_env()
            obs = env.reset()
            info_list = []
            state_list = []
            
            prediction_steps = config.prediction_steps
            for i in range(prediction_steps):
                if not self.is_training:  # 检查是否被停止
                    yield 50, "预测被用户停止"
                    return
                
                action, state = model.predict(obs)
                obs, reward, dones, info = env.step(action)
                
                info_list.append(info[0] if info else {})
                state_list.append(f"{obs}, {action}, {reward}\n")
                
                if config.gui:
                    env.render()
                
                progress = 30 + int((i + 1) / prediction_steps * 60)
                yield progress, f"预测进度: {i+1}/{prediction_steps}"
            
            # 保存预测结果
            file_manager.write_predict_results(
                info_list, config.predict_path, print_to_console=False
            )
            file_manager.write_loop_state(state_list, config.predict_path)
            
            yield 95, f"预测完成，结果已保存到: {config.predict_path}"
            
        except Exception as e:
            raise TrainingError(f"预测失败: {e}")
    
    def _run_all_operations(self, model, config: BaseConfig) -> Iterator[Tuple[int, str]]:
        """运行所有操作"""
        try:
            # 延迟导入evaluate_policy
            from stable_baselines3.common.evaluation import evaluate_policy
            
            # 训练前评估
            yield 15, "开始训练前评估..."
            mean_reward, std_reward = evaluate_policy(
                model, model.get_env(), 
                n_eval_episodes=config.n_eval_episodes
            )
            file_manager.write_eval_result_with_timestamp(
                mean_reward, std_reward, config.eval_path
            )
            
            yield 25, f"训练前评估完成 - 平均奖励: {mean_reward:.4f}"
            
            # 训练
            yield 30, "开始模型训练..."
            from stable_baselines3.common.callbacks import BaseCallback

            class StopTrainingCallback(BaseCallback):
                def __init__(self, training_tab, verbose=0):
                    super(StopTrainingCallback, self).__init__(verbose)
                    self.training_tab = training_tab

                def _on_step(self) -> bool:
                    return self.training_tab.is_training

            stop_callback = StopTrainingCallback(self)
            model.learn(total_timesteps=config.total_timesteps, progress_bar=True, callback=stop_callback)
            
            # 保存模型
            yield 60, "训练完成，正在保存模型..."
            model.save(config.model_path)
            
            # 训练后评估
            yield 70, "开始训练后评估..."
            # 延迟导入evaluate_policy
            from stable_baselines3.common.evaluation import evaluate_policy
            mean_reward, std_reward = evaluate_policy(
                model, model.get_env(), 
                n_eval_episodes=config.n_eval_episodes
            )
            file_manager.write_eval_result_with_timestamp(
                mean_reward, std_reward, config.eval_path
            )
            
            yield 80, f"训练后评估完成 - 平均奖励: {mean_reward:.4f}"
            
            # 预测
            yield 85, "开始模型预测..."
            env = model.get_env()
            obs = env.reset()
            info_list = []
            
            for i in range(config.prediction_steps):
                action, state = model.predict(obs)
                obs, reward, dones, info = env.step(action)
                info_list.append(info[0] if info else {})
                
                if config.gui:
                    env.render()
            
            # 保存预测结果
            file_manager.write_predict_results(
                info_list, config.predict_path, print_to_console=False
            )
            
            yield 95, "所有操作完成"
            
        except Exception as e:
            raise TrainingError(f"执行所有操作失败: {e}")
