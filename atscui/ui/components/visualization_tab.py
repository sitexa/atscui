import ntpath
import os
import glob
import re
import ntpath
import traceback
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr

from atscui.utils.utils import utility_manager
from atscui.utils.file_utils import file_manager, extract_crossname_from_evalfile
from atscui.utils.visualization import Visualizer
from atscui.exceptions import ValidationError, FileOperationError, VisualizationError
from atscui.logging_manager import get_logger



class VisualizationTab:
    """可视化标签页
    
    提供训练过程、预测结果和评估结果的可视化功能。
    """
    
    def __init__(self):
        self.visualizer = Visualizer()
        self.logger = get_logger('visualization_tab')
        
        # UI组件
        self.train_process_file = None
        self.predict_result_file = None
        self.eval_result_file = None
        self.plot_output = None
        self.plot_image = None

    def render(self) -> gr.Column:
        """渲染可视化标签页
        
        Returns:
            gr.Column: 可视化标签页的UI组件
        """
        try:
            # 训练过程可视化
            with gr.Row():
                with gr.Column(scale=2):
                    self.train_process_file = gr.File(
                        label="选择训练过程文件 (支持CSV格式)", 
                        file_types=[".csv"]
                    )
                    plot_train_button = gr.Button(
                        "绘制训练过程图", 
                        variant="secondary",
                        size="sm"
                    )

            # 预测结果可视化
            with gr.Row():
                with gr.Column(scale=2):
                    self.predict_result_file = gr.File(
                        label="选择预测结果文件 (支持JSON格式)", 
                        file_types=[".json"]
                    )
                    plot_predict_button = gr.Button(
                        "绘制预测结果图", 
                        variant="secondary",
                        size="sm"
                    )

            # 评估结果可视化
            with gr.Row():
                with gr.Column(scale=2):
                    self.eval_result_file = gr.File(
                        label="选择评估文件 (支持TXT格式)", 
                        file_types=[".txt"]
                    )
                    plot_eval_button = gr.Button(
                        "绘制评估结果图", 
                        variant="secondary",
                        size="sm"
                    )

            # 输出组件
            self.plot_output = gr.Textbox(
                label="绘图输出", 
                lines=3,
                max_lines=10,
                interactive=False,
                show_copy_button=True
            )
            self.plot_image = gr.Image(
                label="生成的图形",
                show_download_button=True,
                show_share_button=False
            )

            # 绑定事件
            plot_train_button.click(
                self._plot_training_process,
                inputs=[self.train_process_file],
                outputs=[self.plot_image, self.plot_output]
            )

            plot_predict_button.click(
                self._plot_prediction_result,
                inputs=[self.predict_result_file],
                outputs=[self.plot_image, self.plot_output]
            )

            plot_eval_button.click(
                self._plot_evaluation_results,
                inputs=[self.eval_result_file],
                outputs=[self.plot_image, self.plot_output]
            )
            
            self.logger.info("可视化标签页渲染完成")
            
        except Exception as e:
            error_msg = f"渲染可视化标签页失败: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise VisualizationError(error_msg)

    def _plot_training_process(self, file) -> Tuple[Optional[str], str]:
        """绘制训练过程图

        Args:
            file: 上传的训练过程文件

        Returns:
            Tuple[Optional[str], str]: (图片路径, 输出信息)
        """
        try:
            # 1. 基本验证
            if file is None:
                return None, "❌ 请选择训练过程文件"

            base_filename = os.path.basename(file.name)
            if not base_filename.lower().endswith('.csv'):
                return None, "❌ 请选择CSV格式的训练过程文件"

            # 2. 在整个项目中搜索该文件以找到其原始位置
            project_root = str(Path(__file__).resolve().parents[3])
            search_pattern = os.path.join(project_root, '**', base_filename)
            found_files = glob.glob(search_pattern, recursive=True)

            # 3. 处理搜索结果
            if not found_files:
                return None, f"❌ 在项目目录中找不到文件: {base_filename}。请确保文件在项目文件夹内。"

            if len(found_files) > 1:
                self.logger.warning(f"找到多个同名文件: {found_files}。将使用第一个: {found_files[0]}")

            original_file_path = found_files[0]
            original_log_dir = os.path.dirname(original_file_path)

            # 4. 调用可视化工具
            output_path = self.visualizer.plot_all_training_logs(
                specific_log_file_path=original_file_path,
                original_log_directory=original_log_dir
            )


            # 5. 处理结果
            if output_path and file_manager.file_exists(output_path):
                success_msg = f"✅ 训练过程所有指标图已生成: {output_path}"
                self.logger.info(f"训练过程所有指标图生成成功: {output_path}")
                return output_path, success_msg
            else:
                return None, "❌ 训练过程所有指标图生成失败，请检查文件格式和内容"

        except ValidationError as e:
            error_msg = f"❌ 输入验证失败: {e}"
            self.logger.warning(error_msg)
            return None, error_msg

        except FileOperationError as e:
            error_msg = f"❌ 文件操作失败: {e}"
            self.logger.error(error_msg)
            return None, error_msg

        except Exception as e:
            error_msg = f"❌ 绘制训练过程图时发生错误: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None, error_msg

    def _plot_prediction_result(self, file) -> Tuple[Optional[str], str]:
        """绘制预测结果图
        
        Args:
            file: 上传的预测结果文件
            
        Returns:
            Tuple[Optional[str], str]: (图片路径, 输出信息)
        """
        try:
            # 验证文件
            if file is None:
                return None, "❌ 请选择预测结果文件"
            
            if not file_manager.file_exists(file.name):
                return None, "❌ 文件不存在或无法访问"
            
            # 验证文件格式
            if not file.name.lower().endswith('.json'):
                return None, "❌ 请选择JSON格式的预测结果文件"
            
            self.logger.info(f"开始绘制预测结果图: {file.name}")
            
            # 获取文件信息
            folder_name, filename = self._get_gradio_file_info(file)
            
            # 生成图表
            output_path = self.visualizer.plot_predict(file.name, folder_name, filename)
            
            if output_path and file_manager.file_exists(output_path):
                success_msg = f"✅ 预测结果图已生成: {output_path}"
                self.logger.info(f"预测结果图生成成功: {output_path}")
                return output_path, success_msg
            else:
                return None, "❌ 图表生成失败，请检查文件格式和内容"
                
        except ValidationError as e:
            error_msg = f"❌ 输入验证失败: {e}"
            self.logger.warning(error_msg)
            return None, error_msg
            
        except FileOperationError as e:
            error_msg = f"❌ 文件操作失败: {e}"
            self.logger.error(error_msg)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"❌ 绘制预测结果图时发生错误: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None, error_msg

    def _plot_evaluation_results(self, file) -> Tuple[Optional[str], str]:
        """绘制评估结果图
        
        Args:
            file: 上传的评估结果文件
            
        Returns:
            Tuple[Optional[str], str]: (图片路径, 输出信息)
        """
        try:
            # 验证文件
            if file is None:
                return None, "❌ 请选择评估文件"
            
            if not file_manager.file_exists(file.name):
                return None, "❌ 文件不存在或无法访问"
            
            # 验证文件格式
            if not file.name.lower().endswith('.txt'):
                return None, "❌ 请选择TXT格式的评估结果文件"
            
            self.logger.info(f"开始绘制评估结果图: {file.name}")
            
            # 获取文件信息
            folder_name, filename = self._get_gradio_file_info(file)
            eval_filename = ntpath.basename(filename)
            
            # 提取路口名称
            cross_name = extract_crossname_from_evalfile(eval_filename)
            if not cross_name:
                return None, "❌ 无法从文件名中提取路口名称，请检查文件命名格式"
            
            # 生成图表
            output_path = self.visualizer.plot_evaluation(folder_name, cross_name)
            
            if output_path and file_manager.file_exists(output_path):
                success_msg = f"✅ 评估结果图已生成: {output_path}"
                self.logger.info(f"评估结果图生成成功: {output_path}")
                return output_path, success_msg
            else:
                return None, "❌ 图表生成失败，请检查文件格式和内容"
                
        except ValidationError as e:
            error_msg = f"❌ 输入验证失败: {e}"
            self.logger.warning(error_msg)
            return None, error_msg
            
        except FileOperationError as e:
            error_msg = f"❌ 文件操作失败: {e}"
            self.logger.error(error_msg)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"❌ 绘制评估结果图时发生错误: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None, error_msg

    def _get_gradio_file_info(self, file: gr.File) -> Tuple[Optional[str], Optional[str]]:
        """获取Gradio文件信息
        
        Args:
            file: Gradio文件对象
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (推断的文件夹路径, 文件名)
        """
        try:
            if file is None:
                return None, None

            # 获取原始文件名
            filename = os.path.basename(file.name)
            
            # 验证文件名
            if not filename:
                raise ValidationError("无效的文件名")

            conn_ep = r'_conn(\d+)_ep(\d+)'

            # 推断预期的文件夹
            if 'eval' in filename.lower():
                inferred_folder = './evals'
            elif 'predict' in filename.lower():
                inferred_folder = './predicts'
            elif re.search(conn_ep, filename):
                inferred_folder = './outs'
            else:
                inferred_folder = './'  # 默认为当前目录
            
            # 确保文件夹路径是标准化的
            inferred_folder = utility_manager.normalize_path(inferred_folder)
            
            self.logger.debug(f"文件信息 - 文件名: {filename}, 推断文件夹: {inferred_folder}")
            
            return inferred_folder, filename
            
        except Exception as e:
            self.logger.error(f"获取文件信息失败: {e}")
            return None, None
