import os
import glob
import traceback
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr

from atscui.exceptions import ValidationError, FileOperationError, VisualizationError
from atscui.logging_manager import get_logger


class MultiAlgorithmTab:
    """多算法对比分析标签页
    
    提供多种算法的性能对比分析功能。
    """
    
    def __init__(self):
        self.logger = get_logger('multi_algorithm_tab')
        
        # UI组件
        self.analysis_dir = None
        self.analysis_output = None
        self.analysis_image = None
        self.analysis_report = None

    def render(self) -> gr.Column:
        """渲染多算法对比分析标签页
        
        Returns:
            gr.Column: 多算法对比分析标签页的UI组件
        """
        try:
            # 输入区域
            with gr.Row():
                with gr.Column(scale=3):
                    self.analysis_dir = gr.Textbox(
                        label="📁 分析目录路径",
                        placeholder="请输入包含多种算法结果文件的目录路径",
                        value="outs/train",
                        lines=1,
                        info="目录中应包含不同算法的CSV结果文件，默认为训练输出目录"
                    )
                    
                with gr.Column(scale=1):
                    with gr.Row():
                        quick_path_button = gr.Button(
                            "📂 默认目录",
                            variant="secondary",
                            size="sm"
                        )
                        run_analysis_button = gr.Button(
                            "🚀 开始分析", 
                            variant="primary",
                            size="sm",
                            elem_classes=["analysis-button"]
                        )
            
            # 输出区域
            with gr.Row():
                with gr.Column():
                    self.analysis_output = gr.Textbox(
                        label="📋 分析输出", 
                        lines=8,
                        max_lines=15,
                        interactive=False,
                        show_copy_button=True,
                        elem_classes=["analysis-output"]
                    )
            
            # 结果展示区域
            with gr.Row():
                with gr.Column(scale=1):
                    self.analysis_image = gr.Image(
                        label="📊 对比分析图表",
                        show_download_button=True,
                        show_share_button=False,
                        elem_classes=["analysis-chart"]
                    )
                
                with gr.Column(scale=1):
                    self.analysis_report = gr.File(
                        label="📄 详细分析报告",
                        file_types=[".md"],
                        elem_classes=["analysis-report"]
                    )
            
            # 绑定事件
            run_analysis_button.click(
                self._run_multi_algorithm_analysis,
                inputs=[self.analysis_dir],
                outputs=[self.analysis_image, self.analysis_output, self.analysis_report]
            )
            
            quick_path_button.click(
                self._set_default_path,
                outputs=[self.analysis_dir]
            )
            
            self.logger.info("多算法对比分析标签页渲染完成")
            
        except Exception as e:
            error_msg = f"渲染多算法对比分析标签页失败: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise VisualizationError(error_msg)

    def _set_default_path(self) -> str:
        """设置默认路径
        
        Returns:
            str: 默认的训练输出目录路径
        """
        try:
            project_root = str(Path(__file__).resolve().parents[3])
            default_path = os.path.join(project_root, "outs", "train")
            return default_path
        except Exception as e:
            self.logger.warning(f"设置默认路径失败: {e}")
            return ""

    def _run_multi_algorithm_analysis(self, directory_path: str) -> Tuple[Optional[str], str, Optional[str]]:
        """运行多算法对比分析
        
        Args:
            directory_path: 包含多种算法结果文件的目录路径
            
        Returns:
            Tuple[Optional[str], str, Optional[str]]: (图片路径, 输出信息, 报告文件路径)
        """
        try:
            # 验证输入
            if not directory_path or not directory_path.strip():
                return None, "❌ 请输入目录路径", None
            
            directory_path = directory_path.strip()
            
            # 验证目录是否存在
            if not os.path.exists(directory_path):
                return None, f"❌ 目录不存在: {directory_path}", None
            
            if not os.path.isdir(directory_path):
                return None, f"❌ 路径不是目录: {directory_path}", None
            
            self.logger.info(f"开始执行多算法对比分析: {directory_path}")
            
            # 获取comparative_analysis.py脚本路径
            project_root = str(Path(__file__).resolve().parents[3])
            script_path = os.path.join(project_root, "atscui", "utils", "comparative_analysis.py")
            
            if not os.path.exists(script_path):
                return None, f"❌ 找不到多算法对比分析脚本: {script_path}", None
            
            # 运行comparative_analysis.py脚本
            try:
                result = subprocess.run(
                    [sys.executable, script_path, directory_path],
                    capture_output=True,
                    text=True,
                    cwd=project_root,
                    timeout=300  # 5分钟超时
                )
                
                if result.returncode == 0:
                    # 脚本执行成功，查找生成的文件
                    output_image_patterns = [
                        os.path.join(directory_path, "*_multi_algorithm_analysis.png"),
                        os.path.join(directory_path, "comparative_analysis.png")
                    ]
                    
                    output_report_patterns = [
                        os.path.join(directory_path, "*_multi_algorithm_report.md"),
                        os.path.join(directory_path, "comparative_analysis_report.md")
                    ]
                    
                    generated_image = None
                    generated_report = None
                    
                    # 查找生成的图片
                    for pattern in output_image_patterns:
                        matching_files = glob.glob(pattern)
                        if matching_files:
                            generated_image = matching_files[0]
                            break
                    
                    # 查找生成的报告
                    for pattern in output_report_patterns:
                        matching_files = glob.glob(pattern)
                        if matching_files:
                            generated_report = matching_files[0]
                            break
                    
                    # 构建成功消息
                    success_parts = ["✅ 多算法对比分析完成！"]
                    
                    if generated_image:
                        success_parts.append(f"📊 生成的图表: {os.path.basename(generated_image)}")
                    
                    if generated_report:
                        success_parts.append(f"📄 生成的报告: {os.path.basename(generated_report)}")
                    
                    success_parts.append("\n📋 分析输出:")
                    success_parts.append(result.stdout)
                    
                    success_msg = "\n\n".join(success_parts)
                    
                    self.logger.info(f"多算法对比分析成功: 图表={generated_image}, 报告={generated_report}")
                    return generated_image, success_msg, generated_report
                    
                else:
                    error_msg = f"❌ 多算法对比分析失败\n\n错误信息:\n{result.stderr}\n\n输出信息:\n{result.stdout}"
                    self.logger.error(f"多算法对比分析失败: {result.stderr}")
                    return None, error_msg, None
                    
            except subprocess.TimeoutExpired:
                return None, "❌ 分析超时（超过5分钟），请检查数据量或算法复杂度", None
            except subprocess.SubprocessError as e:
                return None, f"❌ 执行分析脚本时发生错误: {e}", None
                
        except Exception as e:
            error_msg = f"❌ 执行多算法对比分析时发生错误: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return None, error_msg, None