import gradio as gr
import traceback
from typing import Optional

from atscui.ui.components.training_tab import TrainingTab
from atscui.ui.components.visualization_tab import VisualizationTab
from atscui.exceptions import UIError, ConfigurationError
from atscui.logging_manager import get_logger
from atscui.config_manager import config_manager


class ATSCUI:
    """交通信号智能体训练系统主界面
    
    负责创建和管理整个用户界面，包括模型训练和结果可视化功能。
    """
    
    def __init__(self):
        self.logger = get_logger('atscui_main')
        self.training_tab: Optional[TrainingTab] = None
        self.visualization_tab: Optional[VisualizationTab] = None
        
        # 加载UI配置
        try:
            self.ui_config = config_manager.get_ui_config()
            # 确保默认配置在UI启动时被创建
            config_manager.create_default_configs()
            self.logger.info("ATSCUI主界面初始化完成")
        except Exception as e:
            self.logger.error(f"初始化ATSCUI失败: {e}")
            raise ConfigurationError(f"UI配置加载失败: {e}")
    
    def create_ui(self) -> gr.Blocks:
        """创建用户界面
        
        Returns:
            gr.Blocks: Gradio界面对象
            
        Raises:
            UIError: 界面创建失败时抛出
        """
        try:
            self.logger.info("开始创建用户界面")
            
            # 创建主界面
            with gr.Blocks(
                theme=gr.themes.Soft(),
                title="交通信号智能体训练系统",
                css=self._get_custom_css(),
                analytics_enabled=False
            ) as demo:
                # 主标题
                gr.Markdown(
                    "# 🚦 交通信号智能体训练系统\n"
                    "*Adaptive Traffic Signal Control using Intelligent Agents*",
                    elem_classes=["main-title"]
                )
                
                # 主要功能标签页
                with gr.Tabs() as tabs:
                    # 模型训练标签页
                    with gr.TabItem("🎯 模型训练", id="training"):
                        try:
                            self.training_tab = TrainingTab()
                            self.training_tab.render()
                            self.logger.info("训练标签页创建成功")
                        except Exception as e:
                            self.logger.error(f"创建训练标签页失败: {e}")
                            gr.Markdown(
                                f"❌ **训练功能暂时不可用**\n\n错误信息: {e}",
                                elem_classes=["error-message"]
                            )
                    
                    # 结果可视化标签页
                    with gr.TabItem("📈 结果可视化", id="visualization"):
                        try:
                            self.visualization_tab = VisualizationTab()
                            self.visualization_tab.render()
                            self.logger.info("可视化标签页创建成功")
                        except Exception as e:
                            self.logger.error(f"创建可视化标签页失败: {e}")
                            gr.Markdown(
                                f"❌ **可视化功能暂时不可用**\n\n错误信息: {e}",
                                elem_classes=["error-message"]
                            )
                
                # 页脚信息
                with gr.Row():
                    gr.Markdown(
                        "---\n"
                        "💡 **使用提示**: 请先上传配置文件，选择算法和操作类型，然后开始训练或预测。\n"
                        "📚 **帮助文档**: [用户手册](./docs) | [API文档](./api-docs) | [问题反馈](./issues)",
                        elem_classes=["footer-info"]
                    )
            
            self.logger.info("用户界面创建完成")
            return demo
            
        except Exception as e:
            error_msg = f"创建用户界面失败: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            raise UIError(error_msg)
    
    def _get_custom_css(self) -> str:
        """获取自定义CSS样式"""
        return """
        .main-title {
            text-align: center;
            color: #2E86AB;
            margin-bottom: 20px;
        }
        .status-info {
            text-align: center;
            font-size: 14px;
            color: #666;
            margin-bottom: 15px;
        }
        .error-message {
            background-color: #fee;
            border: 1px solid #fcc;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }
        .footer-info {
            text-align: center;
            font-size: 12px;
            color: #888;
            margin-top: 20px;
        }
        """
    
    def _get_system_version(self) -> str:
        """获取系统版本信息"""
        try:
            return config_manager.get_system_version()
        except Exception:
            return "未知版本"
    
    def _check_system_health(self) -> bool:
        """检查系统健康状态"""
        try:
            # 检查关键组件是否正常
            health_checks = [
                config_manager.is_healthy(),
                self._check_dependencies(),
                self._check_environment()
            ]
            return all(health_checks)
        except Exception as e:
            self.logger.warning(f"系统健康检查失败: {e}")
            return False
    
    def _check_dependencies(self) -> bool:
        """检查依赖项"""
        try:
            # 延迟导入stable_baselines3以避免TensorFlow初始化错误
            import stable_baselines3
            import sumo
            return True
        except Exception as e:
            self.logger.warning(f"依赖项检查失败: {e}")
            return False
    
    def _check_environment(self) -> bool:
        """检查环境变量"""
        import os
        return "SUMO_HOME" in os.environ
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.training_tab:
                # 如果训练标签页有清理方法，调用它
                if hasattr(self.training_tab, 'cleanup'):
                    self.training_tab.cleanup()
            
            if self.visualization_tab:
                # 如果可视化标签页有清理方法，调用它
                if hasattr(self.visualization_tab, 'cleanup'):
                    self.visualization_tab.cleanup()
            
            self.logger.info("ATSCUI资源清理完成")
            
        except Exception as e:
            self.logger.warning(f"清理资源时出现警告: {e}")
