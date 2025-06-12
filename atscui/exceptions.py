"""统一异常处理模块

定义项目中使用的自定义异常类，提供更好的错误处理和调试信息。
"""


class ATSCUIException(Exception):
    """ATSCUI项目的基础异常类"""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(ATSCUIException):
    """配置相关异常"""
    def __init__(self, message: str, config_field: str = None):
        self.config_field = config_field
        error_code = "CONFIG_ERROR"
        if config_field:
            message = f"配置字段 '{config_field}': {message}"
        super().__init__(message, error_code)


class TrainingError(ATSCUIException):
    """训练相关异常"""
    def __init__(self, message: str, algorithm: str = None):
        self.algorithm = algorithm
        error_code = "TRAINING_ERROR"
        if algorithm:
            message = f"算法 '{algorithm}': {message}"
        super().__init__(message, error_code)


class EnvironmentError(ATSCUIException):
    """环境相关异常"""
    def __init__(self, message: str, env_type: str = None):
        self.env_type = env_type
        error_code = "ENV_ERROR"
        if env_type:
            message = f"环境 '{env_type}': {message}"
        super().__init__(message, error_code)


class ModelError(ATSCUIException):
    """模型相关异常"""
    def __init__(self, message: str, model_path: str = None):
        self.model_path = model_path
        error_code = "MODEL_ERROR"
        if model_path:
            message = f"模型文件 '{model_path}': {message}"
        super().__init__(message, error_code)


class FileOperationError(ATSCUIException):
    """文件操作相关异常"""
    def __init__(self, message: str, file_path: str = None, operation: str = None):
        self.file_path = file_path
        self.operation = operation
        error_code = "FILE_ERROR"
        if file_path and operation:
            message = f"文件操作 '{operation}' 在 '{file_path}': {message}"
        elif file_path:
            message = f"文件 '{file_path}': {message}"
        super().__init__(message, error_code)


class ValidationError(ATSCUIException):
    """数据验证相关异常"""
    def __init__(self, message: str, field_name: str = None, value=None):
        self.field_name = field_name
        self.value = value
        error_code = "VALIDATION_ERROR"
        if field_name:
            message = f"验证失败 '{field_name}' (值: {value}): {message}"
        super().__init__(message, error_code)


class UIError(ATSCUIException):
    """UI相关异常"""
    def __init__(self, message: str, component: str = None):
        self.component = component
        error_code = "UI_ERROR"
        if component:
            message = f"UI组件 '{component}': {message}"
        super().__init__(message, error_code)


class VisualizationError(ATSCUIException):
    """可视化相关异常"""
    def __init__(self, message: str, plot_type: str = None):
        self.plot_type = plot_type
        error_code = "VIZ_ERROR"
        if plot_type:
            message = f"可视化类型 '{plot_type}': {message}"
        super().__init__(message, error_code)