# 交通信号智能体训练系统 - 部署指南

## 问题解决记录

### 主要解决的兼容性问题

1. **NumPy 版本兼容性问题**
   - 问题：NumPy 2.0.2 与 stable_baselines3 不兼容
   - 解决方案：降级到 NumPy 1.26.4

2. **TensorFlow 兼容性问题**
   - 问题：TensorFlow 2.19.0 存在符号链接错误
   - 解决方案：降级到 TensorFlow 2.15.0 并移除 tensorflow-metal

3. **Pandas 版本兼容性问题**
   - 问题：Pandas 2.3.0 与 NumPy 1.x 版本不兼容
   - 解决方案：降级到 Pandas 1.5.3

## 推荐的组件版本清单

### 核心依赖版本

```
# 机器学习框架
numpy==1.26.4
pandas==1.5.3
tensorflow-macos==2.15.0
tensorboard==2.15.0

# 强化学习库
stable-baselines3==2.0.0a5
gym==0.26.2

# 可视化和UI
matplotlib==3.7.2
gradio==4.44.1

# 交通仿真
sumo==1.19.0
traci==1.19.0

# 其他工具库
joblib==1.3.2
scipy==1.11.3
```

### 系统要求

- **操作系统**: macOS (已测试)
- **Python版本**: Python 3.8+
- **SUMO**: 需要单独安装 SUMO 交通仿真软件

### 安装步骤

1. **创建虚拟环境**
   ```bash
   python3 -m venv atscui_env
   source atscui_env/bin/activate
   ```

2. **安装核心依赖**
   ```bash
   pip3 install "numpy<2"
   pip3 install "pandas<2"
   pip3 install tensorflow-macos==2.15.0
   pip3 install tensorboard==2.15.0
   ```

3. **安装强化学习库**
   ```bash
   pip3 install stable-baselines3
   pip3 install gym
   ```

4. **安装其他依赖**
   ```bash
   pip3 install -r requirements.txt
   ```

### 注意事项

1. **避免安装的组件**:
   - `tensorflow-metal`: 在当前配置下会导致符号链接错误
   - `numpy>=2.0`: 与 stable_baselines3 不兼容
   - `pandas>=2.0`: 与 numpy 1.x 版本存在兼容性问题

2. **已知警告**:
   - 系统健康检查警告: `'ConfigManager' object has no attribute 'is_healthy'`
   - 此警告不影响核心功能的正常使用

3. **版本冲突处理**:
   - 如果遇到版本冲突，优先保证 numpy < 2.0
   - TensorFlow 使用 2.15.0 版本以确保稳定性
   - 避免自动升级到最新版本

### 启动应用

```bash
cd /Users/xnpeng/sumoptis/atscui
python -m atscui.main
```

应用将在 `http://127.0.0.1:7860` 启动。

### 功能验证

启动后应该能看到：
- ✅ 训练标签页正常加载
- ✅ 可视化标签页正常加载
- ✅ TensorFlow 正常导入
- ✅ stable_baselines3 正常导入
- ⚠️ 系统健康检查警告（不影响使用）

---

**最后更新**: 2024年12月
**测试环境**: macOS, Python 3.x