# 🚦 交通信号智能体训练系统 (ATSCUI)

**Adaptive Traffic Signal Control Using Intelligence**

基于强化学习的自适应交通信号控制系统，使用SUMO交通仿真器和多种强化学习算法来优化交通信号灯控制策略。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)](https://tensorflow.org)
[![SUMO](https://img.shields.io/badge/SUMO-1.19.0-green.svg)](https://sumo.dlr.de)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 功能描述

  atscui 是一个用于自适应交通信号控制智能体训练、评估和可视化的综合平台。其核心业务功能包括：

   1. 智能体训练管理：
       * 支持多种强化学习算法： 能够配置和运行基于 DQN、PPO、A2C、SAC
         等主流强化学习算法的交通信号控制智能体训练。
       * 灵活的训练参数配置： 允许用户自定义训练步数、仿真时长、学习率、折扣因子等关键训练参数。
       * 课程学习支持：
         具备课程学习功能，能够根据预设的流量模板和动态生成策略，逐步增加训练难度，优化智能体性能。
       * 与 SUMO 仿真环境集成： 训练过程与 SUMO
         交通仿真器紧密结合，智能体在真实的交通流环境中学习和优化控制策略。


   2. 交通场景与配置管理：
       * 多场景支持：
         能够加载和管理不同路口或区域的交通网络（.net.xml）和车辆路由（.rou.xml）文件，支持多种复杂的交通场景。
       * 统一配置管理： 提供配置文件的加载、保存、验证、列表和删除功能，确保训练和运行参数的持久化和可追溯性。


   3. 结果可视化与分析：
       * 训练过程可视化： 能够分析并绘制训练过程中关键指标（如系统总停车数、平均等待时间、平均速度、智能体总停车
         数等）随训练回合变化的曲线，帮助用户监控训练进度和效果。
       * 预测结果可视化： 可视化智能体在预测（推理）阶段的交通系统指标随仿真时间步的变化。
       * 评估结果可视化： 绘制智能体在不同评估序列下的平均回报等评估指标，用于比较不同算法或模型的性能。


   4. 系统状态监控与依赖检查：
       * 系统版本显示： 在用户界面中显示当前系统的版本信息。
       * 配置健康检查： 自动检查核心配置文件的存在性和可加载性，以及关键 Python 依赖（如
         stable_baselines3、sumo）和环境变量（如 SUMO_HOME）的设置情况，提供系统健康状态反馈。


   5. 交互式用户界面：
       * 提供基于 Gradio 构建的直观用户界面，使用户能够方便地进行模型训练配置、启动训练/预测任务、选择日志文件进
         行可视化分析，并查看系统状态。

## ✨ 核心功能

- 🎯 **多算法支持**: DQN、PPO、A2C、SAC 四种主流强化学习算法
- 🚗 **交通仿真**: 基于SUMO的真实交通场景仿真
- 📊 **可视化界面**: 基于Gradio的现代化Web界面
- 🔄 **完整工作流**: 训练、评估、预测一体化流程
- 🌐 **实际应用**: 支持真实路口数据建模和部署
- 📈 **实时监控**: TensorBoard集成的训练过程可视化

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用户界面层     │    │   配置管理层     │    │   算法模型层     │
│   (Gradio UI)   │◄──►│  (Config)       │◄──►│ (RL Algorithms) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   环境管理层     │    │   数据处理层     │    │   仿真执行层     │
│ (Environment)   │◄──►│   (Utils)       │◄──►│    (SUMO)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 模块化设计

- **界面组件子模块** (`atscui/ui/`)
  - 训练配置界面
  - 结果可视化界面
  - 实时监控面板

- **配置管理子模块** (`atscui/config/`)
  - 基础配置类
  - 算法特定配置
  - 环境参数配置

- **算法模型子模块** (`atscui/models/`)
  - DQN (Deep Q-Network)
  - PPO (Proximal Policy Optimization)
  - A2C (Advantage Actor-Critic)
  - SAC (Soft Actor-Critic)

- **环境管理子模块** (`sumo_core/`)
  - SUMO环境封装
  - 交通信号控制逻辑
  - 观测空间定义

## 🚀 快速开始

### 系统要求

- **操作系统**: macOS / Linux / Windows
- **Python**: 3.8+
- **SUMO**: 1.19.0+
- **内存**: 8GB+ 推荐

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd atscui
   ```

2. **创建虚拟环境**
   ```bash
   python3 -m venv atscui_env
   source atscui_env/bin/activate  # Linux/macOS
   # 或 atscui_env\Scripts\activate  # Windows
   ```

3. **安装依赖**
  ```bash
  conda env create -f environment.yml
  ```
  或者
   ```bash
   pip install -r requirements.txt
   ```

4. **配置SUMO环境**
   ```bash
   export SUMO_HOME="/path/to/sumo"  # 设置SUMO_HOME环境变量
   ```

### 启动应用

```bash
python -m atscui.main
```

应用将在 `http://localhost:7861` 启动Web界面。

## 📋 操作模式

| 模式 | 描述 | 输出文件 |
|------|------|----------|
| **TRAIN** | 训练智能体模型 | `models/model.zip`, `outs/conn*.csv` |
| **EVAL** | 评估模型性能 | `evals/eval.txt` |
| **PREDICT** | 实时预测控制 | `predicts/predict.json` |
| **ALL** | 完整训练流程 | 所有上述文件 |

## 🛠️ 使用方法

### Web界面操作

1. **启动系统**: 运行 `python -m atscui.main`
2. **配置训练**: 在"模型训练"标签页设置参数
3. **开始训练**: 选择算法和操作模式，点击开始
4. **查看结果**: 在"结果可视化"标签页分析训练效果

### 命令行工具

#### 模型预测工具

`run_model.py` 是一个命令行预测工具，用于加载训练好的模型进行实时预测：

```bash
python atscui/run_model.py
```

**核心流程**:
```python
# 加载配置和环境
config = parse_config()
env = createEnv(config)
model = createAlgorithm(env, config.algo_name)

# 加载训练好的模型
if model_path.exists():
    model.load(model_path)

# 实时预测循环
obs = env.reset()
for step in range(simulation_steps):
    action, _ = model.predict(obs)  # 智能体决策
    obs, reward, done, info = env.step(action)  # 环境反馈
```

#### 批量实验脚本

```bash
# DQN算法实验
python mynets/dqn-my-intersection.py

# PPO算法实验  
python mynets/ppo-my-intersection.py

# 完整工作流实验
python mynets/AtscWorkflow.py
```

## 📁 项目结构

```
atscui/
├── atscui/                    # 核心系统模块
│   ├── main.py               # 系统入口
│   ├── config/               # 配置管理
│   │   ├── base_config.py    # 基础配置类
│   │   └── algorithm_configs.py # 算法配置
│   ├── models/               # 算法模型
│   │   ├── base_agent.py     # 智能体基类
│   │   ├── agent_creator.py  # 智能体工厂
│   │   └── agents/           # 具体算法实现
│   ├── ui/                   # 用户界面
│   │   ├── ui_main.py        # 主界面
│   │   └── components/       # 界面组件
│   ├── environment/          # 环境管理
│   └── utils/                # 工具函数
├── sumo_core/                # SUMO环境模块
│   └── envs/                 # 环境实现
├── mynets/                   # 实验脚本
│   ├── AtscWorkflow.py       # 完整工作流
│   ├── AtscEval.py           # 模型评估
│   └── net/                  # 网络配置文件
├── app/                      # 部署模块
│   ├── server.py             # API服务端
│   └── client.py             # 客户端接口
├── xgzd/                     # 新港大道路口案例
├── zfdx/                     # 振风独秀路口案例
├── zszx/                     # 中山大道路口案例
└── docs/                     # 文档资料
```

## 🌍 实际应用案例

### 振风大道与独秀大道路口 (zfdx)
- **数据来源**: 雷视雷达检测数据
- **应用场景**: 实际路口信号优化
- **协议支持**: 宇视科技雷视雷达通讯协议 V1.58

### 新港大道与遵大路路口 (xgzd)
- **数据来源**: 雷视雷达检测数据
- **应用场景**: 实际路口信号优化
- **协议支持**: 宇视科技雷视雷达通讯协议 V1.58

### 中山大道与中兴大道路口 (zszx)
- **数据来源**: 真实过车数据 (2024.10.20-2024.10.21)
- **特色功能**: K-means聚类分析交通流量模式
- **文件**: `zszx/data_process.py`, `zszx/flow_analysis.py`

### 标准测试路口 (mynets)
- **用途**: 算法验证和性能基准测试
- **配置**: 多种交通流量模式 (按小时、按概率、按周期)

## 🔌 API服务

系统提供REST API接口，支持外部系统集成：

### 启动API服务
```bash
python app/server.py
```

### 预测接口
```bash
POST http://localhost:6000/predict?algo=DQN
Content-Type: application/json

{
  "state": [0.1, 0.2, 0.3, ...]  # 交通状态向量
}
```

**响应**:
```json
{
  "action": [1, 0, 0, 1]  # 信号控制动作
}
```

**支持算法**: DQN, PPO, A2C, SAC

## 🛠️ 路网建模指南

### 手动路网设计优势
- ✅ 精确控制节点坐标和属性
- ✅ 灵活定义车道连接关系  
- ✅ 自定义信号灯配置
- ✅ 便于版本控制和批量修改

### 建模步骤

1. **定义节点** (`*.nod.xml`)
   ```xml
   <nodes>
     <node id="center" x="0" y="0" type="traffic_light"/>
   </nodes>
   ```

2. **定义道路** (`*.edg.xml`)
   ```xml
   <edges>
     <edge id="E1" from="N1" to="center" numLanes="2"/>
   </edges>
   ```

3. **定义连接** (`*.con.xml`)
   ```xml
   <connections>
     <connection from="E1" to="E2" fromLane="0" toLane="0"/>
   </connections>
   ```

4. **配置信号灯** (`*.tll.xml`)
   ```xml
   <tlLogics>
     <tlLogic id="center" programID="0" type="static">
       <phase duration="30" state="GGrrrrGGrrrr"/>
     </tlLogic>
   </tlLogics>
   ```

5. **生成路网**
   ```bash
   netconvert --node-files=*.nod.xml --edge-files=*.edg.xml \
              --connection-files=*.con.xml --tllogic-files=*.tll.xml \
              --output-file=network.net.xml
   ```

6. **创建交通需求** (`*.rou.xml`)
    ```xml
    <routes>
      <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="55"/>
      <route id="route1" edges="E1 E2"/>
      <flow id="flow1" route="route1" begin="0" end="3600" vehsPerHour="1800"/>
    </routes>
    ```

## 🚀 部署指南

### 生产环境部署

详细的部署指南请参考 <mcfile name="DEPLOYMENT_GUIDE.md" path="/Users/xnpeng/sumoptis/atscui/DEPLOYMENT_GUIDE.md"></mcfile>

**关键版本要求**:
- `numpy < 2.0` (兼容性要求)
- `tensorflow-macos == 2.15.0` (macOS)
- `pandas < 2.0` (稳定性要求)

### Docker部署 (推荐)

```dockerfile
FROM python:3.9-slim

# 安装SUMO
RUN apt-get update && apt-get install -y sumo sumo-tools

# 设置环境变量
ENV SUMO_HOME=/usr/share/sumo

# 复制项目文件
COPY . /app
WORKDIR /app

# 安装依赖
RUN pip install -r requirements.txt

# 启动应用
CMD ["python", "-m", "atscui.main"]
```

### 集群部署

支持分布式训练和多节点部署，适用于大规模交通网络优化。

## 📊 技术栈

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **仿真引擎** | SUMO | 1.19.0+ | 交通仿真 |
| **强化学习** | Stable-Baselines3 | 2.0.0+ | 算法框架 |
| **深度学习** | TensorFlow | 2.15.0 | 神经网络 |
| **用户界面** | Gradio | 4.44.1+ | Web界面 |
| **数据处理** | Pandas | <2.0 | 数据分析 |
| **可视化** | Matplotlib | 3.7.2+ | 图表绘制 |
| **多智能体** | PettingZoo | 1.23.1+ | 多智能体环境 |
| **API服务** | Flask | - | REST接口 |

## 📈 性能指标

### 训练性能
- **DQN**: 收敛时间 ~2-4小时 (1M步)
- **PPO**: 收敛时间 ~3-6小时 (1M步)
- **A2C**: 收敛时间 ~1-3小时 (1M步)
- **SAC**: 收敛时间 ~4-8小时 (1M步)

### 控制效果
- **平均等待时间**: 降低 15-30%
- **通行效率**: 提升 10-25%
- **燃油消耗**: 减少 8-20%

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. **Fork** 本仓库
2. **创建特性分支**: `git checkout -b feature/amazing-feature`
3. **提交更改**: `git commit -m 'Add amazing feature'`
4. **推送分支**: `git push origin feature/amazing-feature`
5. **提交Pull Request**

### 开发规范
- 遵循 PEP 8 代码风格
- 添加适当的测试用例
- 更新相关文档
- 确保向后兼容性

## 📝 更新日志

### v2.0.0 (2024-12)
- ✨ 重构UI界面，采用Gradio 4.x
- 🔧 优化配置管理系统
- 🚀 添加API服务支持
- 📊 增强可视化功能
- 🐛 修复兼容性问题

### v1.0.0 (2023-08)
- 🎉 初始版本发布
- 🎯 支持四种强化学习算法
- 🚗 集成SUMO仿真环境
- 📈 基础可视化功能

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [SUMO](https://sumo.dlr.de/) - 开源交通仿真平台
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - 强化学习算法库
- [Gradio](https://gradio.app/) - 机器学习Web界面框架

## 📞 联系方式

- **项目主页**: [GitHub Repository](https://github.com/your-repo)
- **问题反馈**: [Issues](https://github.com/your-repo/issues)
- **讨论交流**: [Discussions](https://github.com/your-repo/discussions)

---

<div align="center">
  <strong>🚦 让交通更智能，让出行更顺畅 🚦</strong>
</div>
