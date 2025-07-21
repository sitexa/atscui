# ZFDX路口固定周期交通信号控制评估

本目录包含了用于固定周期交通信号控制仿真评估的工具集。

## 📁 文件结构

```
zfdx/
├── net/                              # SUMO仿真文件
│   ├── zfdx.net.xml                   # 路网文件
│   ├── zfdx.tll.xml                   # 交通信号配置（固定周期）
│   ├── zfdx-perhour.rou.xml           # 按小时交通流量路由文件
│   └── zfdx.sumocfg                   # SUMO配置文件
├── fixed_timing_evaluation.py         # 固定周期仿真评估
├── evaluation_results/                # 评估结果存储目录
└── README_evaluation.md               # 本说明文档
```

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：

```bash
# 安装Python依赖
pip install stable-baselines3 pandas matplotlib numpy

# 确保SUMO已正确安装并设置SUMO_HOME环境变量
echo $SUMO_HOME
```

### 2. 运行固定周期仿真

```bash
cd zfdx
python fixed_timing_evaluation.py
```

这将：
- 运行5轮固定周期信号控制仿真
- 每轮仿真1小时（3600秒）
- 收集交通性能指标
- 保存结果到 `evaluation_results/` 目录
- 如果发现智能体结果文件，会自动进行对比分析

### 3. 查看仿真结果

脚本会自动生成：
- 📊 性能统计图表
- 📈 详细统计报告
- 📋 CSV格式的原始数据

## 📊 评估指标

### 主要性能指标

| 指标 | 描述 | 单位 | 目标 |
|------|------|------|------|
| 平均等待时间 | 车辆因红灯或拥堵的等待时间 | 秒 | 越小越好 |
| 平均排队长度 | 各车道的平均排队车辆数 | 车辆 | 越小越好 |
| 平均速度 | 路网中车辆的平均行驶速度 | m/s | 越大越好 |
| 系统吞吐量 | 单位时间通过路口的车辆数 | 车辆/小时 | 越大越好 |
| 平均行程时间 | 车辆通过路网的平均耗时 | 秒 | 越小越好 |

### 环境影响指标

- 燃油消耗量
- CO2排放量

## 🔧 配置说明

### 固定周期信号配置

当前zfdx路口的固定周期配置（`zfdx.tll.xml`）：

```xml
<tlLogic id="tl_1" programID="0" offset="0" type="static">
    <phase duration="50" state="GGGGrrrrrrGGGrrrrrr"/>  <!-- 东西直行 50秒 -->
    <phase duration="3"  state="yyyyrrrrrryyyrrrrrr"/>  <!-- 黄灯 3秒 -->
    <phase duration="20" state="rrrrGrrrrrrrrGGrrrr"/>  <!-- 东西左转 20秒 -->
    <phase duration="3"  state="rrrryrrrrrrrryyrrrr"/>  <!-- 黄灯 3秒 -->
    <phase duration="50" state="rrrrrGGGrrrrrrrGGGr"/>  <!-- 南北直行 50秒 -->
    <phase duration="3"  state="rrrrryyyrrrrrrryyyr"/>  <!-- 黄灯 3秒 -->
    <phase duration="20" state="rrrrrrrrGGrrrrrrrrG"/>  <!-- 南北左转 20秒 -->
    <phase duration="3"  state="rrrrrrrryyrrrrrrrry"/>  <!-- 黄灯 3秒 -->
</tlLogic>
```

**总周期时长**: 152秒

### 仿真参数配置

- **仿真步长**: 5秒
- **最小绿灯时间**: 10秒
- **最大绿灯时间**: 60秒
- **黄灯时间**: 3秒
- **仿真轮数**: 5轮
- **每轮时长**: 3600秒

## 📈 结果分析

### 典型对比结果示例

```
固定周期交通信号控制 - 性能评估报告
============================================================

📊 关键性能指标:
指标                 平均值          标准差
----------------------------------------------------------------------
平均等待时间(s)      45.8           3.2
平均排队长度         15.2           2.1
平均速度(m/s)        8.5            0.8
总吞吐量             1850           85
平均行程时间(s)      88.5           5.3

🎯 总结:
✅ 固定周期控制提供稳定的基线性能
✅ 可作为智能体控制算法的对比基准
✅ 为交通信号优化提供参考数据
```

### 图表输出

脚本会自动生成包含以下内容的统计图表：

1. **箱线图**: 显示各指标的分布情况
2. **时间序列图**: 显示指标随时间的变化趋势
3. **对比柱状图**: 如果有智能体结果，显示性能对比

## 🛠️ 高级用法

### 自定义仿真参数

可以修改脚本中的参数来适应不同的测试需求：

```python
# 在 fixed_timing_evaluation.py 中
fixed_results = evaluator.run_fixed_timing_simulation(
    num_episodes=10,      # 仿真轮数
    episode_length=7200,  # 每轮时长（秒）
    delta_time=5          # 仿真步长（秒）
)

# 在 agent_control_evaluation.py 中
agent_results = evaluator.run_agent_simulation(
    model_path=model_path,
    algorithm="DQN",
    num_episodes=10,
    episode_length=7200,
    delta_time=5
)
```

### 使用不同的交通流量模式

可以替换路由文件来测试不同的交通流量场景：

```python
# 修改路由文件路径
route_file = "net/zfdx-probability.rou.xml"

# 或创建自定义路由文件
```

### 批量测试不同配置

```python
# 修改 fixed_timing_evaluation.py 来测试不同配置
configurations = [
    {'cycle_time': 120, 'green_ratio': 0.6},
    {'cycle_time': 150, 'green_ratio': 0.7}
]
for config in configurations:
    # 运行评估...
```

## 📋 故障排除

### 常见问题

1. **SUMO_HOME未设置**
   ```bash
   export SUMO_HOME=/path/to/sumo
   ```

2. **智能体结果文件格式不匹配**
   - 检查智能体结果CSV文件的列名
   - 确保包含基本指标列

3. **仿真运行缓慢**
   - 减少 `num_episodes` 或 `episode_length`
   - 确保 `use_gui=False`

4. **内存不足**
   - 减少仿真时长
   - 增加 `delta_time` 步长

### 调试模式

启用详细日志输出：

```python
# 在环境创建时添加
env = SumoEnv(
    # ... 其他参数
    sumo_warnings=True,  # 启用SUMO警告
    use_gui=True         # 启用GUI观察
)
```

## 🔬 扩展功能

### 添加新的评估指标

在 `_extract_metrics` 方法中添加：

```python
def _extract_metrics(self, info, episode, step, control_type):
    metrics = {
        # ... 现有指标
        'custom_metric': self._calculate_custom_metric(info)
    }
    return metrics
```

### 添加新的评估指标

在 `_extract_final_metrics` 方法中添加：

```python
def _extract_final_metrics(self, info):
    metrics = {
        # ... 现有指标
        'custom_metric': self._calculate_custom_metric(info)
    }
    return metrics
```

### 多路口扩展

修改网络文件和环境配置以支持多路口仿真。

## 📚 参考资料

- [SUMO官方文档](https://sumo.dlr.de/docs/)
- [Stable-Baselines3文档](https://stable-baselines3.readthedocs.io/)
- [交通信号控制理论](https://en.wikipedia.org/wiki/Traffic_light_control_and_coordination)

## 🤝 贡献

欢迎提交改进建议和bug报告！

---

**注意**: 本工具集专为ZFDX路口设计，但可以轻松适配其他路口场景。只需替换相应的网络文件和路由文件即可。