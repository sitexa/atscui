# 交通信号智能体训练图形界面

本模块是原ui模块的重构。将原模块拆分为3个子模块：界面子模块，配置子模块，算法子模块。

## 模块化设计

- （1）界面组件子模块
  - 配置界面
  - 绘图界面
- （2）配置项子模块
  - 基础配置
  - 算法私有配置
- （3）算法模型子模块
  - 常用4种算法：DQN, PPO, A2C, SAC
- （4）辅助函数

## 操作类型(operation)

- TRAIN: 根据配置训练智能体，并保存模型(models/model.zip)和日志(outs/conn*.csv)
- EVAL: 评估模型，保存评估结果(evals/eval.txt)
- PREDICT: 用模型预测，保存预测结果(predicts/predict.json)

## 运行命令

在项目根目录执行命令：``` python -m atscui.main ```

## 加载模型进行预测

run_model.py是一个简易的命令行程序，指定net.xml,rou.xml,model_path,algo_name,
加载训练后的模型，根据状态选择动作，让环境执行该动作。

测试结果符合预期，即相同的状态observation，会产生相同的动作action。

```
python atscui/run_model.py 
```

```
config = parse_config()
env = createEnv(config)
model = createAlgorithm(env, config.algo_name)
model_obj = Path(config.model_path)

if model_obj.exists():
    print("==========load model==========")
    model.load(model_obj)

obs = env.reset()
for _ in range(10):
    action, _ = model.predict(obs)  # 通过状态变量预测动作变量
    # 将动作action（改变灯态的指令串）发送给信号机
    print(obs)
    for _ in range(3):
        obs, reward, done, info = env.step(action)  # 来自探测器的状态变量
        print(action)
```

## 20250228

构思智能体模型与信号机的交互，即使用优化智能体访问信号机接口，接收输入，经智能体推理后，向信号机发送输出。

app模块中初步设计了服务程序供信号机访问，恰恰反过来了！

## 手动设计路网模型(20250324)

人工手搓路网模型相比使用Sumo NetEdit具有不同的特点，可以准确掌控节点(nod.xml)的id、名称、坐标、数量等主要元素，不受图形界面上众多其他次要元素的影响，准确定义车路连接(edg.xml)，准确定义车道连接（con.xml），准确定义灯组信息(tll.xml)，然后通过netconvert命令生成路网模型net.xml，达到完全掌控，修改更新十分方便。

需求模型也可以使用人工编写，可控度高，更新方便。

### 1， 定义节点 nod.xml

### 2， 定义路连接 edg.xml

### 3， 定义车道连接 con.xml

### 4， 定义灯组信息 tll.xml

### 5， 使用netconvert生成路网net.xml

### 6， 编写需求模型 rou.xml

### 7， 编写sumocfg.xml
