# 生产环境中部署并运行交通信号优化模型的方案

对于训练完成的交通信号智能体，需要将其部署在生产环境中，控制交通信号。通常的方法是采用边缘盒子提供软硬件支持。

## 1,生产环境【边缘盒子】安装操作系统Linux和Python环境

有几种不同方式部署模型，最简单地就是在宿主系统上直接运行推理程序，其次是使用Docker容器运行定制系统，第三是将模型转换为

## 2,安装必要的依赖

``` 
pip install stable-baselines3 torch
```

## 3,准备模型文件

将模型文件拷贝到边缘设备上的文件夹下 /models/

``` 
zszx-1-model-DQN.zip
zszx-1-model-PPO.zip
zszx-2-model-PPO.zip
zszx-2-model-A2C.zip
zszx-2-model-SAC.zip
```

## 4, 编写推理接口

```app/server.py``` 就是一个推理接口，作为一个服务运行后，接受用户请求

## 5，测试接口

```app/client.py```模拟客户端发送请求并接收响应。


## 6,状态空间 state 

状态空间由相位，车道密度，排队长度等构成，维度为：相位数+最小绿+车道密度数+车道排队长度数，我zszx-2路网中，状态空间维度=43。

``` 
"state": 
[
    # 相位独热编码 (4维)
    1, 0, 0, 0,  # 假设当前是东西直行相位

    # 最小绿灯时间标志 (1维)
    1,  # 假设已满足最小绿灯时间

    # 车道密度 (19维，范围0-1)
    # 北进口5个车道
    0.6, 0.5, 0.4,  # 3个直行车道
    0.3, 0.3,  # 2个左转车道

    # 东进口4个车道
    0.4, 0.4, 0.3,  # 3个直行车道
    0.5,  # 1个左转车道

    # 南进口5个车道
    0.5, 0.6, 0.4,  # 3个直行车道
    0.4, 0.3,  # 2个左转车道

    # 西进口5个车道
    0.6, 0.5, 0.4,  # 3个直行车道
    0.4, 0.3,  # 2个左转车道

    # 车道排队长度 (19维，范围0-1)
    # 北进口5个车道
    0.7, 0.6, 0.5,  # 3个直行车道
    0.4, 0.4,  # 2个左转车道

    # 东进口4个车道
    0.5, 0.5, 0.4,  # 3个直行车道
    0.6,  # 1个左转车道

    # 南进口5个车道
    0.6, 0.7, 0.5,  # 3个直行车道
    0.5, 0.4,  # 2个左转车道

    # 西进口5个车道
    0.7, 0.6, 0.5,  # 3个直行车道
    0.5, 0.4  # 2个左转车道
]
```
                                      
## 7,动作空间 action

对应于相位系列，有动作空间. 黄闪是过渡相位，与绿灯相位一一对应，故不是一个独立的动作元素。因此，对应于8相位周期，只有4个动作：[0,1,2,3]。

``` 
    <phase duration="40" state="GGGGGGrrrrrrrrrrrrrGGGGGGrrrrrrrrrrrrrr"/>
    <phase duration="3"  state="yyyyyyrrrrrrrrrrrrryyyyyyrrrrrrrrrrrrrr"/>
    <phase duration="20" state="rrrrrrGGGGrrrrrrrrrrrrrrrGGGGrrrrrrrrrr"/>
    <phase duration="3"  state="rrrrrryyyyrrrrrrrrrrrrrrryyyyrrrrrrrrrr"/>
    <phase duration="50" state="rrrrrrrrrrGGGGGGrrrrrrrrrrrrrGGGGGGrrrr"/>
    <phase duration="5"  state="rrrrrrrrrryyyyyyrrrrrrrrrrrrryyyyyyrrrr"/>
    <phase duration="20" state="rrrrrrrrrrrrrrrrGGGrrrrrrrrrrrrrrrrGGGG"/>
    <phase duration="5"  state="rrrrrrrrrrrrrrrryyyrrrrrrrrrrrrrrrryyyy"/>
```

## 8,不同模型

下列4个模型，是针对2个配置训练出来的，zszx-1与zszx-2的连接方式不同，对应的逻辑灯位数量不同，前一种有55个灯位，后一种有39个灯位。
不同的算法，训练出来的模型也不一样，因而有DQN,PPO,A2C,SAC等4种模型。


``` 
zszx-1-model-DQN.zip
zszx-1-model-PPO.zip
zszx-2-model-PPO.zip
zszx-2-model-A2C.zip
zszx-2-model-SAC.zip
```

## 9,测试示例

在client.py中模拟了一个状态，测试5种模型：

``` 
启动server.py:
python app/server.py

运行client.py:

python app/client.py DQN / PPO / PPO2 / A2C / SAC

结果：{'action': [2]} / {'action': [1]} / {'action': [3]} 
/ {'action': [[0.9976660013198853, 0.9714875221252441, 0.17088332772254944, 0.15951400995254517]]}
/ {'action': [[0.9761184453964233, 0.9589874744415283, 0.13569363951683044, 0.025567293167114258]]}

```

# 《新功能需求》

在跟信号机的交互中，需要解决一个状态维护问题，即：智能体获得的是环境的即时状态，包括当前相位，是否最小绿，车道密度，排队长度。
状态空间的维度可以这样计算，相位数N+是否最小绿1+车道密度M+车道排队长度M = N+1+M+M = 2M+N+1。智能体根据环境状态，推理出下
一个动作，是保持当前相位，还是切换到下一个相位？如果是保持当前相位，则不向信号机发出指令，如果要切换到下一个相位，则发出指令
切换灯组颜色（如：GGGGGGrrrrrrrrrrrrrGGGGGGrrrrrrrrrrrrrr）。这些指令发给信号机控制系统的一个特定接口，控制系统接收到指令后，按其
逻辑驱动灯组通道，保持或者改变灯组颜色，并把执行结果反馈给智能体。智能体维护一个表示是否最小绿灯的逻辑变量。当接收到下一个环境
状态的时候，智能体推理出下一个动作，再调用信号机接口发送指令。

这个需求已经通过另一个项目实现。
- signal-controller (rust): 模拟信号机
- traffic-detector (rust): 模拟雷视机
- smart-agent (Python) : 交通智能体(边缘盒子)