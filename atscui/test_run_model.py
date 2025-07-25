import os
import sys
from pathlib import Path

import numpy as np
from gymnasium.spaces import Box

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atscui.config.base_config import RunningConfig
from atscui.environment import createEnv
from atscui.models.agent_creator import createAlgorithm


def parse_config():
    return RunningConfig(
        net_file="/Users/xnpeng/sumoptis/atscui/zszx/net/zszx-2.net.xml",
        rou_file="/Users/xnpeng/sumoptis/atscui/zszx/net/zszx-perhour-3.rou.xml",
        model_path="/Users/xnpeng/sumoptis/atscui/models/zszx-2-model-SAC.zip",
        csv_path="/Users/xnpeng/sumoptis/atscui/outs",
        predict_path="/Users/xnpeng/sumoptis/atscui/predicts",
        eval_path="/Users/xnpeng/sumoptis/atscui/evals",
        single_agent=True,
        algo_name="SAC")


def get_obs(seed: int):
    observation_space = Box(low=0, high=1, shape=(43,), seed=seed, dtype=np.float32)

    # 生成符合要求的 observation (0,1) 之间的随机值
    observation = observation_space.sample()
    return observation


def running():
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
        print("obs=====", obs)
        for _ in range(3):
            obs, reward, done, info = env.step(action)  # 来自探测器的状态变量
            print("action=====", action)


# 便捷函数
def run_model():
    return running()

if __name__ == "__main__":
    running()

"""
加载训练过的算法模型model，根据环境env的状态obs选择动作action，让环境执行该动作。
该程序与图形界面里的PREDICT操作相同。

测试结果符合预期，即相同的状态observation，会产生相同的动作action。

(1) PPO 模型
=========================

注：黄灯不算相位。下表只有4个相位：0，1，2，3

<phase duration="40" state="GGGGGGrrrrrrrrrrrrrGGGGGGrrrrrrrrrrrrrr"/>
<phase duration="3"  state="yyyyyyrrrrrrrrrrrrryyyyyyrrrrrrrrrrrrrr"/>
<phase duration="20" state="rrrrrrGGGGrrrrrrrrrrrrrrrGGGGrrrrrrrrrr"/>
<phase duration="3"  state="rrrrrryyyyrrrrrrrrrrrrrrryyyyrrrrrrrrrr"/>
<phase duration="50" state="rrrrrrrrrrGGGGGGrrrrrrrrrrrrrGGGGGGrrrr"/>
<phase duration="5"  state="rrrrrrrrrryyyyyyrrrrrrrrrrrrryyyyyyrrrr"/>
<phase duration="20" state="rrrrrrrrrrrrrrrrGGGrrrrrrrrrrrrrrrrGGGG"/>
<phase duration="5"  state="rrrrrrrrrrrrrrrryyyrrrrrrrrrrrrrrrryyyy"/>

结果：
长向量是observation,shapge(43,)
短向量是action,shape(1,)

[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
[1]
[1]
[1]
[[1.         0.         0.         0.         1.         0.
  0.04093887 0.         0.04093887 0.         0.02709538 0.02709538
  0.         0.02709538 0.         0.04242082 0.         0.04242082
  0.04242082 0.02741228 0.02741228 0.02741228 0.02741228 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]]
[2]
[2]
[2]
[[0.         0.         1.         0.         1.         0.08187773
  0.04093887 0.04093887 0.04093887 0.         0.05419075 0.
  0.         0.05419075 0.04242082 0.08484163 0.         0.12726244
  0.08484163 0.05482456 0.05482456 0.02741228 0.02741228 0.02741228
  0.04093887 0.         0.         0.04093887 0.         0.
  0.         0.         0.02709538 0.04242082 0.         0.
  0.04242082 0.04242082 0.         0.         0.         0.02741228
  0.        ]]
[3]
[3]
[3]
[[0.         0.         0.         1.         1.         0.08187773
  0.08187773 0.08187773 0.04093887 0.         0.02709538 0.05419075
  0.02709538 0.02709538 0.08484163 0.04242082 0.08484163 0.16968326
  0.12726244 0.05482456 0.08223684 0.05482456 0.02741228 0.
  0.04093887 0.04093887 0.04093887 0.04093887 0.         0.02709538
  0.         0.         0.         0.04242082 0.04242082 0.04242082
  0.12726244 0.08484163 0.02741228 0.02741228 0.         0.
  0.        ]]
[1]
[1]
[1]
[[0.         1.         0.         0.         1.         0.12281659
  0.12281659 0.08187773 0.04093887 0.         0.05419075 0.02709538
  0.05419075 0.05419075 0.08484163 0.08484163 0.12726244 0.04242082
  0.04242082 0.08223684 0.10964912 0.08223684 0.02741228 0.02741228
  0.08187773 0.08187773 0.04093887 0.         0.         0.02709538
  0.02709538 0.02709538 0.02709538 0.08484163 0.04242082 0.08484163
  0.         0.         0.05482456 0.05482456 0.02741228 0.02741228
  0.        ]]
[2]
[2]
[2]
[[0.         0.         1.         0.         1.         0.16375546
  0.12281659 0.12281659 0.04093887 0.         0.02709538 0.02709538
  0.         0.05419075 0.12726244 0.12726244 0.12726244 0.08484163
  0.08484163 0.02741228 0.05482456 0.02741228 0.05482456 0.02741228
  0.12281659 0.12281659 0.08187773 0.04093887 0.         0.
  0.         0.         0.02709538 0.08484163 0.08484163 0.12726244
  0.04242082 0.04242082 0.         0.         0.         0.02741228
  0.        ]]
[3]
[3]
[3]
[[0.         0.         0.         1.         1.         0.16375546
  0.16375546 0.16375546 0.04093887 0.         0.02709538 0.05419075
  0.         0.02709538 0.16968326 0.12726244 0.16968326 0.12726244
  0.12726244 0.05482456 0.08223684 0.05482456 0.02741228 0.
  0.16375546 0.12281659 0.12281659 0.04093887 0.         0.02709538
  0.         0.         0.         0.12726244 0.12726244 0.12726244
  0.08484163 0.08484163 0.02741228 0.02741228 0.02741228 0.
  0.        ]]
[0]
Step #100.00 (0ms ?*RT. ?UPS, TraCI: 14ms, vehicles TOT 86 ACT 52 BUF 0)                   
 Retrying in 1 seconds
[0]
[0]
[[1.         0.         0.         0.         0.         0.04093887
  0.         0.         0.04093887 0.         0.02709538 0.
  0.         0.02709538 0.04242082 0.         0.         0.04242082
  0.         0.02741228 0.         0.         0.02741228 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]]
[0]
[0]
[0]
[[1.         0.         0.         0.         1.         0.04093887
  0.04093887 0.         0.04093887 0.         0.05419075 0.
  0.         0.02709538 0.04242082 0.04242082 0.         0.08484163
  0.04242082 0.05482456 0.05482456 0.02741228 0.02741228 0.02741228
  0.         0.         0.         0.04093887 0.         0.
  0.         0.         0.         0.         0.         0.
  0.04242082 0.         0.         0.         0.         0.
  0.        ]]
[3]
[3]
[3]
[[0.         0.         0.         1.         1.         0.08187773
  0.04093887 0.04093887 0.04093887 0.         0.05419075 0.02709538
  0.02709538 0.02709538 0.04242082 0.04242082 0.04242082 0.12726244
  0.08484163 0.08223684 0.08223684 0.05482456 0.02741228 0.
  0.04093887 0.04093887 0.         0.04093887 0.         0.02709538
  0.         0.         0.         0.04242082 0.         0.
  0.08484163 0.04242082 0.02741228 0.02741228 0.02741228 0.
  0.        ]]
[3]
[3]
[3]

(2) SAC 模型
=========================
obs===== [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
action===== [[0.5398255  0.82606965 0.6208428  0.6893597 ]]
action===== [[0.5398255  0.82606965 0.6208428  0.6893597 ]]
action===== [[0.5398255  0.82606965 0.6208428  0.6893597 ]]
obs===== [[1.         0.         0.         0.         1.         0.
  0.04093887 0.         0.04093887 0.         0.02709538 0.02709538
  0.         0.02709538 0.         0.04242082 0.         0.04242082
  0.04242082 0.02741228 0.02741228 0.02741228 0.02741228 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]]
action===== [[0.8836006  0.0457429  0.799798   0.79089284]]
action===== [[0.8836006  0.0457429  0.799798   0.79089284]]
action===== [[0.8836006  0.0457429  0.799798   0.79089284]]
obs===== [[1.         0.         0.         0.         1.         0.04093887
  0.         0.04093887 0.04093887 0.         0.05419075 0.02709538
  0.         0.05419075 0.         0.04242082 0.         0.12726244
  0.08484163 0.08223684 0.08223684 0.02741228 0.02741228 0.02741228
  0.         0.         0.         0.04093887 0.         0.02709538
  0.         0.         0.02709538 0.         0.         0.
  0.04242082 0.04242082 0.02741228 0.         0.         0.02741228
  0.        ]]
action===== [[0.549207   0.78193265 0.32913688 0.4854654 ]]
action===== [[0.549207   0.78193265 0.32913688 0.4854654 ]]
action===== [[0.549207   0.78193265 0.32913688 0.4854654 ]]
obs===== [[0.         1.         0.         0.         1.         0.04093887
  0.08187773 0.04093887 0.         0.         0.05419075 0.02709538
  0.05419075 0.05419075 0.04242082 0.08484163 0.         0.04242082
  0.04242082 0.08223684 0.10964912 0.08223684 0.05482456 0.02741228
  0.04093887 0.04093887 0.         0.         0.         0.02709538
  0.02709538 0.02709538 0.02709538 0.04242082 0.         0.
  0.         0.         0.05482456 0.02741228 0.05482456 0.02741228
  0.02741228]]
action===== [[0.5403766  0.62649703 0.8097092  0.54428816]]
action===== [[0.5403766  0.62649703 0.8097092  0.54428816]]
action===== [[0.5403766  0.62649703 0.8097092  0.54428816]]
obs===== [[0.         0.         1.         0.         1.         0.08187773
  0.08187773 0.08187773 0.04093887 0.         0.02709538 0.02709538
  0.         0.08128612 0.08484163 0.04242082 0.08484163 0.08484163
  0.08484163 0.05482456 0.02741228 0.05482456 0.05482456 0.05482456
  0.04093887 0.04093887 0.04093887 0.         0.         0.
  0.         0.         0.05419075 0.04242082 0.04242082 0.04242082
  0.04242082 0.04242082 0.         0.         0.         0.02741228
  0.02741228]]
action===== [[0.51691216 0.6444877  0.47153208 0.84359026]]
action===== [[0.51691216 0.6444877  0.47153208 0.84359026]]
action===== [[0.51691216 0.6444877  0.47153208 0.84359026]]
obs===== [[0.         0.         0.         1.         1.         0.12281659
  0.08187773 0.12281659 0.04093887 0.         0.05419075 0.02709538
  0.02709538 0.02709538 0.08484163 0.08484163 0.12726244 0.12726244
  0.12726244 0.05482456 0.08223684 0.05482456 0.02741228 0.02741228
  0.08187773 0.08187773 0.08187773 0.04093887 0.         0.02709538
  0.02709538 0.         0.         0.08484163 0.04242082 0.08484163
  0.08484163 0.08484163 0.02741228 0.02741228 0.         0.
  0.        ]]
action===== [[0.05728266 0.8679855  0.70592064 0.21335316]]
action===== [[0.05728266 0.8679855  0.70592064 0.21335316]]
action===== [[0.05728266 0.8679855  0.70592064 0.21335316]]
obs===== [[0.         1.         0.         0.         1.         0.12281659
  0.12281659 0.16375546 0.         0.         0.05419075 0.05419075
  0.02709538 0.05419075 0.12726244 0.12726244 0.12726244 0.04242082
  0.04242082 0.10964912 0.08223684 0.08223684 0.02741228 0.
  0.08187773 0.08187773 0.12281659 0.         0.         0.02709538
  0.02709538 0.02709538 0.02709538 0.08484163 0.08484163 0.12726244
  0.         0.         0.05482456 0.05482456 0.02741228 0.
  0.        ]]
action===== [[0.3889625  0.46625105 0.9812868  0.9878291 ]]
Step #100.00 (0ms ?*RT. ?UPS, TraCI: 16ms, vehicles TOT 86 ACT 55 BUF 0)                   
 Retrying in 1 seconds
action===== [[0.3889625  0.46625105 0.9812868  0.9878291 ]]
action===== [[0.3889625  0.46625105 0.9812868  0.9878291 ]]
obs===== [[1.         0.         0.         0.         0.         0.04093887
  0.         0.         0.04093887 0.         0.02709538 0.
  0.         0.02709538 0.04242082 0.         0.         0.04242082
  0.         0.02741228 0.         0.         0.02741228 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]]
action===== [[0.01930815 0.34882876 0.7679751  0.3967641 ]]
action===== [[0.01930815 0.34882876 0.7679751  0.3967641 ]]
action===== [[0.01930815 0.34882876 0.7679751  0.3967641 ]]
obs===== [[0.         0.         1.         0.         0.         0.04093887
  0.04093887 0.04093887 0.04093887 0.         0.         0.02709538
  0.         0.02709538 0.04242082 0.04242082 0.         0.08484163
  0.04242082 0.08223684 0.02741228 0.02741228 0.02741228 0.02741228
  0.         0.         0.         0.04093887 0.         0.
  0.         0.         0.         0.         0.         0.
  0.04242082 0.         0.         0.         0.         0.
  0.        ]]
action===== [[0.39558157 0.3247567  0.54272443 0.70877606]]
action===== [[0.39558157 0.3247567  0.54272443 0.70877606]]
action===== [[0.39558157 0.3247567  0.54272443 0.70877606]]
obs===== [[0.         0.         0.         1.         0.         0.08187773
  0.08187773 0.04093887 0.04093887 0.         0.02709538 0.02709538
  0.02709538 0.02709538 0.04242082 0.08484163 0.04242082 0.12726244
  0.08484163 0.08223684 0.05482456 0.02741228 0.         0.02741228
  0.04093887 0.04093887 0.04093887 0.04093887 0.         0.
  0.         0.         0.         0.04242082 0.         0.
  0.08484163 0.04242082 0.         0.         0.         0.
  0.        ]]
action===== [[0.6041254  0.3302226  0.6244049  0.85403955]]
action===== [[0.6041254  0.3302226  0.6244049  0.85403955]]
action===== [[0.6041254  0.3302226  0.6244049  0.85403955]]
Step #50.00 (0ms ?*RT. ?UPS, TraCI: 126ms, vehicles TOT 43 ACT 40 BUF 0)      

"""

"""
2025-02-13 运行结果：

 python atscui/run_model.py
=====create ContinuousEnv for SAC=====
 Retrying in 1 seconds
=====ContinuousSumoEnv ts_ids= ['tl_1']
Step #0.00 (0ms ?*RT. ?UPS, TraCI: 10ms, vehicles TOT 0 ACT 0 BUF 0)                     
=====env:action_space: Box(0.0, 1.0, (4,), float32)
==========load model==========
 Retrying in 1 seconds
obs===== [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
  0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
action===== [[0.9832226 0.9060782 0.7002458 0.557242 ]]
action===== [[0.9832226 0.9060782 0.7002458 0.557242 ]]
action===== [[0.9832226 0.9060782 0.7002458 0.557242 ]]
obs===== [[1.         0.         0.         0.         1.         0.
  0.04093887 0.         0.04093887 0.         0.02709538 0.02709538
  0.         0.02709538 0.         0.04242082 0.         0.04242082
  0.04242082 0.02741228 0.02741228 0.02741228 0.02741228 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]]
action===== [[0.0979684  0.66954887 0.00990981 0.7740284 ]]
action===== [[0.0979684  0.66954887 0.00990981 0.7740284 ]]
action===== [[0.0979684  0.66954887 0.00990981 0.7740284 ]]
obs===== [[0.         0.         0.         1.         1.         0.08187773
  0.04093887 0.04093887 0.04093887 0.         0.08128612 0.
  0.         0.02709538 0.04242082 0.08484163 0.         0.12726244
  0.08484163 0.08223684 0.08223684 0.02741228 0.02741228 0.
  0.04093887 0.         0.         0.04093887 0.         0.02709538
  0.         0.         0.         0.04242082 0.         0.
  0.04242082 0.04242082 0.02741228 0.         0.         0.
  0.        ]]
action===== [[0.67823267 0.9867314  0.87866044 0.03966099]]
action===== [[0.67823267 0.9867314  0.87866044 0.03966099]]
action===== [[0.67823267 0.9867314  0.87866044 0.03966099]]
obs===== [[0.         1.         0.         0.         1.         0.08187773
  0.08187773 0.08187773 0.         0.         0.05419075 0.05419075
  0.02709538 0.02709538 0.04242082 0.08484163 0.08484163 0.04242082
  0.04242082 0.10964912 0.08223684 0.08223684 0.02741228 0.02741228
  0.04093887 0.04093887 0.04093887 0.         0.         0.02709538
  0.02709538 0.         0.         0.04242082 0.04242082 0.04242082
  0.         0.         0.05482456 0.02741228 0.05482456 0.
  0.        ]]
action===== [[0.4909009  0.68213904 0.99301183 0.09698772]]
action===== [[0.4909009  0.68213904 0.99301183 0.09698772]]
action===== [[0.4909009  0.68213904 0.99301183 0.09698772]]
obs===== [[0.         0.         1.         0.         1.         0.12281659
  0.12281659 0.08187773 0.04093887 0.         0.02709538 0.
  0.02709538 0.05419075 0.08484163 0.08484163 0.12726244 0.12726244
  0.04242082 0.05482456 0.02741228 0.02741228 0.05482456 0.02741228
  0.08187773 0.08187773 0.04093887 0.         0.         0.
  0.         0.         0.02709538 0.04242082 0.08484163 0.08484163
  0.04242082 0.         0.         0.         0.         0.02741228
  0.        ]]
action===== [[0.8379971  0.16212416 0.01390719 0.78014195]]
action===== [[0.8379971  0.16212416 0.01390719 0.78014195]]
action===== [[0.8379971  0.16212416 0.01390719 0.78014195]]
obs===== [[1.         0.         0.         0.         1.         0.04093887
  0.         0.04093887 0.04093887 0.         0.02709538 0.02709538
  0.02709538 0.05419075 0.04242082 0.         0.         0.12726244
  0.12726244 0.08223684 0.05482456 0.05482456 0.05482456 0.05482456
  0.         0.         0.         0.04093887 0.         0.
  0.         0.         0.02709538 0.         0.         0.
  0.08484163 0.08484163 0.02741228 0.02741228 0.02741228 0.02741228
  0.02741228]]
action===== [[0.6131926  0.9682949  0.60073495 0.7530304 ]]
action===== [[0.6131926  0.9682949  0.60073495 0.7530304 ]]
action===== [[0.6131926  0.9682949  0.60073495 0.7530304 ]]
obs===== [[0.         1.         0.         0.         1.         0.04093887
  0.08187773 0.04093887 0.         0.         0.05419075 0.02709538
  0.02709538 0.08128612 0.04242082 0.04242082 0.04242082 0.04242082
  0.04242082 0.1370614  0.05482456 0.08223684 0.05482456 0.05482456
  0.         0.04093887 0.         0.         0.         0.02709538
  0.02709538 0.         0.05419075 0.04242082 0.         0.
  0.         0.         0.05482456 0.05482456 0.05482456 0.05482456
  0.02741228]]
action===== [[0.65512764 0.14941344 0.92129576 0.0602074 ]]
Step #100.00 (0ms ?*RT. ?UPS, TraCI: 13ms, vehicles TOT 86 ACT 46 BUF 0)                   
 Retrying in 1 seconds
action===== [[0.65512764 0.14941344 0.92129576 0.0602074 ]]
action===== [[0.65512764 0.14941344 0.92129576 0.0602074 ]]
obs===== [[1.         0.         0.         0.         0.         0.04093887
  0.         0.         0.04093887 0.         0.02709538 0.
  0.         0.02709538 0.04242082 0.         0.         0.04242082
  0.         0.02741228 0.         0.         0.02741228 0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.         0.         0.         0.         0.         0.
  0.        ]]
action===== [[0.97913694 0.703805   0.34766948 0.7927896 ]]
action===== [[0.97913694 0.703805   0.34766948 0.7927896 ]]
action===== [[0.97913694 0.703805   0.34766948 0.7927896 ]]
obs===== [[1.         0.         0.         0.         1.         0.
  0.04093887 0.04093887 0.04093887 0.         0.02709538 0.02709538
  0.         0.02709538 0.04242082 0.04242082 0.         0.08484163
  0.04242082 0.08223684 0.02741228 0.02741228 0.02741228 0.02741228
  0.         0.         0.         0.04093887 0.         0.
  0.         0.         0.         0.         0.         0.
  0.04242082 0.         0.         0.         0.         0.
  0.        ]]
action===== [[0.14372873 0.9727735  0.4583111  0.07230189]]
action===== [[0.14372873 0.9727735  0.4583111  0.07230189]]
action===== [[0.14372873 0.9727735  0.4583111  0.07230189]]
obs===== [[0.         1.         0.         0.         1.         0.08187773
  0.08187773 0.         0.         0.         0.05419075 0.02709538
  0.02709538 0.05419075 0.04242082 0.08484163 0.04242082 0.08484163
  0.         0.08223684 0.08223684 0.05482456 0.02741228 0.02741228
  0.04093887 0.04093887 0.         0.         0.         0.02709538
  0.02709538 0.         0.02709538 0.04242082 0.04242082 0.
  0.         0.         0.02741228 0.02741228 0.02741228 0.02741228
  0.        ]]
action===== [[0.8300315  0.20475608 0.03267679 0.9069828 ]]
action===== [[0.8300315  0.20475608 0.03267679 0.9069828 ]]
action===== [[0.8300315  0.20475608 0.03267679 0.9069828 ]]
Step #50.00 (0ms ?*RT. ?UPS, TraCI: 127ms, vehicles TOT 43 ACT 39 BUF 0) 
"""