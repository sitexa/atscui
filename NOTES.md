# 测试说明：2023-07-17--2023-08-08

## 1. 本次测试时间为2023-07-17--2023-08-08，在MacOS M1 Ventura 13.4.1 和 Ubuntu 22.04 LTS (GPU 4090)上测试通过。
## 2. 各主要组件的版本如下：
   - python 3.10.0
   - pytorch 2.0.1
   - stable-baselines3 2.0.0a13
   - pettingzoo 1.23.1
   - supersuit 3.9.0
   - sumo-rl 1.4.3
   - sumo 1.18.0
   - traci 1.18.0
   - gymnasium 0.28.1
   - gym 0.26.2
   - ray 2.5.0
   - ray[rllib] 2.5.0
## 3. experiments里的所有案例都运行成功，画图成功。
## 4. 所有的测试结果都没有分析，所有的超参数都没有进行优化。
## 5. 后期计划
   - 结果分析
   - 超参数优化
   - 采用真实的路口数据进行训练和优化

# 日志目录说明：在config_manager.py中配置 (2025-06-13)

- evals : 用于存储模型评估的结果。eval_path="evals/default-eval-DQN.txt"。
- logs : 用于存放 TensorBoard 的日志。
- monitor : 用于 stable_baselines3.common.monitor.Monitor 模块，负责存储 SUMO 环境的监控数据。
- outs : 用于存储模型运行的输出文件，通常是 CSV 格式。csv_path="outs/default-DQN" 。
- predicts : 用于存储模型的预测结果，通常是 JSON 格式。predict_path="predicts/default-predict-DQN.json"。
