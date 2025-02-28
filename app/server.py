from flask import Flask, request, jsonify
import torch
from stable_baselines3 import DQN, PPO, A2C, SAC
import os

app = Flask(__name__)

# 加载模型
model_path = '/Users/xnpeng/sumoptis/atscui/models/'
model_dqn = DQN.load(os.path.join(model_path, 'zszx-1-model-DQN.zip'))
model_ppo = PPO.load(os.path.join(model_path, 'zszx-1-model-PPO.zip'))
model_ppo2 = PPO.load(os.path.join(model_path, 'zszx-2-model-PPO.zip'))
model_a2c = A2C.load(os.path.join(model_path, 'zszx-2-model-A2C.zip'))
model_sac = SAC.load(os.path.join(model_path, 'zszx-2-model-SAC.zip'))


@app.route('/predict', methods=['POST'])
def predict():
    algo = request.args.get('algo', 'DQN').upper()
    print(f"Using algorithm: {algo}")

    data = request.get_json()

    # 这里可以处理输入数据并转换为合适的格式
    state = data['state']  # 这里假设数据是state形式
    state_tensor = torch.tensor(state).float().unsqueeze(0)  # 转换为Tensor

    # 推理
    if algo == 'DQN':
        action = model_dqn.predict(state_tensor)
    elif algo == 'PPO':
        action = model_ppo.predict(state_tensor)
    elif algo == 'PPO2':
        action = model_ppo2.predict(state_tensor)
    elif algo == 'A2C':
        action = model_sac.predict(state_tensor)
    elif algo == 'SAC':
        action = model_sac.predict(state_tensor)
    else:
        action = None

    # 返回预测结果
    return jsonify({'action': action[0].tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6000)
