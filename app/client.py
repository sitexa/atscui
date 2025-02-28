import requests
import argparse

parser = argparse.ArgumentParser(description="输入算法名称")
parser.add_argument("algo",type=str,help="算法名称，如dqn,ppo,ppo2,a2c,sac")
args = parser.parse_args()
algo = args.algo

url = f"http://localhost:6000/predict?algo={algo}"
data = {
    "state": [
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
}

try:
    response = requests.post(url, json=data)
    response.raise_for_status()  # 如果响应状态码不是 2xx，将抛出异常
    print(response.json())  # 输出返回的预测结果
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
