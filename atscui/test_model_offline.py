import os
import sys
from pathlib import Path
import numpy as np
from gymnasium.spaces import Box
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3 import SAC


@dataclass
class SensorData:
    """传感器原始数据结构"""
    lane_id: str
    vehicle_count: int          # 车道上的车辆数量
    halting_count: int          # 停车数量
    average_speed: float        # 平均速度 (m/s)
    waiting_time: float         # 累积等待时间 (s)
    lane_length: float          # 车道长度 (m)
    timestamp: float            # 时间戳


@dataclass
class IntersectionConfig:
    """路口配置信息"""
    intersection_id: str
    lanes: List[str]            # 车道ID列表
    num_phases: int             # 相位数量
    min_green_time: int         # 最小绿灯时间
    yellow_time: int            # 黄灯时间
    cci_weights: Dict[str, float] = None  # CCI权重配置
    cci_threshold: float = 0.5  # CCI阈值
    
    def __post_init__(self):
        if self.cci_weights is None:
            self.cci_weights = {'queue': 0.4, 'wait': 0.4, 'speed': 0.2}


def load_model(model_path: str, algo_name: str):
    """加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        algo_name: 算法名称 (DQN, PPO, A2C, SAC)
        
    Returns:
        加载的模型实例
    """
    model_obj = Path(model_path)
    if not model_obj.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if algo_name == "DQN":
        model = DQN.load(model_path)
    elif algo_name == "PPO":
        model = PPO.load(model_path)
    elif algo_name == "A2C":
        model = A2C.load(model_path)
    elif algo_name == "SAC":
        model = SAC.load(model_path)
    else:
        raise ValueError(f"不支持的算法: {algo_name}")
    
    print(f"==========成功加载{algo_name}模型==========")
    return model


def calculate_cci_from_sensor_data(sensor_data_list: List[SensorData], 
                                   config: IntersectionConfig) -> float:
    """从传感器数据计算综合拥堵指数 (CCI)
    
    Args:
        sensor_data_list: 各车道的传感器数据列表
        config: 路口配置信息
        
    Returns:
        计算得到的CCI值 (0-1之间)
    """
    if not sensor_data_list:
        return 0.0
    
    # 1. 计算最大队列长度指标
    max_queue_ratio = 0.0
    for data in sensor_data_list:
        # 计算队列密度：停车数量 / 车道容量
        lane_capacity = data.lane_length / 7.5  # 假设每辆车占用7.5米
        queue_ratio = min(data.halting_count / lane_capacity, 1.0) if lane_capacity > 0 else 0.0
        max_queue_ratio = max(max_queue_ratio, queue_ratio)
    
    # 2. 计算最大等待时间指标
    max_waiting_time = max(data.waiting_time for data in sensor_data_list)
    norm_wait = min(max_waiting_time / 300.0, 1.0)  # 归一化到300秒
    
    # 3. 计算平均速度指标（反向）
    total_vehicles = sum(data.vehicle_count for data in sensor_data_list)
    if total_vehicles > 0:
        weighted_avg_speed = sum(data.average_speed * data.vehicle_count for data in sensor_data_list) / total_vehicles
    else:
        weighted_avg_speed = 15.0  # 默认自由流速度
    
    norm_speed_inv = 1.0 - min(weighted_avg_speed / 15.0, 1.0)  # 15 m/s为参考速度
    
    # 4. 计算加权CCI
    cci = (
        config.cci_weights['queue'] * max_queue_ratio +
        config.cci_weights['wait'] * norm_wait +
        config.cci_weights['speed'] * norm_speed_inv
    )
    
    return min(cci, 1.0)


def determine_control_mode(cci: float, config: IntersectionConfig) -> str:
    """根据CCI确定控制模式
    
    Args:
        cci: 综合拥堵指数
        config: 路口配置信息
        
    Returns:
        控制模式: 'sequential' 或 'flexible'
    """
    return 'flexible' if cci > config.cci_threshold else 'sequential'


def process_sensor_data_to_observation(sensor_data_list: List[SensorData],
                                       config: IntersectionConfig,
                                       current_phase: int,
                                       time_since_last_change: int,
                                       target_dim: int = None) -> np.ndarray:
    """将传感器数据处理成智能体观测向量
    
    Args:
        sensor_data_list: 传感器数据列表
        config: 路口配置信息
        current_phase: 当前相位
        time_since_last_change: 距离上次相位变化的时间
        target_dim: 目标观测向量维度，如果为None则自动计算
        
    Returns:
        观测向量 (numpy数组)
    """
    # 1. 相位独热编码
    phase_encoding = [1 if current_phase == i else 0 for i in range(config.num_phases)]
    
    # 2. 最小绿灯时间标志
    min_green_flag = [1 if time_since_last_change >= config.min_green_time + config.yellow_time else 0]
    
    # 3. 计算车道密度和队列
    densities = []
    queues = []
    
    for lane_id in config.lanes:
        # 查找对应车道的传感器数据
        lane_data = next((data for data in sensor_data_list if data.lane_id == lane_id), None)
        
        if lane_data:
            # 计算密度：车辆数 / 车道容量
            lane_capacity = lane_data.lane_length / 7.5  # 假设每辆车占用7.5米
            density = min(lane_data.vehicle_count / lane_capacity, 1.0) if lane_capacity > 0 else 0.0
            
            # 计算队列：停车数 / 车道容量
            queue = min(lane_data.halting_count / lane_capacity, 1.0) if lane_capacity > 0 else 0.0
        else:
            density = 0.0
            queue = 0.0
        
        densities.append(density)
        queues.append(queue)
    
    # 4. 计算CCI和控制模式
    cci = calculate_cci_from_sensor_data(sensor_data_list, config)
    control_mode = determine_control_mode(cci, config)
    control_mode_flag = 1 if control_mode == 'flexible' else 0
    
    # 5. 组装基础观测向量
    base_observation = phase_encoding + min_green_flag + densities + queues
    
    # 6. 根据目标维度调整观测向量
    if target_dim is None:
        # 默认包含CCI和控制模式
        observation = base_observation + [cci, control_mode_flag]
    elif target_dim == 43:
        # 43维: 不包含CCI和控制模式
        observation = base_observation[:]
        # 确保维度正确
        if len(observation) > 43:
            observation = observation[:43]
        elif len(observation) < 43:
            observation.extend([0.0] * (43 - len(observation)))
    elif target_dim == 45:
        # 45维: 包含CCI和控制模式
        observation = base_observation + [cci, control_mode_flag]
        # 确保维度正确
        if len(observation) > 45:
            observation = observation[:45]
        elif len(observation) < 45:
            observation.extend([0.0] * (45 - len(observation)))
    else:
        # 其他维度: 先包含所有信息，然后调整到目标维度
        observation = base_observation + [cci, control_mode_flag]
        if len(observation) > target_dim:
            observation = observation[:target_dim]
        elif len(observation) < target_dim:
            observation.extend([0.0] * (target_dim - len(observation)))
    
    return np.array(observation, dtype=np.float32)


def simulate_radar_sensor_data(config: IntersectionConfig, 
                               scenario: str = "normal") -> List[SensorData]:
    """模拟雷视机传感器数据
    
    Args:
        config: 路口配置信息
        scenario: 交通场景 ("normal", "heavy", "light", "emergency")
        
    Returns:
        模拟的传感器数据列表
    """
    current_time = time.time()
    sensor_data_list = []
    
    # 根据场景设置参数
    scenario_params = {
        "light": {"vehicle_range": (0, 3), "speed_range": (12, 15), "wait_range": (0, 10)},
        "normal": {"vehicle_range": (2, 8), "speed_range": (8, 12), "wait_range": (5, 30)},
        "heavy": {"vehicle_range": (6, 15), "speed_range": (3, 8), "wait_range": (20, 120)},
        "emergency": {"vehicle_range": (10, 20), "speed_range": (1, 5), "wait_range": (60, 300)}
    }
    
    params = scenario_params.get(scenario, scenario_params["normal"])
    
    for lane_id in config.lanes:
        # 模拟传感器检测数据
        vehicle_count = np.random.randint(*params["vehicle_range"])
        halting_count = min(vehicle_count, np.random.randint(0, vehicle_count + 1))
        average_speed = np.random.uniform(*params["speed_range"])
        waiting_time = np.random.uniform(*params["wait_range"]) * halting_count
        lane_length = 100.0  # 假设车道长度100米
        
        sensor_data = SensorData(
            lane_id=lane_id,
            vehicle_count=vehicle_count,
            halting_count=halting_count,
            average_speed=average_speed,
            waiting_time=waiting_time,
            lane_length=lane_length,
            timestamp=current_time
        )
        
        sensor_data_list.append(sensor_data)
    
    return sensor_data_list


def test_real_world_sensor_processing():
    """测试真实世界传感器数据处理"""
    print("\n========== 真实世界传感器数据处理测试 ==========")
    
    # 创建路口配置
    config = IntersectionConfig(
        intersection_id="test_intersection",
        lanes=["lane_1", "lane_2", "lane_3", "lane_4"],
        num_phases=4,
        min_green_time=10,
        yellow_time=3,
        cci_weights={'queue': 0.4, 'wait': 0.4, 'speed': 0.2},
        cci_threshold=0.5
    )
    
    # 测试不同交通场景
    scenarios = ["light", "normal", "heavy", "emergency"]
    
    for scenario in scenarios:
        print(f"\n--- {scenario.upper()} 交通场景 ---")
        
        # 模拟传感器数据
        sensor_data = simulate_radar_sensor_data(config, scenario)
        
        # 显示原始传感器数据
        print("原始传感器数据:")
        for data in sensor_data:
            print(f"  {data.lane_id}: 车辆={data.vehicle_count}, 停车={data.halting_count}, "
                  f"速度={data.average_speed:.1f}m/s, 等待={data.waiting_time:.1f}s")
        
        # 计算CCI
        cci = calculate_cci_from_sensor_data(sensor_data, config)
        control_mode = determine_control_mode(cci, config)
        
        print(f"计算结果:")
        print(f"  CCI = {cci:.3f}")
        print(f"  控制模式 = {control_mode}")
        
        # 生成观测向量 (使用默认维度)
        observation = process_sensor_data_to_observation(
            sensor_data, config, current_phase=0, time_since_last_change=15
        )
        
        print(f"  观测向量维度 = {observation.shape}")
        print(f"  观测向量 = {observation}")


def generate_test_observations(num_samples: int = 10, obs_dim: int = None, seed: int = 42):
    """生成测试用的观测数据
    
    Args:
        num_samples: 生成的样本数量
        obs_dim: 观测维度，如果为None则使用默认45维
        seed: 随机种子
        
    Returns:
        观测数据列表
    """
    if obs_dim is None:
        obs_dim = 45
        
    np.random.seed(seed)
    observations = []
    
    for i in range(num_samples):
        # 生成符合交通信号控制特征的观测数据
        obs = np.zeros(obs_dim, dtype=np.float32)
        
        # 相位独热编码 (假设4个相位)
        phase_idx = i % 4
        if phase_idx < obs_dim:
            obs[phase_idx] = 1.0
        
        # 最小绿灯时间标志
        if obs_dim > 4:
            obs[4] = np.random.choice([0, 1])
        
        # 根据观测维度调整车道数量
        if obs_dim == 43:
            # 43维: 4(相位) + 1(绿灯标志) + 19(密度) + 19(队列) = 43
            density_lanes = 19
            queue_lanes = 19
            density_start = 5
            queue_start = 24
        elif obs_dim == 45:
            # 45维: 4(相位) + 1(绿灯标志) + 20(密度) + 20(队列) = 45
            density_lanes = 20
            queue_lanes = 20
            density_start = 5
            queue_start = 25
        else:
            # 其他维度，动态计算
            remaining = obs_dim - 5  # 减去相位和绿灯标志
            density_lanes = min(remaining // 2, 20)
            queue_lanes = min(remaining - density_lanes, 20)
            density_start = 5
            queue_start = 5 + density_lanes
        
        # 车道密度
        for j in range(density_start, min(density_start + density_lanes, obs_dim)):
            obs[j] = np.random.uniform(0, 1)
            
        # 车道队列
        for j in range(queue_start, min(queue_start + queue_lanes, obs_dim)):
            obs[j] = np.random.uniform(0, 1)
            
        # CCI和控制模式 (仅在45维时添加)
        if obs_dim == 45 and obs_dim > 43:
            obs[43] = np.random.uniform(0, 1)  # CCI
            obs[44] = np.random.choice([0, 1])  # 控制模式
        
        observations.append(obs.reshape(1, -1))  # 添加batch维度
    
    return observations


def generate_realistic_scenarios(obs_dim: int = 45):
    """生成现实场景的测试数据
    
    Args:
        obs_dim: 观测向量维度
    
    Returns:
        包含不同交通场景的观测数据
    """
    scenarios = {}
    
    if obs_dim == 43:
        # 43维场景 (不包含CCI和控制模式)
        scenarios = {
            "空闲时段": np.array([1, 0, 0, 0, 1] + [0.1] * 19 + [0.05] * 19, dtype=np.float32),
            "高峰时段": np.array([0, 1, 0, 0, 1] + [0.8] * 19 + [0.9] * 19, dtype=np.float32),
            "不均衡流量": np.array([0, 0, 1, 0, 1] + ([0.9, 0.2, 0.8, 0.1] * 5)[:19] + ([0.8, 0.1, 0.7, 0.05] * 5)[:19], dtype=np.float32),
            "紧急情况": np.array([0, 0, 0, 1, 0] + [0.95] * 19 + [0.98] * 19, dtype=np.float32)
        }
    elif obs_dim == 45:
        # 45维场景 (包含CCI和控制模式)
        scenarios = {
            "空闲时段": np.array([1, 0, 0, 0, 1] + [0.1] * 20 + [0.05] * 20 + [0.2, 0], dtype=np.float32),
            "高峰时段": np.array([0, 1, 0, 0, 1] + [0.8] * 20 + [0.9] * 20 + [0.8, 1], dtype=np.float32),
            "不均衡流量": np.array([0, 0, 1, 0, 1] + [0.9, 0.2, 0.8, 0.1] * 5 + [0.8, 0.1, 0.7, 0.05] * 5 + [0.6, 0], dtype=np.float32),
            "紧急情况": np.array([0, 0, 0, 1, 0] + [0.95] * 20 + [0.98] * 20 + [0.9, 1], dtype=np.float32)
        }
    else:
        # 其他维度，生成简化场景
        for scenario_name in ["空闲时段", "高峰时段", "不均衡流量", "紧急情况"]:
            obs = np.random.uniform(0, 1, obs_dim).astype(np.float32)
            # 设置相位编码
            if obs_dim >= 4:
                obs[:4] = 0
                obs[0] = 1  # 默认相位0
            scenarios[scenario_name] = obs
    
    return {name: obs.reshape(1, -1) for name, obs in scenarios.items()}


def test_model_offline(model_path: str, algo_name: str):
    """脱离仿真环境测试模型
    
    Args:
        model_path: 模型文件路径
        algo_name: 算法名称
    """
    print(f"\n========== 开始脱离仿真环境测试 {algo_name} 模型 ==========")
    print(f"模型路径: {model_path}")
    
    # 加载模型
    try:
        model = load_model(model_path, algo_name)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    print(f"\n模型信息:")
    print(f"- 算法: {algo_name}")
    print(f"- 观测空间: {model.observation_space}")
    print(f"- 动作空间: {model.action_space}")
    
    # 获取模型期望的观测维度
    expected_obs_dim = model.observation_space.shape[0]
    print(f"模型期望观测维度: {expected_obs_dim}")
    
    # 测试1: 随机观测数据
    print(f"\n========== 测试1: 随机观测数据 ==========")
    random_observations = generate_test_observations(num_samples=5, obs_dim=expected_obs_dim)
    
    for i, obs in enumerate(random_observations):
        try:
            action, _states = model.predict(obs, deterministic=True)
            print(f"随机样本 {i+1}:")
            print(f"  观测: {obs.flatten()[:10]}... (显示前10维)")
            print(f"  动作: {action}")
            if hasattr(action, 'shape') and len(action.shape) > 0:
                print(f"  动作形状: {action.shape}")
        except Exception as e:
            print(f"随机样本 {i+1} 预测失败: {e}")
    
    # 测试2: 现实场景
    print(f"\n========== 测试2: 现实交通场景 ==========")
    scenarios = generate_realistic_scenarios(obs_dim=expected_obs_dim)
    
    for scenario_name, obs in scenarios.items():
        try:
            action, _states = model.predict(obs, deterministic=True)
            print(f"{scenario_name}:")
            print(f"  观测特征: 相位={np.argmax(obs[0][:4])}, 密度均值={np.mean(obs[0][5:25]):.3f}, 队列均值={np.mean(obs[0][25:45]):.3f}")
            print(f"  预测动作: {action}")
        except Exception as e:
            print(f"{scenario_name} 预测失败: {e}")
    
    # 测试3: 一致性测试
    print(f"\n========== 测试3: 模型一致性测试 ==========")
    test_obs = random_observations[0]
    actions = []
    
    for i in range(5):
        action, _states = model.predict(test_obs, deterministic=True)
        actions.append(action)
    
    all_same = all(np.array_equal(actions[0], action) for action in actions)
    print(f"相同输入的一致性: {'通过' if all_same else '失败'}")
    print(f"5次预测结果: {actions}")
    
    # 测试4: 边界情况
    print(f"\n========== 测试4: 边界情况测试 ==========")
    
    # 全零观测
    zero_obs = np.zeros((1, model.observation_space.shape[0]), dtype=np.float32)
    try:
        action, _states = model.predict(zero_obs, deterministic=True)
        print(f"全零观测 -> 动作: {action}")
    except Exception as e:
        print(f"全零观测预测失败: {e}")
    
    # 全一观测
    ones_obs = np.ones((1, model.observation_space.shape[0]), dtype=np.float32)
    try:
        action, _states = model.predict(ones_obs, deterministic=True)
        print(f"全一观测 -> 动作: {action}")
    except Exception as e:
        print(f"全一观测预测失败: {e}")
    
    print(f"\n========== 脱离仿真环境测试完成 ==========")


def test_zfdx_dqn():
    """测试ZFDX DQN模型"""
    model_path = "/Users/xnpeng/sumoptis/atscui/models/zfdx-model-DQN.zip"
    test_model_offline(model_path, "DQN")


def test_zszx_sac():
    """测试ZSZX SAC模型"""
    model_path = "/Users/xnpeng/sumoptis/atscui/models/zszx-2-model-SAC.zip"
    test_model_offline(model_path, "SAC")


def test_model_with_real_sensor_data(model_path: str, algo_name: str):
    """使用真实传感器数据测试模型
    
    Args:
        model_path: 模型文件路径
        algo_name: 算法名称
    """
    print(f"\n========== 使用真实传感器数据测试 {algo_name} 模型 ==========")
    
    # 加载模型
    try:
        model = load_model(model_path, algo_name)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 获取模型期望的观测维度
    expected_obs_dim = model.observation_space.shape[0]
    print(f"模型期望观测维度: {expected_obs_dim}")
    
    # 创建路口配置
    config = IntersectionConfig(
        intersection_id="real_intersection",
        lanes=["lane_1", "lane_2", "lane_3", "lane_4"],
        num_phases=4,
        min_green_time=10,
        yellow_time=3,
        cci_weights={'queue': 0.4, 'wait': 0.4, 'speed': 0.2},
        cci_threshold=0.5
    )
    
    # 测试不同交通场景下的模型决策
    scenarios = ["light", "normal", "heavy", "emergency"]
    
    for scenario in scenarios:
        print(f"\n--- {scenario.upper()} 场景下的智能体决策 ---")
        
        # 模拟传感器数据
        sensor_data = simulate_radar_sensor_data(config, scenario)
        
        # 处理成观测向量，指定目标维度
        observation = process_sensor_data_to_observation(
            sensor_data, config, current_phase=0, time_since_last_change=15,
            target_dim=expected_obs_dim
        )
        
        print(f"  生成观测向量维度: {observation.shape}")
        
        # 模型预测
        try:
            action, _states = model.predict(observation.reshape(1, -1), deterministic=True)
            
            # 计算场景特征
            total_vehicles = sum(data.vehicle_count for data in sensor_data)
            total_halting = sum(data.halting_count for data in sensor_data)
            avg_speed = np.mean([data.average_speed for data in sensor_data])
            cci = calculate_cci_from_sensor_data(sensor_data, config)
            control_mode = determine_control_mode(cci, config)
            
            print(f"  场景特征: 车辆={total_vehicles}, 停车={total_halting}, 平均速度={avg_speed:.1f}m/s")
            print(f"  CCI={cci:.3f}, 控制模式={control_mode}")
            print(f"  智能体决策: {action}")
            
            # 解释决策
            if algo_name in ["DQN", "PPO"]:
                phase_names = ["南北直行", "南北左转", "东西直行", "东西左转"]
                if isinstance(action, (int, np.integer)):
                    print(f"  决策解释: 选择相位 {action} ({phase_names[action % 4]})")
                else:
                    print(f"  决策解释: 选择相位 {action[0]} ({phase_names[action[0] % 4]})")
            elif algo_name == "SAC":
                print(f"  决策解释: 连续动作值 {action}")
                
        except Exception as e:
            print(f"  模型预测失败: {e}")
            print(f"  观测向量形状: {observation.shape}")
            print(f"  期望维度: {expected_obs_dim}")


def main():
    """主函数"""
    print("脱离仿真环境的模型测试工具")
    print("=" * 50)
    
    # 1. 测试真实传感器数据处理算法
    test_real_world_sensor_processing()
    
    # 2. 测试可用的模型
    models_to_test = [
        ("/Users/xnpeng/sumoptis/atscui/models/zfdx-model-DQN.zip", "DQN"),
        ("/Users/xnpeng/sumoptis/atscui/models/zfdx-model-SAC.zip", "SAC"),
        ("/Users/xnpeng/sumoptis/atscui/models/zfdx-model-PPO.zip", "PPO"),
        ("/Users/xnpeng/sumoptis/atscui/models/zszx-2-model-SAC.zip", "SAC"),
    ]
    
    for model_path, algo_name in models_to_test:
        if Path(model_path).exists():
            # 标准离线测试
            test_model_offline(model_path, algo_name)
            # 真实传感器数据测试
            test_model_with_real_sensor_data(model_path, algo_name)
        else:
            print(f"\n模型文件不存在，跳过测试: {model_path}")
    
    print("\n========== 所有测试完成 ==========")
    print("\n总结:")
    print("1. 真实传感器数据处理算法已实现，包括:")
    print("   - 从雷视机等检测设备获取原始数据")
    print("   - 计算综合拥堵指数(CCI)")
    print("   - 确定控制模式(sequential/flexible)")
    print("   - 生成智能体观测向量")
    print("2. 模型可以脱离SUMO仿真环境，直接使用真实传感器数据进行决策")
    print("3. 支持多种交通场景的测试和验证")


if __name__ == "__main__":
    main()
