"""This module contains the TrafficSignal class, which represents a traffic signal in the simulation."""
import os
import sys
from typing import Callable, List, Union

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")
import numpy as np
from gymnasium import spaces

import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class BaseTrafficSignal:
    """Base class for a Traffic Signal controlling an intersection.

    It is responsible for retrieving information and changing the traffic phase using the Traci API.

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).
    """

    # Default min gap of SUMO (see https://sumo.dlr.de/docs/Simulation/Safety.html).
    MIN_GAP = 2.5

    def __init__(
        self,
        env,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        begin_time: int,
        reward_fn: Union[str, Callable],
        sumo,
    ):
        self.id = ts_id
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.sumo = sumo
        self.switching_penalty = 0.5 # New: 切换相位的惩罚。

        if isinstance(reward_fn, str):
            if reward_fn in self.reward_fns:
                self.reward_fn = self.reward_fns[reward_fn]
            else:
                raise NotImplementedError(f"Reward function {reward_fn} not implemented")
        else:
            self.reward_fn = reward_fn

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.observation_fn = self.env.observation_class(self)
        self.observation_space = self.observation_fn.observation_space

        self.action_space = None  # To be defined by subclasses

    def _build_phases(self):
        """Correctly builds traffic signal phases from the SUMO .net.xml file."""
        # Get all program logics for the traffic signal
        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        if not programs:
            raise ValueError(f"No program logic found for traffic light {self.id}")

        # We use the first program (usually programID '0' or 'default')
        logic = programs[0]
        phases = logic.phases

        self.green_phases = []
        self.yellow_dict = {}

        # First pass: identify all unique green phases
        for phase in phases:
            state = phase.state
            # A green phase contains 'G' or 'g' and does not contain 'y'
            if ('G' in state or 'g' in state) and 'y' not in state:
                # Avoid adding duplicate green phases if the logic contains them
                if state not in [p.state for p in self.green_phases]:
                    self.green_phases.append(phase)
        
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        # Second pass: build the yellow phase dictionary
        # This part is crucial for smooth transitions between green phases
        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s_idx in range(len(p1.state)):
                    # If a light was green in p1 and is red in p2, it should be yellow
                    if (p1.state[s_idx] in 'Gg') and (p2.state[s_idx] in 'r'):
                        yellow_state += 'y'
                    else:
                        yellow_state += p1.state[s_idx]
                
                # Check if this yellow state already exists
                if yellow_state not in [p.state for p in self.all_phases]:
                    self.yellow_dict[(i, j)] = len(self.all_phases)
                    self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))
                else:
                    # Find existing yellow phase index
                    for idx, p in enumerate(self.all_phases):
                        if p.state == yellow_state:
                            self.yellow_dict[(i, j)] = idx
                            break

        # Ensure the logic Sumo uses is updated with our parsed phases
        # This is important if the original file had a different structure
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        # Set the initial state to the first green phase
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.green_phases[0].state)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, action: int, continuous_values=None):
        """Set the next phase for the traffic signal.
        
        Args:
            action: The action to take (discrete)
            continuous_values: Optional continuous action values for enhanced control
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def compute_observation(self):
        return self.observation_fn()

    def compute_reward(self):
        self.last_reward = self.reward_fn(self)
        if np.isnan(self.last_reward):
            logger.warning("Reward is NaN!")
        return self.last_reward

    def _pressure_reward(self):
        return self.get_pressure()

    def _average_speed_reward(self):
        return self.get_average_speed()

    def _queue_reward(self):
        return -self.get_total_queued()

    def _diff_waiting_time_reward(self):
        ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _red_green_pressure_diff(self):
        """Computes the pressure reward and applies a penalty for switching phases."""
        # 1. 计算基础的压力回报
        try:
            current_phase_state = self.green_phases[self.green_phase].state
        except IndexError:
            return -1  # Fallback for invalid phase

        green_lanes_queue = 0
        red_lanes_queue = 0
        lanes_queue = self.get_lanes_queue()
        controlled_links = self.sumo.trafficlight.getControlledLinks(self.id)

        lane_to_state_index = {}
        link_index = 0
        for links in controlled_links:
            if links:
                lane = links[0][0]
                if lane not in lane_to_state_index:
                    lane_to_state_index[lane] = link_index
                    link_index += 1

        for i, lane_id in enumerate(self.lanes):
            if lane_id in lane_to_state_index:
                state_index = lane_to_state_index[lane_id]
                if state_index < len(current_phase_state):
                    light_state = current_phase_state[state_index]
                    if light_state.lower() == 'g':
                        green_lanes_queue += lanes_queue[i]
                    else:
                        red_lanes_queue += lanes_queue[i]

        pressure_reward = red_lanes_queue - green_lanes_queue

        # 2. 引入切换惩罚
        switch_penalty = self.switching_penalty if self.time_since_last_phase_change == 1 else 0

        # 3. 返回最终回报
        return pressure_reward - switch_penalty

    def _red_lane_queue_penalty(self):
        """Computes the pressure reward based ONLY on the queue of lanes with a red light."""
        # 1. 计算红灯压力
        try:
            current_phase_state = self.green_phases[self.green_phase].state
        except IndexError:
            return -100  # Return a large penalty

        red_lanes_queue = 0
        lanes_queue = self.get_lanes_queue()
        
        controlled_links = self.sumo.trafficlight.getControlledLinks(self.id)
        lane_to_state_index = {}
        link_index = 0
        for links in controlled_links:
            if links:
                lane = links[0][0]
                if lane not in lane_to_state_index:
                    lane_to_state_index[lane] = link_index
                    link_index += 1

        for i, lane_id in enumerate(self.lanes):
            if lane_id in lane_to_state_index:
                state_index = lane_to_state_index[lane_id]
                if state_index < len(current_phase_state):
                    light_state = current_phase_state[state_index]
                    if light_state.lower() != 'g':
                        red_lanes_queue += lanes_queue[i]

        # 2. 引入切换惩罚
        switch_penalty = self.switching_penalty if self.time_since_last_phase_change == 1 else 0

        # 3. 返回最终回报 (只包含红灯压力和切换惩罚)
        # Pressure is a penalty, so it should be negative.
        return (red_lanes_queue * -1) - switch_penalty

    def _weighted_sum_reward(self): #新的奖励函数
        # Component 1: 直接惩罚全局排队长度
        queue_penalty = -self.get_total_queued() * 1.0

        # Component 2: 奖励等待时间减少
        current_ts_wait = sum(self.get_accumulated_waiting_time_per_lane()) / 100.0
        diff_waiting_time_reward = (self.last_measure - current_ts_wait) * 0.5
        self.last_measure = current_ts_wait # Update last_measure for the next step

        # Component 3: 惩罚不必要的相位切换
        # If time_since_last_phase_change is 1, it means a phase change happened in the last step.
        switch_penalty = -self.switching_penalty if self.time_since_last_phase_change == 1 else 0

        total_reward = queue_penalty + diff_waiting_time_reward + switch_penalty
        return total_reward

    def _observation_fn_default(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        density = self.get_lanes_density()
        queue = self.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_accumulated_waiting_time_per_lane(self) -> List[float]:
        wait_time_per_lane = []
        for lane in self.lanes:
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh)
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum(
                        [self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane]
                    )
                wait_time += self.env.vehicles[veh][veh_lane]
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane

    def get_average_speed(self) -> float:
        avg_speed = 0.0
        vehs = self._get_veh_list()
        if len(vehs) == 0:
            return 1.0
        for v in vehs:
            avg_speed += self.sumo.vehicle.getSpeed(v) / self.sumo.vehicle.getAllowedSpeed(v)
        return avg_speed / len(vehs)

    def get_pressure(self):
        return sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes) - sum(
            self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes
        )

    def get_lanes_density(self) -> List[float]:
        lanes_density = [
            self.sumo.lane.getLastStepVehicleNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, density) for density in lanes_density]

    def get_lanes_queue(self) -> List[float]:
        lanes_queue = [
            self.sumo.lane.getLastStepHaltingNumber(lane)
            / (self.lanes_length[lane] / (self.MIN_GAP + self.sumo.lane.getLastStepLength(lane)))
            for lane in self.lanes
        ]
        return [min(1, queue) for queue in lanes_queue]

    def get_total_queued(self) -> int:
        return sum(self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes)

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list

    @classmethod
    def register_reward_fn(cls, fn: Callable):
        if fn.__name__ in cls.reward_fns:
            raise KeyError(f"Reward function {fn.__name__} already exists")
        cls.reward_fns[fn.__name__] = fn

    reward_fns = {
        "diff-waiting-time": _diff_waiting_time_reward,
        "average-speed": _average_speed_reward,
        "queue": _queue_reward,
        "net_flow_pressure": _pressure_reward,  # 出口车道车辆数 - 入口车道车辆数
        "red_green_pressure_diff": _red_green_pressure_diff,  # 红灯车道排队数 - 绿灯车道排队数
        "red_lane_queue_penalty": _red_lane_queue_penalty,  # 仅基于红灯车道排队长度的负奖励
        "weighted-sum": _weighted_sum_reward,
        # 保持向后兼容性的别名
        # "pressure": _pressure_reward,
        # "pressure_v2": _red_green_pressure_diff,
        # "pressure_v3": _red_lane_queue_penalty,
    }


class TrafficSignal(BaseTrafficSignal):
    """Traffic Signal with discrete action space."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Action space depends on the current control mode, which is dynamic
        # For sequential, it's 2 (stay/change)
        # For flexible, it's num_green_phases
        # We set it to the maximum possible for now, and it will be handled dynamically
        self.action_space = spaces.Discrete(max(2, self.num_green_phases)) 

    def set_next_phase(self, action: int, continuous_values=None):
        """Set the next phase for discrete traffic signal control.
        
        Args:
            action: The discrete action to take
            continuous_values: Ignored for discrete control (compatibility parameter)
        """
        if self.env.current_control_mode == 'sequential':
            is_change_action = (action == 1)
            is_min_green_passed = (self.time_since_last_phase_change >= self.yellow_time + self.min_green)

            if is_change_action and is_min_green_passed:
                next_phase_index = (self.green_phase + 1) % self.num_green_phases
                self.sumo.trafficlight.setRedYellowGreenState(
                    self.id, self.all_phases[self.yellow_dict[(self.green_phase, next_phase_index)]].state
                )
                self.green_phase = next_phase_index
                self.next_action_time = self.env.sim_step + self.delta_time
                self.is_yellow = True
                self.time_since_last_phase_change = 0
            else:
                self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
                self.next_action_time = self.env.sim_step + self.delta_time
        else:  # flexible mode
            new_phase = int(action)
            if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
                self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
                self.next_action_time = self.env.sim_step + self.delta_time
            else:
                self.sumo.trafficlight.setRedYellowGreenState(
                    self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
                )
                self.green_phase = new_phase
                self.next_action_time = self.env.sim_step + self.delta_time
                self.is_yellow = True
                self.time_since_last_phase_change = 0


class ContinuousTrafficSignal(BaseTrafficSignal):
    """Traffic Signal with continuous action space."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_green_phases,), dtype=np.float32)
        # 添加相位切换稳定性控制
        self.phase_stability_threshold = 0.1  # 相位切换的最小置信度差异
        self.last_action_values = None

    def set_next_phase(self, new_phase: int, continuous_values=None):
        """
        Sets the next green phase with stable continuous control.
        
        Args:
            new_phase: The discrete phase to switch to
            continuous_values: The original continuous action values for fine-grained control
        """
        # 相位切换稳定性检查
        if continuous_values is not None:
            # 检查是否应该切换相位（避免因微小差异导致的频繁切换）
            if self.last_action_values is not None:
                current_max_value = continuous_values[new_phase]
                last_max_value = np.max(self.last_action_values)
                # 只有当新相位的置信度明显高于之前的最大值时才切换
                if current_max_value - last_max_value < self.phase_stability_threshold:
                    new_phase = self.green_phase  # 保持当前相位
            
            self.last_action_values = continuous_values.copy()
        
        # 计算当前相位的扩展持续时间（如果提供了连续值）
        phase_duration = self.delta_time
        if continuous_values is not None and new_phase < len(continuous_values):
            # 使用连续值来微调相位持续时间，但不修改全局delta_time
            continuous_factor = continuous_values[new_phase]
            # 将连续值映射到持续时间调整因子 [0.8, 1.3] - 更保守的范围
            duration_multiplier = 0.8 + continuous_factor * 0.5
            phase_duration = int(self.delta_time * duration_multiplier)
            # 确保在合理范围内
            phase_duration = max(self.min_green, min(self.max_green, phase_duration))
        
        if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.next_action_time = self.env.sim_step + phase_duration
        else:
            self.sumo.trafficlight.setRedYellowGreenState(
                self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state
            )
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + phase_duration
            self.is_yellow = True
            self.time_since_last_phase_change = 0
