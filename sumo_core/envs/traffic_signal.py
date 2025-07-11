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
        phase_control: str = "flexible",
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
        self.phase_control = phase_control
        self.switching_penalty = 0.5 # New: 切换相位的惩罚。

        if isinstance(reward_fn, str):
            if reward_fn in self.reward_fns:
                self.reward_fn = self.reward_fns[reward_fn]
            else:
                raise NotImplementedError(f"Reward function {reward_fn} not implemented")
        else:
            self.reward_fn = reward_fn

        self.observation_fn = self.env.observation_class(self)

        self._build_phases()

        self.lanes = list(
            dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id))
        )  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_length = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes + self.out_lanes}

        self.observation_space = self.observation_fn.observation_space()
        self.action_space = None  # To be defined by subclasses

    def _build_phases(self):
        phases = self.sumo.trafficlight.getAllProgramLogics(self.id)[0].phases
        if self.env.fixed_ts:
            self.num_green_phases = len(phases) // 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            if "y" not in state and (state.count("r") + state.count("s") != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(self.max_green, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            for j, p2 in enumerate(self.green_phases):
                if i == j:
                    continue
                yellow_state = ""
                for s in range(len(p1.state)):
                    if (p1.state[s] == "G" or p1.state[s] == "g") and (p2.state[s] == "r" or p2.state[s] == "s"):
                        yellow_state += "y"
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i, j)] = len(self.all_phases)
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))

        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step

    def update(self):
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            self.is_yellow = False

    def set_next_phase(self, action: int):
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
        "pressure": _pressure_reward,
        "weighted-sum": _weighted_sum_reward, # New default reward function
    }


class TrafficSignal(BaseTrafficSignal):
    """Traffic Signal with discrete action space."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.phase_control == 'sequential':
            self.action_space = spaces.Discrete(2)  # 0: stay, 1: change
            # Set default reward function for sequential mode if not specified
            if isinstance(self.reward_fn, str) and self.reward_fn == "diff-waiting-time":
                self.reward_fn = self.reward_fns["weighted-sum"]
        else:  # flexible
            self.action_space = spaces.Discrete(self.num_green_phases)

    def set_next_phase(self, action: int):
        if self.phase_control == 'sequential':
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

    def set_next_phase(self, new_phase: int):
        """
        Sets the next green phase.
        Note: This method receives a discrete action (the result of argmax) from the environment.
        """
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
