"""SUMO Environment for Traffic Signal Control."""
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
import sumolib
import traci
from gymnasium.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .observations import DefaultObservationFunction, ObservationFunction
from .traffic_signal import TrafficSignal, ContinuousTrafficSignal, BaseTrafficSignal

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    raise ImportError("Please declare the environment variable 'SUMO_HOME'")

LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

import logging
from pprint import pprint

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class BaseSumoEnv(gym.Env):
    """Base SUMO Environment for Traffic Signal Control.

    Args:
        net_file (str): SUMO .net.xml file
        route_file (str): SUMO .rou.xml file
        out_csv_name (Optional[str]): name of the .csv output with simulation results. If None, no output is generated
        use_gui (bool): Whether to run SUMO simulation with the SUMO GUI
        virtual_display (Optional[Tuple[int,int]]): Resolution of the virtual display for rendering
        begin_time (int): The time step (in seconds) the simulation starts. Default: 0
        num_seconds (int): Number of simulated seconds on SUMO. The duration in seconds of the simulation. Default: 20000
        max_depart_delay (int): Vehicles are discarded if they could not be inserted after max_depart_delay seconds. Default: -1 (no delay)
        waiting_time_memory (int): Number of seconds to remember the waiting time of a vehicle (see https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getAccumulatedWaitingTime). Default: 1000
        time_to_teleport (int): Time in seconds to teleport a vehicle to the end of the edge if it is stuck. Default: -1 (no teleport)
        delta_time (int): Simulation seconds between actions. Default: 5 seconds
        yellow_time (int): Duration of the yellow phase. Default: 2 seconds
        min_green (int): Minimum green time in a phase. Default: 5 seconds
        max_green (int): Max green time in a phase. Default: 60 seconds. Warning: This parameter is currently ignored!
        single_agent (bool): If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (returns dict of observations, rewards, dones, infos).
        reward_fn (str/function/dict): String with the name of the reward function used by the agents, a reward function, or dictionary with reward functions assigned to individual traffic lights by their keys.
        observation_class (ObservationFunction): Inherited class which has both the observation function and observation space.
        add_system_info (bool): If true, it computes system metrics (total queue, total waiting time, average speed) in the info dictionary.
        add_per_agent_info (bool): If true, it computes per-agent (per-traffic signal) metrics (average accumulated waiting time, average queue) in the info dictionary.
        sumo_seed (int/string): Random seed for sumo. If 'random' it uses a randomly chosen seed.
        fixed_ts (bool): If true, it will follow the phase configuration in the route_file and ignore the actions given in the :meth:`step` method.
        sumo_warnings (bool): If true, it will print SUMO warnings.
        additional_sumo_cmd (str): Additional SUMO command line arguments.
        phase_control (str): Control mode for the traffic signal phases. Can be 'sequential' or 'flexible'.
        render_mode (str): Mode of rendering. Can be 'human' or 'rgb_array'. Default: None
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    CONNECTION_LABEL = 0

    def __init__(
        self,
        traffic_signal_class: type[BaseTrafficSignal],
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 3,
        min_green: int = 10,
        max_green: int = 60,
        single_agent: bool = False,
        reward_fn: Union[str, Callable, dict] = "diff-waiting-time",
        observation_class: ObservationFunction = DefaultObservationFunction,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        phase_control: str = "sequential",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"], "Invalid render mode."
        self.render_mode = render_mode
        self.virtual_display = virtual_display
        self.disp = None

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui or self.render_mode is not None:
            self._sumo_binary = sumolib.checkBinary("sumo-gui")
        else:
            self._sumo_binary = sumolib.checkBinary("sumo")

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        self.sim_max_time = begin_time + num_seconds
        self.delta_time = delta_time
        self.max_depart_delay = max_depart_delay
        self.waiting_time_memory = waiting_time_memory
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.reward_fn = reward_fn
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.additional_sumo_cmd = additional_sumo_cmd
        self.add_system_info = add_system_info
        self.add_per_agent_info = add_per_agent_info
        self.phase_control = phase_control
        self.label = str(BaseSumoEnv.CONNECTION_LABEL)
        BaseSumoEnv.CONNECTION_LABEL += 1
        self.sumo = None
        self.traffic_signal_class = traffic_signal_class

        if LIBSUMO:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net])
            conn = traci
        else:
            traci.start([sumolib.checkBinary("sumo"), "-n", self._net], label="init_connection" + self.label)
            conn = traci.getConnection("init_connection" + self.label)

        self.ts_ids = list(conn.trafficlight.getIDList())
        self.observation_class = observation_class

        self._create_traffic_signals(conn)

        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float("inf"), float("inf"))
        self.episode = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}
        self.step_counter = 0
        self.print_interval = 500

    def _create_traffic_signals(self, conn):
        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: self.traffic_signal_class(
                    self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green,
                    self.begin_time, self.reward_fn[ts], conn, self.phase_control
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: self.traffic_signal_class(
                    self, ts, self.delta_time, self.yellow_time, self.min_green, self.max_green,
                    self.begin_time, self.reward_fn, conn, self.phase_control
                )
                for ts in self.ts_ids
            }

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n", self._net,
            "-r", self._route,
            "--max-depart-delay", str(self.max_depart_delay),
            "--waiting-time-memory", str(self.waiting_time_memory),
            "--time-to-teleport", str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            if "DEFAULT_VIEW" not in dir(traci.gui):
                traci.gui.DEFAULT_VIEW = "View #0"
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")

    def reset(self, seed: Optional[int] = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.step_counter = 0

        if self.episode != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        self._create_traffic_signals(self.sumo)

        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            self._compute_observations()
            return self.observations

    @property
    def sim_step(self) -> float:
        return self.sumo.simulation.getTime()

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones["__all__"] = self.sim_step >= self.sim_max_time
        return dones

    def _compute_info(self):
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        self.metrics.append(info.copy())
        return info

    def _compute_observations(self):
        self.observations.update(
            {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act or self.fixed_ts}
        )
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act or self.fixed_ts}

    def _compute_rewards(self):
        self.rewards.update(
            {ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act or self.fixed_ts}
        )
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act or self.fixed_ts}

    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space

    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space

    def observation_spaces(self, ts_id: str):
        return self.traffic_signals[ts_id].observation_space

    def action_spaces(self, ts_id: str) -> gym.spaces.Discrete:
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()

    def _get_system_info(self):
        vehicles = self.sumo.vehicle.getIDList()
        speeds = [self.sumo.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.sumo.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        return {
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }

    def _get_per_agent_info(self):
        stopped = [self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids]
        accumulated_waiting_time = [sum(self.traffic_signals[ts].get_accumulated_waiting_time_per_lane()) for ts in self.ts_ids]
        average_speed = [self.traffic_signals[ts].get_average_speed() for ts in self.ts_ids]
        info = {}
        for i, ts in enumerate(self.ts_ids):
            info[f"{ts}_stopped"] = stopped[i]
            info[f"{ts}_accumulated_waiting_time"] = accumulated_waiting_time[i]
            info[f"{ts}_average_speed"] = average_speed[i]
        info["agents_total_stopped"] = sum(stopped)
        info["agents_total_accumulated_waiting_time"] = sum(accumulated_waiting_time)
        return info

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        if self.disp is not None:
            self.disp.stop()
            self.disp = None
        self.sumo = None

    def __del__(self):
        self.close()

    def render(self):
        if self.render_mode == "human":
            return
        elif self.render_mode == "rgb_array":
            img = self.disp.grab()
            return np.array(img)

    def save_csv(self, out_csv_name, episode):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + f"_conn{self.label}_ep{episode}" + ".csv", index=False)

    def encode(self, state, ts_id):
        phase = int(np.where(state[: self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1:]]
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        return min(int(density * 10), 9)


class SumoEnv(BaseSumoEnv):
    """SUMO Environment for Traffic Signal Control with discrete actions."""

    def __init__(self, **kwargs):
        super().__init__(traffic_signal_class=TrafficSignal, **kwargs)

    def step(self, action: Union[dict, int]):
        self.step_counter += 1
        if self.fixed_ts or action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False
        truncated = dones["__all__"]
        info = self._compute_info()

        if self.step_counter % self.print_interval == 0:
            print(f"==========SumoEnv-321:step {self.step_counter}::info==========")
            pprint(info)

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info

    def _apply_actions(self, actions):
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)


class ContinuousSumoEnv(BaseSumoEnv):
    """SUMO Environment for Traffic Signal Control with continuous actions."""

    def __init__(self, **kwargs):
        super().__init__(traffic_signal_class=ContinuousTrafficSignal, **kwargs)
        # The single_agent is set to True for continuous action space
        self.single_agent = True

    def step(self, action: np.ndarray):
        if self.fixed_ts or len(action) == 0:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards()
        dones = self._compute_dones()
        terminated = False
        truncated = dones["__all__"]
        info = self._compute_info()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], terminated, truncated, info
        else:
            return observations, rewards, dones, info

    def _apply_actions(self, actions):
        if self.single_agent:
            discrete_action = np.argmax(actions)
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(discrete_action)
        else:
            # This part is for multi-agent continuous control, which is not the primary use case here.
            # It converts a dictionary of continuous actions to discrete actions.
            discrete_actions = {ts: np.argmax(act) for ts, act in actions.items()}
            for ts, action in discrete_actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)


def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class SumoEnvPZ(AECEnv, EzPickle):
    """A wrapper for the SUMO environment that implements the AECEnv interface from PettingZoo."""

    metadata = {"render.modes": ["human", "rgb_array"], "name": "sumo_rl_v0", "is_parallelizable": True}

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs
        self.seed()
        # Note: The PettingZoo wrapper is designed for the discrete environment.
        self.env = SumoEnv(**self._kwargs)
        self.render_mode = self.env.render_mode

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}
        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.step_counter = 0
        self.print_interval = 1000

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.env.reset(seed=seed, **kwargs)
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.compute_info()
        self.step_counter = 0

    def compute_info(self):
        self.infos = {a: {} for a in self.agents}
        infos = self.env._compute_info()
        for a in self.agents:
            for k, v in infos.items():
                if k.startswith(a) or k.startswith("system"):
                    self.infos[a][k] = v

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        return self.env.observations[agent].copy()

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def save_csv(self, out_csv_name, episode):
        self.env.save_csv(out_csv_name, episode)

    def step(self, action):
        if self.truncations[self.agent_selection] or self.terminations[self.agent_selection]:
            return self._was_dead_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception(f"Action for agent {agent} must be in {self.action_spaces[agent]}.")
        if not self.env.fixed_ts:
            self.env._apply_actions({agent: action})
        if self._agent_selector.is_last():
            if not self.env.fixed_ts:
                self.env._run_steps()
            else:
                for _ in range(self.env.delta_time):
                    self.env._sumo_step()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.compute_info()
        else:
            self._clear_rewards()

        done = self.env._compute_dones()["__all__"]
        self.truncations = {a: done for a in self.agents}
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()