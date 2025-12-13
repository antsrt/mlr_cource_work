import time

import lcm
import numpy as np
import torch

from aliengo_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from aliengo_gym_deploy.lcm_types.obs_action_lcmt import obs_action_lcmt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class LCMAgent():
    def __init__(self, cfg, se, command_profile):
        print("Initializing LCMAgent...")
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
        self.cfg = cfg
        self.se = se
        self.command_profile = command_profile
        self.include_history_steps = self.cfg["env"].get("include_history_steps", self.cfg["env"].get("num_observation_history", 1))

        # cfg["control"]["decimation"] tells how many simulation steps happen before 
        # one control action is applied. 
        # For instance, if this is 4, it means the controller acts once every 4 sim steps.
        # cfg["sim"]["dt"] is the simulation timestep — how often the simulation itself updates (in seconds). 
        # For example, it might be 0.005 (i.e. 5 ms per step)
        # self.dt: The control timestep, i.e., how often the policy/controller runs.
        # Eg. self.dt = 4 * 0.005 = 0.02 seconds = 20 ms
        # This means the simulation runs every 5 ms. But our controller acts every 20 ms
        # In other words, if our simulator steps at 200 Hz (dt = 0.005s), 
        # but our control policy runs at 50 Hz, then we apply control every 4 sim steps. 
        # So, the below line controls how often our policy is applied inside the simulation.

        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]
        print("control->decimation = "+str(self.cfg["control"]["decimation"]))
        print("sim->dt = "+str(self.cfg["sim"]["dt"]))
        print("dt = "+str(self.dt))
        

        self.timestep = 0

        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_envs = 1
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.num_commands = self.cfg["commands"]["num_commands"]
        self.device = 'cpu'

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        # match training: scale only x/y lin vel and yaw rate
        self.commands_scale = np.array(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]]
        )


        joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", ]
        self.default_dof_pos = np.array([self.cfg["init_state"]["default_joint_angles"][name] for name in joint_names])
        try:
            self.default_dof_pos_scale = np.array([self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"],
                                                   self.cfg["init_state"]["default_hip_scales"], self.cfg["init_state"]["default_thigh_scales"], self.cfg["init_state"]["default_calf_scales"]])
        except KeyError:
            self.default_dof_pos_scale = np.ones(12)
        self.default_dof_pos = self.default_dof_pos * self.default_dof_pos_scale

        self.p_gains = np.zeros(12)
        self.d_gains = np.zeros(12)
        for i in range(12):
            joint_name = joint_names[i]
            found = False
            for dof_name in self.cfg["control"]["stiffness"].keys():
                if dof_name in joint_name:
                    self.p_gains[i] = self.cfg["control"]["stiffness"][dof_name]
                    self.d_gains[i] = self.cfg["control"]["damping"][dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg["control"]["control_type"] in ["P", "V"]:
                    print(f"PD gain of joint {joint_name} were not defined, setting them to zero")

        print(f"p_gains: {self.p_gains}")

        self.commands = np.zeros((1, self.num_commands))
        self.actions = torch.zeros((1, self.num_actions))
        self.last_actions = torch.zeros((1, self.num_actions))
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(12)
        self.dof_vel = np.zeros(12)
        self.body_linear_vel = np.zeros(3)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(12)
        self.joint_vel_target = np.zeros(12)
        self.torques = np.zeros(12)
        self.contact_state = np.ones(4)

        self.joint_idxs = self.se.joint_idxs

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float)

        if "obs_scales" in self.cfg.keys():
            self.obs_scales = self.cfg["obs_scales"]
        else:
            self.obs_scales = self.cfg["normalization"]["obs_scales"]

        self.is_currently_probing = False
        self.hip_scale_reduction = self.cfg["control"].get("hip_scale_reduction", 1.0)

    def set_probing(self, is_currently_probing):
        self.is_currently_probing = is_currently_probing

    def get_obs(self):
        self.gravity_vector = self.se.get_gravity_vector()
        cmds, reset_timer = self.command_profile.get_command(self.timestep * self.dt, probe=self.is_currently_probing)
        self.commands[:, :] = cmds[:self.num_commands]
        # sensors
        if reset_timer:
            self.reset_gait_indices()
        self.dof_pos = self.se.get_dof_pos()
        self.dof_vel = self.se.get_dof_vel()
        self.body_linear_vel = self.se.get_body_linear_vel()
        self.body_angular_vel = self.se.get_body_angular_vel()

        clip_actions = self.cfg["normalization"]["clip_actions"]
        action_clip = torch.clamp(self.actions, -clip_actions, clip_actions).to(self.device)

        ob_tensors = [
            torch.tensor(self.body_angular_vel.reshape(1, -1), device=self.device, dtype=torch.float32) * self.obs_scales["ang_vel"],
            torch.tensor(self.gravity_vector.reshape(1, -1), device=self.device, dtype=torch.float32),
            torch.tensor(self.commands[:, :3] * self.commands_scale, device=self.device, dtype=torch.float32),
            torch.tensor((self.dof_pos - self.default_dof_pos).reshape(1, -1), device=self.device, dtype=torch.float32) * self.obs_scales["dof_pos"],
            torch.tensor(self.dof_vel.reshape(1, -1), device=self.device, dtype=torch.float32) * self.obs_scales["dof_vel"],
            action_clip,
        ]

        obs = torch.cat(ob_tensors, dim=1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def get_privileged_observations(self):
        clip_actions = self.cfg["normalization"]["clip_actions"]
        action_clip = torch.clamp(self.actions, -clip_actions, clip_actions).to(self.device)
        priv_tensors = [
            torch.tensor(self.body_linear_vel.reshape(1, -1), device=self.device, dtype=torch.float32) * self.obs_scales["lin_vel"],
            torch.tensor(self.body_angular_vel.reshape(1, -1), device=self.device, dtype=torch.float32) * self.obs_scales["ang_vel"],
            torch.tensor(self.gravity_vector.reshape(1, -1), device=self.device, dtype=torch.float32),
            torch.tensor(self.commands[:, :3] * self.commands_scale, device=self.device, dtype=torch.float32),
            torch.tensor((self.dof_pos - self.default_dof_pos).reshape(1, -1), device=self.device, dtype=torch.float32) * self.obs_scales["dof_pos"],
            torch.tensor(self.dof_vel.reshape(1, -1), device=self.device, dtype=torch.float32) * self.obs_scales["dof_vel"],
            action_clip,
        ]
        priv = torch.cat(priv_tensors, dim=1)
        priv = torch.nan_to_num(priv, nan=0.0, posinf=0.0, neginf=0.0)
        return priv
    
    def publish_obs_n_act(self, obs, action):
        data = obs_action_lcmt()
        data.observation = obs["obs"][0,:].detach().cpu().numpy()
        data.action = action[0, :12].detach().cpu().numpy()
        lc.publish("obs_action", data.encode())

    def publish_action(self, action, calibrated=False, hard_reset=False):

        command_for_robot = pd_tau_targets_lcmt()
        self.joint_pos_target = \
            (action[0, :12].detach().cpu().numpy() * self.cfg["control"]["action_scale"]).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.hip_scale_reduction
        # self.joint_pos_target[[0, 3, 6, 9]] *= -1
        self.joint_pos_target = self.joint_pos_target
        self.joint_pos_target += self.default_dof_pos
        joint_pos_target = self.joint_pos_target[self.joint_idxs]
        self.joint_vel_target = np.zeros(12)
        
        # print(f'cjp {self.joint_pos_target}')

        command_for_robot.q_des = joint_pos_target
        command_for_robot.qd_des = self.joint_vel_target
        # Используем те же PD, что в обучении AMP/residual (kp=40, kd=1.2).
        command_for_robot.kp = np.ones(12) * 40.0
        command_for_robot.kd = np.ones(12) * 1.2 
        command_for_robot.tau_ff = np.zeros(12)
        command_for_robot.se_contactState = np.zeros(4)
        command_for_robot.timestamp_us = int(time.time() * 10 ** 6)
        command_for_robot.id = 0

        if hard_reset:
            command_for_robot.id = -1

        command_for_robot.calibrated = calibrated

        self.torques = (self.joint_pos_target - self.dof_pos) * self.p_gains + (self.joint_vel_target - self.dof_vel) * self.d_gains
        lc.publish("pd_plustau_targets", command_for_robot.encode())

    def reset(self):
        self.actions = torch.zeros((1, self.num_actions), device=self.device)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def reset_gait_indices(self):
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float)

    def step(self, actions, calibrated=False, hard_reset=False):
        clip_actions = self.cfg["normalization"]["clip_actions"]
        self.last_actions = self.actions[:]
        actions = actions.to(self.device)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, calibrated, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0: print(f'frq: {1 / (time.time() - self.time)} Hz')
        self.time = time.time()
        obs = self.get_obs()

        # clock accounting
        if self.num_commands >= 9:
            frequencies = torch.tensor(self.commands[:, 4], device=self.device, dtype=torch.float32)
            phases = torch.tensor(self.commands[:, 5], device=self.device, dtype=torch.float32)
            offsets = torch.tensor(self.commands[:, 6], device=self.device, dtype=torch.float32)
            if self.num_commands == 8:
                bounds = 0
                durations = torch.tensor(self.commands[:, 7], device=self.device, dtype=torch.float32)
            else:
                bounds = torch.tensor(self.commands[:, 7], device=self.device, dtype=torch.float32)
                durations = torch.tensor(self.commands[:, 8], device=self.device, dtype=torch.float32)
        else:
            frequencies = torch.zeros_like(self.gait_indices)
            phases = torch.zeros_like(self.gait_indices)
            offsets = torch.zeros_like(self.gait_indices)
            bounds = 0
            durations = torch.zeros_like(self.gait_indices)
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if "pacing_offset" in self.cfg["commands"] and self.cfg["commands"]["pacing_offset"]:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + phases]
        else:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + bounds,
                                 self.gait_indices + phases]
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * self.foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * self.foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * self.foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * self.foot_indices[3])


        images = {'front': self.se.get_camera_front(),
                  'bottom': self.se.get_camera_bottom(),
                  'rear': self.se.get_camera_rear(),
                  'left': self.se.get_camera_left(),
                  'right': self.se.get_camera_right()
                  }
        downscale_factor = 2
        temporal_downscale = 3

        privileged_obs = self.get_privileged_observations()
        infos = {"joint_pos": self.dof_pos[np.newaxis, :],
                 "joint_vel": self.dof_vel[np.newaxis, :],
                 "joint_pos_target": self.joint_pos_target[np.newaxis, :],
                 "joint_vel_target": self.joint_vel_target[np.newaxis, :],
                 "body_linear_vel": self.body_linear_vel[np.newaxis, :],
                 "body_angular_vel": self.body_angular_vel[np.newaxis, :],
                 "contact_state": self.contact_state[np.newaxis, :],
                 "clock_inputs": self.clock_inputs[np.newaxis, :],
                 "body_linear_vel_cmd": self.commands[:, 0:2],
                 "body_angular_vel_cmd": self.commands[:, 2:],
                 "privileged_obs": privileged_obs,
                 }

        self.timestep += 1
        return obs, None, None, infos
