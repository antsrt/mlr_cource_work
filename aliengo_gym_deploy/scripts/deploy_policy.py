import argparse
import pathlib

import lcm

# Сначала конфиги/isaacgym (torch ещё не импортирован в нашем коде).
from legged_gym.envs.aliengo.aliengo_amp_config import AliengoAMPCfg, AliengoAMPCfgPPO
from legged_gym.envs.aliengo.aliengo_residual_config import AliengoResidualCfg, AliengoResidualCfgPPO
from legged_gym.utils.helpers import class_to_dict

# После того как isaacgym инициализировался, можно тянуть torch-зависимые модули.
from rl.rl_algorithms.proprio_base.modules.actor_critic import ActorCritic
from rl.rl_algorithms.proprio_base.modules.actor_critic_residual import ActorCriticResidual

from aliengo_gym_deploy.utils.deployment_runner import DeploymentRunner
from aliengo_gym_deploy.envs.lcm_agent import LCMAgent
from aliengo_gym_deploy.envs.history_wrapper import HistoryWrapper
from aliengo_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from aliengo_gym_deploy.utils.command_profile import RCControllerProfile


lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_AMP_CKPT = REPO_ROOT / "logs" / "aliengo_amp" / "video_limp" / "model_25000.pt"


def get_cfgs(policy_type):
    if policy_type == "amp":
        env_cfg = AliengoAMPCfg()
        ppo_cfg = AliengoAMPCfgPPO()
    else:
        env_cfg = AliengoResidualCfg()
        ppo_cfg = AliengoResidualCfgPPO()
    env_cfg_dict = class_to_dict(env_cfg)
    ppo_cfg_dict = class_to_dict(ppo_cfg)
    return env_cfg, env_cfg_dict, ppo_cfg_dict


def build_policy(policy_type, checkpoint_path, device):
    import torch
    env_cfg, env_cfg_dict, ppo_cfg_dict = get_cfgs(policy_type)
    policy_kwargs = ppo_cfg_dict["policy"]
    if policy_type == "amp":
        actor = ActorCritic(
            num_actor_obs=env_cfg.env.num_observations,
            num_critic_obs=env_cfg.env.num_privileged_obs,
            num_actions=env_cfg.env.num_actions,
            **policy_kwargs,
        )
    else:
        actor = ActorCriticResidual(
            num_actor_obs=env_cfg.env.num_observations,
            num_critic_obs=env_cfg.env.num_privileged_obs,
            num_actions=env_cfg.env.num_actions,
            **policy_kwargs,
        )

    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model_state_dict", state)
    # Compatibility with checkpoints that stored std under action_std.
    if "action_std" in state_dict and "std" not in state_dict:
        state_dict["std"] = state_dict.pop("action_std")
    missing, unexpected = actor.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys while loading policy: {missing}")
    if unexpected:
        print(f"Unexpected keys while loading policy: {unexpected}")

    actor.to(device)
    actor.eval()

    def policy_fn(obs, info):
        obs_dict = {
            "obs": obs["obs"].to(device=device, dtype=torch.float32),
            "privileged_obs": (obs.get("privileged_obs").to(device=device, dtype=torch.float32) if obs.get("privileged_obs") is not None else obs["obs"].to(device=device, dtype=torch.float32)),
        }
        if "proprio_hist" in obs:
            obs_dict["proprio_hist"] = obs["proprio_hist"].to(device=device, dtype=torch.float32)
        elif "obs_history" in obs:
            obs_dict["proprio_hist"] = obs["obs_history"].to(device=device, dtype=torch.float32)
        else:
            obs_dict["proprio_hist"] = obs["obs"].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            action = actor.act_inference(obs_dict)
        info["latent"] = 0
        return action.detach().cpu()

    return policy_fn, env_cfg_dict


def load_and_run_policy(args):
    import torch
    checkpoint = pathlib.Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is None:
        if args.policy_type == "amp" and DEFAULT_AMP_CKPT.exists():
            checkpoint = DEFAULT_AMP_CKPT
        else:
            raise ValueError("Provide --checkpoint for the selected policy type.")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.device)
    policy, env_cfg_dict = build_policy(args.policy_type, checkpoint, device)

    se = StateEstimator(lc, use_cameras=False)
    control_dt = env_cfg_dict["control"]["decimation"] * env_cfg_dict["sim"]["dt"]

    command_profile = RCControllerProfile(
        dt=control_dt,
        state_estimator=se,
        x_scale=args.max_vel,
        y_scale=1.0,
        yaw_scale=args.max_yaw_vel,
        max_steps=args.max_steps,
    )

    hardware_agent = LCMAgent(env_cfg_dict, se, command_profile)
    se.spin()

    hardware_agent = HistoryWrapper(hardware_agent)

    root = REPO_ROOT / "logs"
    experiment_name = args.experiment_name or f"aliengo_{args.policy_type}_deploy"
    deployment_runner = DeploymentRunner(experiment_name=experiment_name, se=None, log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    deployment_runner.add_command_profile(command_profile)

    deployment_runner.run(max_steps=args.max_steps, logging=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Deploy Aliengo policies")
    parser.add_argument("--policy-type", choices=["amp", "residual"], default="amp", help="Policy to deploy.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to policy checkpoint.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for running the policy.")
    parser.add_argument("--max-vel", type=float, default=1.0, help="Scale for linear velocity commands.")
    parser.add_argument("--max-yaw-vel", type=float, default=1.0, help="Scale for yaw rate commands.")
    parser.add_argument("--max-steps", type=int, default=10000, help="Number of control steps to run.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Custom name for deployment logs.")

    args = parser.parse_args()

    load_and_run_policy(args)
