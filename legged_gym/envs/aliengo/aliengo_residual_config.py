# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Residual fine-tuning: base policy frozen, residual trained with task reward and KL to base.

from legged_gym.envs.aliengo.aliengo_amp_config import AliengoAMPCfg, MOTION_FILES
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class AliengoResidualCfg(AliengoAMPCfg):
    class commands(AliengoAMPCfg.commands):
        curriculum = False
        class ranges(AliengoAMPCfg.commands.ranges):
            # Фокус на высоких скоростях, где база слабеет.
            lin_vel_x = [1.1, 3.0]
            lin_vel_y = [-1.5, 1.5]
            ang_vel_yaw = [-3.14, 3.14]

    class rewards(AliengoAMPCfg.rewards):
        class scales(AliengoAMPCfg.rewards.scales):
            # Вернули исходные веса трекинга.
            tracking_lin_vel = 2.0 * 1.0 / (0.005 * 6)
            tracking_ang_vel = 0.8 * 1.0 / (0.005 * 6)

    class privileged_info(AliengoAMPCfg.privileged_info):
        enable_foot_contact = False


class AliengoResidualCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "ResidualOnPolicyRunner"

    class policy(LeggedRobotCfgPPO.policy):
        policy_class_name = "ActorCriticResidual"
        base_checkpoint = "{LEGGED_GYM_ROOT_DIR}/logs/aliengo_amp/video_limp/model_25000.pt"
        residual_hidden_dims = [256, 128]
        # Масштаб residual приведён к базовой политике, чтобы при необходимости
        # residual мог полностью компенсировать/заменить базу.
        residual_scale = 1.0

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.0
        base_kl_coef = 0.02
        # Исходный коэффициент L2 для residual.
        residual_l2_coef = 0.1
        num_learning_epochs = 5
        num_mini_batches = 4
        schedule = "fixed"

    class runner(LeggedRobotCfgPPO.runner):
        run_name = "residual"
        experiment_name = "aliengo_residual"
        algorithm_class_name = "ResidualPPO"
        policy_class_name = "ActorCriticResidual"
        max_iterations = 20000
        save_interval = 200
        # Тёплый старт: больше шума и слабее KL/L2 на начале.
        warmup_iters = 3000
        warmup_std = 0.35
        warmup_kl_coef = 0.01
        warmup_l2_coef = 0.05
        # start fresh residual training (base weights are loaded inside policy)
        load_run = -1
        checkpoint = -1
        resume = False
