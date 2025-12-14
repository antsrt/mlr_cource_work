# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Lightweight inference script that mirrors `play.py`, but replaces joystick
# control with a user-defined command generator. Edit `control_cmd` to shape
# the velocity commands you want to feed into the policy at each step.
#
# Дополнительная функциональность для вычисления метрики успешности копирования походки:
# 
# МЕТРИКА УСПЕШНОСТИ КОПИРОВАНИЯ ПОХОДКИ:
# Этот скрипт позволяет записывать положения сочленений ног робота при разных скоростях
# и сравнивать походки базовой и дообученной моделей. Данные нормализуются по времени,
# чтобы компенсировать разницу в частоте движения при разных скоростях.
#
# Примеры использования:
# 1. Записать походку базовой модели со скоростью 0.8 м/с:
#    python play_manual_copy.py --task <task_name> --load_run <run_name> --save_gait --gait_velocity 0.8
#
# 2. Записать походку дообученной модели со скоростью 1.4 м/с:
#    python play_manual_copy.py --task <task_name> --load_run <run_name> --save_gait --gait_velocity 1.4
#
# 3. Сравнить две записи походки:
#    python play_manual_copy.py --compare_gaits --gait_file1 gait_data/gait_vel_0.80.pkl --gait_file2 gait_data/gait_vel_1.40.pkl
#
# Результаты сравнения сохраняются в директории gait_comparison/:
#   - gait_comparison_normalized.png: графики нормализованных данных (для сравнения формы походки)
#   - gait_comparison_original.png: графики исходных данных (для визуализации временных рядов)
#   - gait_metrics.pkl: метрики сравнения (MSE, RMSE, MAE, корреляция, косинусное сходство)
#
# Вычисляемые метрики:
#   - MSE (Mean Squared Error): средний квадрат ошибки
#   - RMSE (Root Mean Squared Error): корень из среднего квадрата ошибки
#   - MAE (Mean Absolute Error): средняя абсолютная ошибка
#   - Корреляция: коэффициент корреляции Пирсона (по каждому сочленению)
#   - Косинусное сходство: мера схожести направлений векторов
#   - RMSE по каждому сочленению отдельно

import os
import sys
import pickle
from typing import Tuple, Dict, Optional
from pathlib import Path

import isaacgym
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import interpolate

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def control_cmd(step: int, max_steps: int, target_velocity: float = 1.0) -> Tuple[float, float, float]:
    """
    User-editable command generator.
    Args:
        step: current simulation step (0-based).
        max_steps: total planned steps for this rollout.
        target_velocity: целевая скорость движения (м/с).
    Returns:
        (lin_vel_x, lin_vel_y, ang_vel_yaw)
    """
    # Constant forward command with zero yaw.
    lin_x = target_velocity  # m/s
    lin_y = 0.0  # m/s
    yaw_rate = 0.0  # rad/s
    return lin_x, lin_y, yaw_rate

# lin vel x: -0.211 .. 1.451 м/с фактически максимум 1.1 м/с
# lin vel y: -0.421 .. 0.806 м/с фактически максимум 2.0 м/с точно, больше не тестил
# ang vel z: -1.818 .. 1.197 рад/с при 0.6 идет прямо


def save_gait_data(logger, velocity: float, output_dir: str = "gait_data") -> str:
    """
    Сохраняет данные о походке (положения сочленений) в файл.
    
    Args:
        logger: Logger объект с записанными данными
        velocity: Скорость, с которой была записана походка
        output_dir: Директория для сохранения файлов
    
    Returns:
        Путь к сохраненному файлу
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Извлекаем данные о положениях сочленений
    dof_pos_data = logger.state_log.get('dof_pos', [])
    dof_names = logger.state_log.get('dof_names_sel', [])
    
    if len(dof_pos_data) == 0:
        raise ValueError("Нет данных о положениях сочленений для сохранения")
    
    # Преобразуем в numpy массив
    dof_pos_array = np.array(dof_pos_data)
    
    # Создаем временную ось
    dt = logger.dt
    time_array = np.arange(len(dof_pos_data)) * dt
    
    # Сохраняем метаданные
    gait_data = {
        'dof_positions': dof_pos_array,
        'time': time_array,
        'dt': dt,
        'velocity': velocity,
        'dof_names': dof_names[0] if len(dof_names) > 0 else [],
        'num_steps': len(dof_pos_data)
    }
    
    # Имя файла с указанием скорости
    filename = f"gait_vel_{velocity:.2f}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(gait_data, f)
    
    print(f"Данные походки сохранены: {filepath}")
    print(f"  Скорость: {velocity} м/с")
    print(f"  Количество шагов: {len(dof_pos_data)}")
    print(f"  Время записи: {time_array[-1]:.2f} с")
    
    return filepath


def load_gait_data(filepath: str) -> Dict:
    """
    Загружает данные о походке из файла.
    
    Args:
        filepath: Путь к файлу с данными
    
    Returns:
        Словарь с данными походки
    """
    with open(filepath, 'rb') as f:
        gait_data = pickle.load(f)
    return gait_data


def normalize_time_series(data1: np.ndarray, time1: np.ndarray, 
                          data2: np.ndarray, time2: np.ndarray,
                          method: str = 'interpolation') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Нормализует две временные последовательности для сравнения.
    Нормализует по времени, чтобы компенсировать разницу в частоте движения
    при разных скоростях.
    
    Args:
        data1: Данные первой последовательности (num_steps, num_joints)
        time1: Временная ось первой последовательности
        data2: Данные второй последовательности (num_steps, num_joints)
        time2: Временная ось второй последовательности
        method: Метод нормализации ('interpolation' или 'phase')
    
    Returns:
        (normalized_data1, normalized_data2, normalized_time)
    """
    if method == 'interpolation':
        # Нормализуем по длине более короткой последовательности
        min_length = min(len(data1), len(data2))
        normalized_time = np.linspace(0, 1, min_length)
        
        # Нормализуем временные оси исходных данных к [0, 1]
        # Это позволяет сравнивать походки с разной частотой
        if len(time1) > 1 and time1[-1] > time1[0]:
            time1_norm = (time1 - time1[0]) / (time1[-1] - time1[0])
        else:
            time1_norm = np.linspace(0, 1, len(time1)) if len(time1) > 1 else np.array([0.0])
        
        if len(time2) > 1 and time2[-1] > time2[0]:
            time2_norm = (time2 - time2[0]) / (time2[-1] - time2[0])
        else:
            time2_norm = np.linspace(0, 1, len(time2)) if len(time2) > 1 else np.array([0.0])
        
        # Интерполируем данные на нормализованную временную ось
        normalized_data1 = np.zeros((min_length, data1.shape[1]))
        normalized_data2 = np.zeros((min_length, data2.shape[1]))
        
        for joint_idx in range(data1.shape[1]):
            if len(time1_norm) > 1 and len(data1) > 1:
                try:
                    interp_func1 = interpolate.interp1d(time1_norm, data1[:, joint_idx], 
                                                       kind='linear', 
                                                       bounds_error=False,
                                                       fill_value=(data1[0, joint_idx], data1[-1, joint_idx]))
                    normalized_data1[:, joint_idx] = interp_func1(normalized_time)
                except Exception:
                    # Fallback: простое масштабирование
                    normalized_data1[:, joint_idx] = np.interp(normalized_time, time1_norm, data1[:, joint_idx])
            else:
                normalized_data1[:, joint_idx] = data1[0, joint_idx] if len(data1) > 0 else 0.0
            
            if len(time2_norm) > 1 and len(data2) > 1:
                try:
                    interp_func2 = interpolate.interp1d(time2_norm, data2[:, joint_idx], 
                                                       kind='linear',
                                                       bounds_error=False,
                                                       fill_value=(data2[0, joint_idx], data2[-1, joint_idx]))
                    normalized_data2[:, joint_idx] = interp_func2(normalized_time)
                except Exception:
                    # Fallback: простое масштабирование
                    normalized_data2[:, joint_idx] = np.interp(normalized_time, time2_norm, data2[:, joint_idx])
            else:
                normalized_data2[:, joint_idx] = data2[0, joint_idx] if len(data2) > 0 else 0.0
        
        return normalized_data1, normalized_data2, normalized_time
    
    elif method == 'phase':
        # Нормализация по фазе (по количеству шагов)
        min_length = min(len(data1), len(data2))
        normalized_time = np.linspace(0, 1, min_length)
        
        # Просто обрезаем до минимальной длины
        normalized_data1 = data1[:min_length]
        normalized_data2 = data2[:min_length]
        
        return normalized_data1, normalized_data2, normalized_time
    
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")


def compute_gait_similarity_metrics(data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет метрики схожести двух походок.
    
    Args:
        data1: Нормализованные данные первой походки (num_steps, num_joints)
        data2: Нормализованные данные второй походки (num_steps, num_joints)
    
    Returns:
        Словарь с метриками
    """
    metrics = {}
    
    # 1. Mean Squared Error (MSE)
    mse = np.mean((data1 - data2) ** 2)
    metrics['mse'] = float(mse)
    
    # 2. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    metrics['rmse'] = float(rmse)
    
    # 3. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(data1 - data2))
    metrics['mae'] = float(mae)
    
    # 4. Корреляция (по каждому сочленению, затем среднее)
    correlations = []
    for joint_idx in range(data1.shape[1]):
        if np.std(data1[:, joint_idx]) > 1e-6 and np.std(data2[:, joint_idx]) > 1e-6:
            corr = np.corrcoef(data1[:, joint_idx], data2[:, joint_idx])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    metrics['mean_correlation'] = float(np.mean(correlations)) if correlations else 0.0
    metrics['min_correlation'] = float(np.min(correlations)) if correlations else 0.0
    metrics['max_correlation'] = float(np.max(correlations)) if correlations else 0.0
    
    # 5. Косинусное сходство (по каждому сочленению, затем среднее)
    cosine_similarities = []
    for joint_idx in range(data1.shape[1]):
        vec1 = data1[:, joint_idx]
        vec2 = data2[:, joint_idx]
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 > 1e-6 and norm2 > 1e-6:
            cosine_sim = dot_product / (norm1 * norm2)
            cosine_similarities.append(cosine_sim)
    
    metrics['mean_cosine_similarity'] = float(np.mean(cosine_similarities)) if cosine_similarities else 0.0
    
    # 6. Метрика по каждому сочленению отдельно
    per_joint_rmse = np.sqrt(np.mean((data1 - data2) ** 2, axis=0))
    metrics['per_joint_rmse'] = per_joint_rmse.tolist()
    
    return metrics


def compare_gaits(filepath1: str, filepath2: str, 
                  normalize_method: str = 'interpolation',
                  plot_comparison: bool = True,
                  output_dir: str = "gait_comparison") -> Dict:
    """
    Сравнивает две записи походки и вычисляет метрики.
    
    Args:
        filepath1: Путь к первой записи (базовая модель)
        filepath2: Путь ко второй записи (дообученная модель)
        normalize_method: Метод нормализации ('interpolation' или 'phase')
        plot_comparison: Создавать ли графики сравнения
        output_dir: Директория для сохранения результатов
    
    Returns:
        Словарь с метриками сравнения
    """
    # Загружаем данные
    gait1 = load_gait_data(filepath1)
    gait2 = load_gait_data(filepath2)
    
    print(f"\nСравнение походок:")
    print(f"  Базовая модель (vel={gait1['velocity']:.2f} м/с): {filepath1}")
    print(f"  Дообученная модель (vel={gait2['velocity']:.2f} м/с): {filepath2}")
    
    # Нормализуем данные
    data1_norm, data2_norm, time_norm = normalize_time_series(
        gait1['dof_positions'], gait1['time'],
        gait2['dof_positions'], gait2['time'],
        method=normalize_method
    )
    
    # Вычисляем метрики
    metrics = compute_gait_similarity_metrics(data1_norm, data2_norm)
    
    # Добавляем информацию о загруженных данных
    metrics['gait1_velocity'] = gait1['velocity']
    metrics['gait2_velocity'] = gait2['velocity']
    metrics['gait1_dof_names'] = gait1['dof_names']
    metrics['gait2_dof_names'] = gait2['dof_names']
    
    # Выводим результаты
    print(f"\nМетрики схожести походки:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Средняя корреляция: {metrics['mean_correlation']:.4f}")
    print(f"  Среднее косинусное сходство: {metrics['mean_cosine_similarity']:.4f}")
    
    if len(metrics['per_joint_rmse']) > 0:
        print(f"\n  RMSE по сочленениям:")
        dof_names = gait1['dof_names'] if len(gait1['dof_names']) > 0 else [f"joint_{i}" for i in range(len(metrics['per_joint_rmse']))]
        for name, rmse in zip(dof_names, metrics['per_joint_rmse']):
            print(f"    {name}: {rmse:.6f}")
    
    # Создаем графики сравнения
    if plot_comparison:
        os.makedirs(output_dir, exist_ok=True)
        plot_gait_comparison(gait1, gait2, data1_norm, data2_norm, time_norm, 
                           metrics, output_dir)
    
    # Сохраняем метрики
    metrics_filepath = os.path.join(output_dir, "gait_metrics.pkl")
    with open(metrics_filepath, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"\nМетрики сохранены: {metrics_filepath}")
    
    return metrics


def plot_gait_comparison(gait1: Dict, gait2: Dict,
                         data1_norm: np.ndarray, data2_norm: np.ndarray,
                         time_norm: np.ndarray, metrics: Dict,
                         output_dir: str):
    """
    Создает графики сравнения двух походок.
    """
    dof_names = gait1['dof_names'] if len(gait1['dof_names']) > 0 else \
                [f"joint_{i}" for i in range(data1_norm.shape[1])]
    
    num_joints = data1_norm.shape[1]
    n_cols = 2
    n_rows = (num_joints + n_cols - 1) // n_cols
    
    # График 1: Нормализованные данные (сравнение)
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if num_joints == 1:
        axes1 = [axes1]
    else:
        axes1 = axes1.flatten()
    
    for joint_idx in range(num_joints):
        ax = axes1[joint_idx]
        ax.plot(time_norm, data1_norm[:, joint_idx], 'b-', label=f'Базовая (vel={gait1["velocity"]:.2f})', linewidth=2)
        ax.plot(time_norm, data2_norm[:, joint_idx], 'r--', label=f'Дообученная (vel={gait2["velocity"]:.2f})', linewidth=2)
        ax.set_xlabel('Нормализованное время')
        ax.set_ylabel('Положение [рад]')
        ax.set_title(f'{dof_names[joint_idx]} (RMSE: {metrics["per_joint_rmse"][joint_idx]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Скрываем лишние subplots
    for idx in range(num_joints, len(axes1)):
        axes1[idx].set_visible(False)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, "gait_comparison_normalized.png")
    plt.savefig(comparison_path, dpi=150)
    print(f"График сравнения сохранен: {comparison_path}")
    plt.close()
    
    # График 2: Исходные данные (по времени)
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if num_joints == 1:
        axes2 = [axes2]
    else:
        axes2 = axes2.flatten()
    
    for joint_idx in range(num_joints):
        ax = axes2[joint_idx]
        ax.plot(gait1['time'], gait1['dof_positions'][:, joint_idx], 'b-', 
               label=f'Базовая (vel={gait1["velocity"]:.2f})', linewidth=2, alpha=0.7)
        ax.plot(gait2['time'], gait2['dof_positions'][:, joint_idx], 'r--', 
               label=f'Дообученная (vel={gait2["velocity"]:.2f})', linewidth=2, alpha=0.7)
        ax.set_xlabel('Время [с]')
        ax.set_ylabel('Положение [рад]')
        ax.set_title(f'{dof_names[joint_idx]} (исходные данные)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for idx in range(num_joints, len(axes2)):
        axes2[idx].set_visible(False)
    
    plt.tight_layout()
    original_path = os.path.join(output_dir, "gait_comparison_original.png")
    plt.savefig(original_path, dpi=150)
    print(f"График исходных данных сохранен: {original_path}")
    plt.close()

def play_manual(args, save_gait: bool = False, gait_velocity: float = 1.0, 
                do_compare_gaits: bool = False, gait_file1: Optional[str] = None, 
                gait_file2: Optional[str] = None):
    """
    Запускает симуляцию с ручным управлением.
    
    Args:
        args: Аргументы командной строки
        save_gait: Сохранять ли данные о походке
        gait_velocity: Скорость для сохранения (используется в имени файла)
        compare_gaits: Сравнивать ли две записи походки
        gait_file1: Путь к первой записи для сравнения
        gait_file2: Путь ко второй записи для сравнения
    """
    # Если нужно только сравнить записи
    if do_compare_gaits and gait_file1 and gait_file2:
        metrics = compare_gaits(gait_file1, gait_file2)
        return metrics
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override for deterministic single-env inference.
    env_cfg.env.num_envs = 1
    env_cfg.env.get_commands_from_joystick = False
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_gains = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.commands.resampling_time = 1e9  # effectively disable random resampling

    # AMP data preload kept minimal for inference.
    train_cfg.runner.amp_num_preload_transitions = 1
    train_cfg.runner.resume = True  # load policy weights via args/load_run
    # Prevent CLI default ("debug") from overwriting runner class in cfg.
    args.runner_class_name = None

    # Prepare environment and policy.
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # Disable internal random command resampling to keep full manual control.
    def _noop_resample(env_ids):
        return
    env._resample_commands = _noop_resample  # type: ignore

    obs_dict = env.reset()

    # Логгер для визуализации графиков.
    from legged_gym.utils import Logger
    logger = Logger(env.dt)
    selected_joint_names = [
        "FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint",
        "FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint",
    ]
    dof_names = list(getattr(env, 'dof_names', []))
    selected_dof_indices = []
    missing = []
    for n in selected_joint_names:
        if n in dof_names:
            selected_dof_indices.append(dof_names.index(n))
        else:
            missing.append(n)
    if len(missing) > 0:
        print("Logger selection warning, some requested joints not found in env.dof_names:", missing)
    logger.log_state('dof_names_sel', selected_joint_names)
    robot_index = 0
    stop_state_log = 1000
    stop_rew_log = env.max_episode_length + 1

    # Manual residual hook: set this to shape (num_envs, num_actions).
    env.action_residual[:] = 0.0

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)

    policy = ppo_runner.get_inference_policy(device=env.device)

    max_steps = int(1 * env.max_episode_length)
    # Align initial observations with our commands before the first policy call.
    lin_x, lin_y, yaw_rate = control_cmd(0, env.max_episode_length, gait_velocity)
    env.commands[:, 0] = lin_x
    env.commands[:, 1] = lin_y
    env.commands[:, 2] = yaw_rate
    env.compute_observations()
    obs_dict = env.get_observations()
    try:
        for i in range(10 * int(env.max_episode_length)):
            ep_step = int(env.episode_length_buf[0].item())
            lin_x, lin_y, yaw_rate = control_cmd(ep_step, env.max_episode_length, gait_velocity)
            env.commands[:, 0] = lin_x
            env.commands[:, 1] = lin_y
            env.commands[:, 2] = yaw_rate
            # env.action_residual[:, :] = torch.tensor([...], device=env.device)
            env.compute_observations()
            obs_dict = env.get_observations()

            actions = policy(obs_dict)
            obs_dict, rews, dones, infos, _, _ = env.step(actions.detach())

            # Логгирование данных для графиков
            try:
                full_tgt = (actions[robot_index].detach().cpu().numpy() * env.cfg.control.action_scale)
            except Exception:
                full_tgt = (actions[robot_index].cpu().numpy() * env.cfg.control.action_scale)
            try:
                full_pos = env.dof_pos[robot_index].detach().cpu().numpy() if hasattr(env.dof_pos[robot_index], 'detach') else env.dof_pos[robot_index].cpu().numpy()
            except Exception:
                full_pos = np.array(env.dof_pos[robot_index])
            try:
                full_vel = env.dof_vel[robot_index].detach().cpu().numpy() if hasattr(env.dof_vel[robot_index], 'detach') else env.dof_vel[robot_index].cpu().numpy()
            except Exception:
                full_vel = np.array(env.dof_vel[robot_index])
            try:
                full_torque = env.torques[robot_index].detach().cpu().numpy() if hasattr(env.torques[robot_index], 'detach') else env.torques[robot_index].cpu().numpy()
            except Exception:
                full_torque = np.array(env.torques[robot_index])

            sel_tgt = np.array([full_tgt[idx] for idx in selected_dof_indices]) if len(selected_dof_indices) > 0 else np.array([])
            sel_pos = np.array([full_pos[idx] for idx in selected_dof_indices]) if len(selected_dof_indices) > 0 else np.array([])
            sel_vel = np.array([full_vel[idx] for idx in selected_dof_indices]) if len(selected_dof_indices) > 0 else np.array([])
            sel_torque = np.array([full_torque[idx] for idx in selected_dof_indices]) if len(selected_dof_indices) > 0 else np.array([])

            logger.log_states(
                {
                    'dof_pos_target': sel_tgt,
                    'dof_pos': sel_pos,
                    'dof_vel': sel_vel,
                    'dof_torque': sel_torque,
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
            if i == stop_state_log:
                logger.plot_states()
            if 0 < i < stop_rew_log:
                if infos.get("episode"):
                    num_episodes = torch.sum(env.reset_buf).item()
                    if num_episodes > 0:
                        logger.log_rewards(infos["episode"], num_episodes)
            elif i == stop_rew_log:
                logger.print_rewards()
    
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected, saving gait data...")
    finally:
        if save_gait:
            save_gait_data(logger, gait_velocity)


if __name__ == "__main__":
    args = get_args()
    
    # Парсим дополнительные аргументы для работы с метриками походки
    # Используем простой парсинг из sys.argv, чтобы не конфликтовать с get_args()
    save_gait = '--save_gait' in sys.argv
    do_compare_gaits = '--compare_gaits' in sys.argv
    
    gait_velocity = 1.0
    if '--gait_velocity' in sys.argv:
        idx = sys.argv.index('--gait_velocity')
        if idx + 1 < len(sys.argv):
            try:
                gait_velocity = float(sys.argv[idx + 1])
            except ValueError:
                print("Предупреждение: неверное значение для --gait_velocity, используется 1.0")
    
    gait_file1 = None
    if '--gait_file1' in sys.argv:
        idx = sys.argv.index('--gait_file1')
        if idx + 1 < len(sys.argv):
            gait_file1 = sys.argv[idx + 1]
    
    gait_file2 = None
    if '--gait_file2' in sys.argv:
        idx = sys.argv.index('--gait_file2')
        if idx + 1 < len(sys.argv):
            gait_file2 = sys.argv[idx + 1]
    
    play_manual(
        args, 
        save_gait=save_gait,
        gait_velocity=gait_velocity,
        do_compare_gaits=do_compare_gaits,
        gait_file1=gait_file1,
        gait_file2=gait_file2
    )
