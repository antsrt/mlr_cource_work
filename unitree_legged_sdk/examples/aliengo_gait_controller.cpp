#include "unitree_legged_sdk/unitree_legged_sdk.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <unistd.h>

using namespace UNITREE_LEGGED_SDK;

namespace {

constexpr double kDt = 0.001;                    // 1 kHz control loop
constexpr double kStandDuration = 3.0;           // seconds spent interpolating to the stand pose
constexpr double kWaitDuration = 5.0;            // seconds to wait before starting gait
constexpr int kWalkSteps = 15000;                // 15 seconds of walking at 1 kHz
constexpr uint16_t kTargetPort = 8007;
constexpr uint16_t kLocalPort = 8082;
constexpr int kLowCmdLength = 610;
constexpr int kLowStateLength = 771;

constexpr double kThighOffset = 0.083;
constexpr double kLegOffsetX = 0.2407;
constexpr double kLegOffsetY = 0.051;
constexpr double kThighLength = 0.25;
constexpr double kCalfLength = 0.25;
constexpr double kFootRad = std::sqrt(kLegOffsetX * kLegOffsetX
                                      + std::pow(kLegOffsetY + kThighOffset, 2.0));

constexpr std::array<double, 12> kLowerJointLimits = {
    -0.773, -0.424, -2.675,
    -0.773, -0.424, -2.675,
    -0.773, -0.424, -2.675,
    -0.773, -0.424, -2.675
};

constexpr std::array<double, 12> kUpperJointLimits = {
     0.947,  3.827, -0.711,
     0.947,  3.827, -0.711,
     0.947,  3.827, -0.711,
     0.947,  3.827, -0.711
};

struct Vec3 {
  double x;
  double y;
  double z;
};

constexpr Vec3 OffsetVec(const Vec3& v, double dz) {
  return Vec3{v.x, v.y, v.z + dz};
}

constexpr std::array<Vec3, 4> OffsetFeet(const std::array<Vec3, 4>& feet,
                                         double dz) {
  return {OffsetVec(feet[0], dz),
          OffsetVec(feet[1], dz),
          OffsetVec(feet[2], dz),
          OffsetVec(feet[3], dz)};
}

Vec3 LerpVec(const Vec3& start, const Vec3& end, double alpha) {
  return {
      start.x + (end.x - start.x) * alpha,
      start.y + (end.y - start.y) * alpha,
      start.z + (end.z - start.z) * alpha};
}

std::array<Vec3, 4> InterpolateFeet(const std::array<Vec3, 4>& start,
                                    const std::array<Vec3, 4>& end,
                                    double alpha) {
  std::array<Vec3, 4> feet{};
  for (int i = 0; i < 4; ++i) {
    feet[i] = LerpVec(start[i], end[i], alpha);
  }
  return feet;
}

enum class Leg : int {
  FR = 0,
  FL = 1,
  RR = 2,
  RL = 3
};

const std::array<Vec3, 4> kStandingFeet = {{
    {0.2407, -0.1378, -0.4},   // FR
    {0.2407,  0.1378, -0.4},   // FL
    {-0.2407, -0.1378, -0.4},  // RR
    {-0.2407,  0.1378, -0.4}   // RL
}};

constexpr double kStandHoldDz = 0.05;
constexpr std::array<Vec3, 4> kStandHoldFeet = OffsetFeet(kStandingFeet, kStandHoldDz);

struct GaitParams {
  double period{0.5};
  double r{0.5};
  double swing_h{0.09};
  double dbody_h{-0.05};
  std::array<double, 4> bias{0.0, 0.5, 0.5, 0.0};
};

constexpr std::array<double, 3> kWalkKpLeg = {50.0, 50.0, 50.0};
constexpr std::array<double, 3> kWalkKdLeg = {2.0, 2.0, 2.0};
constexpr double kStandGainScale = 0.25;

double wrap01(double value) {
  double wrapped = std::fmod(value, 1.0);
  if (wrapped < 0.0) {
    wrapped += 1.0;
  }
  return wrapped;
}

std::pair<double, bool> gaitPhase(double phase, double bias, double r) {
  const double time_fraction = wrap01(phase / (2.0 * M_PI) + 0.5 + bias);
  const bool contact = time_fraction < r;
  const double phi = contact ? time_fraction / r : (time_fraction - r) / (1.0 - r);
  return {phi, contact};
}

Vec3 endSwingPos(Leg leg,
                 double forward_vel,
                 double turn_rate,
                 double t_swing,
                 const std::array<double, 8>& foot_deltas) {
  const Vec3& stand = kStandingFeet[static_cast<int>(leg)];
  const double dtheta = turn_rate * t_swing;
  const double angle = std::atan2(stand.y, stand.x) + dtheta;
  const double dx_turn = kFootRad * std::cos(angle) - stand.x;
  const double dy_turn = kFootRad * std::sin(angle) - stand.y;
  const int idx = static_cast<int>(leg) * 2;
  const double dx_ff = foot_deltas[idx];
  const double dy_ff = foot_deltas[idx + 1];
  return {
      forward_vel * t_swing + dx_turn + dx_ff,
      dy_turn + dy_ff,
      0.0};
}

Vec3 startSwingPos(Leg leg,
                   double forward_vel,
                   double turn_rate,
                   double t_swing) {
  const auto end_pos = endSwingPos(leg, forward_vel, turn_rate, t_swing,
                                   std::array<double, 8>{});
  return {-end_pos.x, -end_pos.y, -end_pos.z};
}

double cycloid(double phi, double start, double end) {
  const double delta = end - start;
  return delta / (2.0 * M_PI) * (2.0 * M_PI * phi - std::sin(2.0 * M_PI * phi)) + start;
}

Vec3 footPosSwing(double swing_h,
                  double phi,
                  const Vec3& start_pos,
                  const Vec3& end_pos) {
  return {
      cycloid(phi, start_pos.x, end_pos.x),
      cycloid(phi, start_pos.y, end_pos.y),
      0.5 * swing_h * (1.0 - std::cos(2.0 * M_PI * phi))
  };
}

Vec3 footPosContact(double phi,
                    const Vec3& start_pos,
                    const Vec3& end_pos) {
  return {
      cycloid(phi, start_pos.x, end_pos.x),
      cycloid(phi, start_pos.y, end_pos.y),
      0.0
  };
}

Vec3 forwardKinematics(Leg leg, const Vec3& q) {
  const bool front_leg = (leg == Leg::FR || leg == Leg::FL);
  const bool right_leg = (leg == Leg::FR || leg == Leg::RR);
  const double side_sign = right_leg ? -1.0 : 1.0;

  const double l1 = side_sign * kThighOffset;
  const double l2 = -kThighLength;
  const double l3 = -kCalfLength;

  const double s1 = std::sin(q.x);
  const double s2 = std::sin(q.y);
  const double s3 = std::sin(q.z);
  const double c1 = std::cos(q.x);
  const double c2 = std::cos(q.y);
  const double c3 = std::cos(q.z);

  const double c23 = c2 * c3 - s2 * s3;
  const double s23 = s2 * c3 + c2 * s3;

  const double p0_hip = l3 * s23 + l2 * s2;
  const double p1_hip = -l3 * s1 * c23 + l1 * c1 - l2 * c2 * s1;
  const double p2_hip = l3 * c1 * c23 + l1 * s1 + l2 * c1 * c2;

  const double px = p0_hip + (front_leg ? kLegOffsetX : -kLegOffsetX);
  const double py = p1_hip + (right_leg ? -kLegOffsetY : kLegOffsetY);
  const double pz = p2_hip;

  return {px, py, pz};
}

std::array<Vec3, 4> forwardKinematicsAllLegs(const std::array<double, 12>& q) {
  std::array<Vec3, 4> feet{};
  for (int leg = 0; leg < 4; ++leg) {
    const Vec3 joint{q[leg * 3 + 0], q[leg * 3 + 1], q[leg * 3 + 2]};
    feet[leg] = forwardKinematics(static_cast<Leg>(leg), joint);
  }
  return feet;
}

Vec3 inverseKinematics(Leg leg, const Vec3& p) {
  const bool front_leg = (leg == Leg::FR || leg == Leg::FL);
  const bool right_leg = (leg == Leg::FR || leg == Leg::RR);

  const double fx = front_leg ? kLegOffsetX : -kLegOffsetX;
  const double fy = right_leg ? -kLegOffsetY : kLegOffsetY;
  const double px = p.x - fx;
  const double py = p.y - fy;
  const double pz = p.z;

  const double b2y = right_leg ? -kThighOffset : kThighOffset;
  const double b3z = -kThighLength;
  const double b4z = -kCalfLength;

  const double hip_offset = kThighOffset;
  const double c = std::sqrt(px * px + py * py + pz * pz);
  const double b = std::sqrt(std::max(c * c - hip_offset * hip_offset, 1e-9));

  const double L_sq = std::max(py * py + pz * pz - b2y * b2y, 1e-9);
  const double L = std::sqrt(L_sq);
  double q1 = std::atan2(pz * b2y + py * L, py * b2y - pz * L);

  double temp = (b3z * b3z + b4z * b4z - b * b) / (2.0 * std::abs(b3z * b4z));
  const double clip_hi = std::cos(M_PI + kUpperJointLimits[2]);
  const double clip_lo = std::cos(M_PI + kLowerJointLimits[2]);
  temp = std::clamp(temp, clip_hi, clip_lo);
  double q3 = std::acos(temp);
  q3 = -(M_PI - q3);

  const double a1 = py * std::sin(q1) - pz * std::cos(q1);
  const double a2 = px;
  const double m1 = b4z * std::sin(q3);
  const double m2 = b3z + b4z * std::cos(q3);
  double q2 = std::atan2(m1 * a1 + m2 * a2, m1 * a2 - m2 * a1);

  return {q1, q2, q3};
}

std::array<double, 12> inverseKinematicsAllLegs(const std::array<Vec3, 4>& feet) {
  std::array<double, 12> q{};
  for (int leg = 0; leg < 4; ++leg) {
    const Vec3 angles = inverseKinematics(static_cast<Leg>(leg), feet[leg]);
    q[leg * 3 + 0] = angles.x;
    q[leg * 3 + 1] = angles.y;
    q[leg * 3 + 2] = angles.z;
  }
  return q;
}

void clampJointLimits(std::array<double, 12>& q) {
  for (int i = 0; i < 12; ++i) {
    q[i] = std::clamp(q[i], kLowerJointLimits[i], kUpperJointLimits[i]);
  }
}

std::array<double, 12> makeAllJointGains(const std::array<double, 3>& leg_gains,
                                         double scale) {
  std::array<double, 12> gains{};
  for (int leg = 0; leg < 4; ++leg) {
    for (int j = 0; j < 3; ++j) {
      gains[leg * 3 + j] = leg_gains[j] * scale;
    }
  }
  return gains;
}

std::array<double, 12> lerpArray(const std::array<double, 12>& start,
                                 const std::array<double, 12>& end,
                                 double alpha) {
  std::array<double, 12> out{};
  for (int i = 0; i < 12; ++i) {
    out[i] = start[i] + (end[i] - start[i]) * alpha;
  }
  return out;
}

}  // namespace

class AliengoGaitController {
 public:
  AliengoGaitController()
      : stand_steps_(static_cast<int>(kStandDuration / kDt)),
        wait_steps_(static_cast<int>(kWaitDuration / kDt)),
        walk_start_step_(stand_steps_ + wait_steps_),
        walk_end_step_(walk_start_step_ + kWalkSteps),
        udp_(kLocalPort, "192.168.123.10", kTargetPort,
             kLowCmdLength, kLowStateLength),
        stand_target_feet_(kStandHoldFeet),
        standup_start_feet_(kStandHoldFeet),
        stand_pose_q_(inverseKinematicsAllLegs(kStandHoldFeet)),
        walk_kp_(makeAllJointGains(kWalkKpLeg, 1.0)),
        walk_kd_(makeAllJointGains(kWalkKdLeg, 1.0)),
        stand_kp_start_(makeAllJointGains(kWalkKpLeg, kStandGainScale)),
        stand_kd_start_(makeAllJointGains(kWalkKdLeg, kStandGainScale)) {
    udp_.InitCmdData(cmd_);
    cmd_.levelFlag = LOWLEVEL;
    for (int i = 0; i < 12; ++i) {
      cmd_.motorCmd[i].mode = 0x0A;
      cmd_.motorCmd[i].q = PosStopF;
      cmd_.motorCmd[i].dq = 0.0;
      cmd_.motorCmd[i].Kp = 0.0;
      cmd_.motorCmd[i].Kd = 0.0;
      cmd_.motorCmd[i].tau = 0.0;
    }
    udp_.SetSend(cmd_);

    log_.open("aliengo_log.csv", std::ios::out | std::ios::trunc);
    if (!log_) {
      std::cerr << "Failed to open aliengo_log.csv for writing.\n";
    } else {
      log_ << "timestep_ms";
      constexpr std::array<const char*, 4> prefixes = {"q_des_", "q_", "qd_", "tau_"};
      const std::array<std::string, 12> joint_names = {
          "FR0", "FR1", "FR2", "FL0", "FL1", "FL2",
          "RR0", "RR1", "RR2", "RL0", "RL1", "RL2"
      };
      for (const char* prefix : prefixes) {
        for (const auto& name : joint_names) {
          log_ << ',' << prefix << name;
        }
      }
      log_ << '\n';
    }

    clampJointLimits(stand_pose_q_);
    last_q_des_ = stand_pose_q_;
    current_contact_.fill(true);
    current_leg_phase_.fill(0.0);
  }

  void UDPRecv() { udp_.Recv(); }
  void UDPSend() { udp_.Send(); }

  void RobotControl() {
    udp_.GetRecv(state_);

    if (!stateReady()) {
      return;
    }

    if (!have_init_) {
      for (int i = 0; i < 12; ++i) {
        init_q_[i] = state_.motorState[i].q;
      }
      last_q_des_ = init_q_;
      standup_start_feet_ = forwardKinematicsAllLegs(init_q_);
      have_init_ = true;
      applyCommand(init_q_, stand_kp_start_, stand_kd_start_);
      return;
    }

    std::array<double, 12> q_des{};
    std::array<double, 12> kp_cmd{};
    std::array<double, 12> kd_cmd{};

    if (motion_time_ < stand_steps_) {
      const double alpha = std::clamp(
          static_cast<double>(motion_time_) / std::max(1, stand_steps_), 0.0, 1.0);
      auto feet = InterpolateFeet(standup_start_feet_, stand_target_feet_, alpha);
      q_des = inverseKinematicsAllLegs(feet);
      clampJointLimits(q_des);
      kp_cmd = lerpArray(stand_kp_start_, walk_kp_, alpha);
      kd_cmd = lerpArray(stand_kd_start_, walk_kd_, alpha);
      current_contact_.fill(true);
      current_leg_phase_.fill(0.0);
    } else if (motion_time_ < walk_start_step_) {
      q_des = stand_pose_q_;
      kp_cmd = walk_kp_;
      kd_cmd = walk_kd_;
      current_contact_.fill(true);
      current_leg_phase_.fill(0.0);
    } else if (motion_time_ < walk_end_step_) {
      const auto traj = computeJointTargets();
      q_des = traj.q;
      kp_cmd = walk_kp_;
      kd_cmd = walk_kd_;
      current_contact_ = traj.contact;
      current_leg_phase_ = traj.leg_phase;
      if (!log_active_) {
        log_active_ = true;
      }
      phase_ += 2.0 * M_PI * kDt / gait_params_.period;
      if (phase_ > 2.0 * M_PI) {
        phase_ -= 2.0 * M_PI;
      }
    } else {
      q_des = stand_pose_q_;
      kp_cmd = walk_kp_;
      kd_cmd = walk_kd_;
      current_contact_.fill(true);
      current_leg_phase_.fill(0.0);
      if (!finished_) {
        finished_ = true;
        log_active_ = false;
        std::cout << "Max steps reached. Stopping gait and finishing log.\n";
        if (log_) {
          log_.flush();
        }
      }
    }

    last_q_des_ = q_des;
    applyCommand(q_des, kp_cmd, kd_cmd);

    const double log_timestamp_ms =
        (motion_time_ - walk_start_step_) * kDt * 1000.0;
    if (log_active_ && log_) {
      logSample(std::max(0.0, log_timestamp_ms), q_des);
    }

    ++motion_time_;
  }

 private:
  struct JointTrajectory {
    std::array<double, 12> q;
    std::array<bool, 4> contact;
    std::array<double, 4> leg_phase;
  };

  JointTrajectory computeJointTargets() {
    const double t_swing = (1.0 - gait_params_.r) * gait_params_.period;
    const std::array<double, 8> foot_fall_deltas{};
    std::array<Vec3, 4> feet{};
    std::array<bool, 4> contact_flags{};
    std::array<double, 4> leg_phase{};

    for (int leg = 0; leg < 4; ++leg) {
      const auto [phi_value, contact] = gaitPhase(phase_, gait_params_.bias[leg], gait_params_.r);
      const auto start_swing = startSwingPos(static_cast<Leg>(leg),
                                             forward_vel_des_,
                                             turn_rate_des_,
                                             t_swing);
      const auto end_swing = endSwingPos(static_cast<Leg>(leg),
                                         forward_vel_des_,
                                         turn_rate_des_,
                                         t_swing,
                                         foot_fall_deltas);
      Vec3 delta{};
      if (contact) {
        delta = footPosContact(phi_value, end_swing, start_swing);
      } else {
        delta = footPosSwing(gait_params_.swing_h, phi_value, start_swing, end_swing);
      }
      contact_flags[leg] = contact;
      leg_phase[leg] = phi_value;
      Vec3 foot = kStandingFeet[leg];
      foot.x += delta.x;
      foot.y += delta.y;
      foot.z += delta.z;
      foot.z -= gait_params_.dbody_h;
      feet[leg] = foot;
    }

    auto q_des = inverseKinematicsAllLegs(feet);
    clampJointLimits(q_des);
    return {q_des, contact_flags, leg_phase};
  }

  void applyCommand(const std::array<double, 12>& q_des,
                    const std::array<double, 12>& kp,
                    const std::array<double, 12>& kd) {
    for (int i = 0; i < 12; ++i) {
      cmd_.motorCmd[i].mode = 0x0A;
      cmd_.motorCmd[i].q = q_des[i];
      cmd_.motorCmd[i].dq = 0.0;
      cmd_.motorCmd[i].Kp = kp[i];
      cmd_.motorCmd[i].Kd = kd[i];
      cmd_.motorCmd[i].tau = 0.0;
    }

    udp_.SetSend(cmd_);
  }

  void logSample(double timestep_ms, const std::array<double, 12>& q_des) {
    log_ << timestep_ms;
    for (double q : q_des) {
      log_ << ',' << q;
    }
    for (int i = 0; i < 12; ++i) {
      log_ << ',' << state_.motorState[i].q;
    }
    for (int i = 0; i < 12; ++i) {
      log_ << ',' << state_.motorState[i].dq;
    }
    for (int i = 0; i < 12; ++i) {
      log_ << ',' << state_.motorState[i].tauEst;
    }
    log_ << '\n';
  }

  const int stand_steps_;
  const int wait_steps_;
  const int walk_start_step_;
  const int walk_end_step_;
  UDP udp_;
  LowCmd cmd_{};
  LowState state_{};
  bool have_init_{false};
  int motion_time_{0};
  double phase_{0.0};

  GaitParams gait_params_{};
  double forward_vel_des_{0.35};
  double turn_rate_des_{0.0};

  std::array<double, 12> init_q_{};
  std::array<double, 12> last_q_des_{};
  std::array<Vec3, 4> stand_target_feet_;
  std::array<Vec3, 4> standup_start_feet_;
  std::array<double, 12> stand_pose_q_;
  std::array<double, 12> walk_kp_;
  std::array<double, 12> walk_kd_;
  std::array<double, 12> stand_kp_start_;
  std::array<double, 12> stand_kd_start_;
  std::ofstream log_;
  bool log_active_{false};
  bool finished_{false};
  std::array<bool, 4> current_contact_{};
  std::array<double, 4> current_leg_phase_{};

  bool stateReady() const {
    if (state_.tick != 0) {
      return true;
    }
    for (int i = 0; i < 12; ++i) {
      const double pos = static_cast<double>(state_.motorState[i].q);
      const double vel = static_cast<double>(state_.motorState[i].dq);
      if (std::abs(pos) > 1e-6 || std::abs(vel) > 1e-6) {
        return true;
      }
    }
    return false;
  }
};

int main() {
  std::cout << "Aliengo gait controller (stand -> wait -> walk).\n"
            << "Ensure the robot is secure before continuing. Press Enter to start...\n";
  std::cin.get();

  InitEnvironment();
  AliengoGaitController controller;

  LoopFunc loop_udp_recv("udp_recv", kDt, 3,
                         boost::bind(&AliengoGaitController::UDPRecv, &controller));
  LoopFunc loop_udp_send("udp_send", kDt, 3,
                         boost::bind(&AliengoGaitController::UDPSend, &controller));
  LoopFunc loop_control("control_loop", kDt,
                        boost::bind(&AliengoGaitController::RobotControl, &controller));

  loop_udp_recv.start();
  loop_udp_send.start();
  loop_control.start();

  while (true) {
    sleep(10);
  }

  return 0;
}
