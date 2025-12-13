/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "unitree_legged_sdk/aliengo_const.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Hardware E-stop must remain armed during this calibration routine.

using namespace UNITREE_LEGGED_SDK;

namespace {

constexpr uint16_t TARGET_PORT = 8007;
constexpr uint16_t LOCAL_PORT = 8082;
constexpr char TARGET_IP[] = "192.168.123.10";
const int LOW_CMD_LENGTH = 610;
const int LOW_STATE_LENGTH = 771;

constexpr std::array<int, 12> kMotorOrder = {
    FR_0, FR_1, FR_2,
    FL_0, FL_1, FL_2,
    RR_0, RR_1, RR_2,
    RL_0, RL_1, RL_2};

constexpr std::array<const char *, 12> kMotorLabels = {
    "FR_0", "FR_1", "FR_2",
    "FL_0", "FL_1", "FL_2",
    "RR_0", "RR_1", "RR_2",
    "RL_0", "RL_1", "RL_2"};

constexpr std::array<float, 12> kGravityComp = {
    -1.6f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f};

enum class Phase {
    INIT,
    MOVE,
    SETTLE,
    DWELL_LOG,
    NEXT,
    DONE
};

enum class JointType {
    Hip = 0,
    Thigh = 1,
    Calf = 2
};

const char *PhaseToString(Phase phase) {
    switch (phase) {
        case Phase::INIT: return "INIT";
        case Phase::MOVE: return "MOVE";
        case Phase::SETTLE: return "SETTLE";
        case Phase::DWELL_LOG: return "DWELL_LOG";
        case Phase::NEXT: return "NEXT";
        case Phase::DONE: return "DONE";
    }
    return "UNKNOWN";
}

JointType JointTypeForIndex(size_t index) {
    return static_cast<JointType>(index % 3);
}

struct JointInfo {
    JointType type;
    std::string name;
    double rawLower;
    double rawUpper;
    double narrowLower;
    double narrowUpper;
};

struct TargetPoint {
    int jointIndex;
    std::string name;
    double target;
    double lower;
    double upper;
};

struct Config {
    double dt = 0.002;
    double initDuration = 0.5;
    double moveDuration = 1.0;
    double holdMin = 0.25;
    double dwellDuration = 1.0;
    double cooldownDuration = 0.2;
    double posEps = 0.1;
    double velEps = 0.22;
    double limitMargin = 0.1;
    double limitGuard = 0.02;
    std::array<double, 3> kp = {17.0, 17.0, 13.5};
    std::array<double, 3> kd = {1.5, 1.3, 1.7};
    std::array<int, 3> targetCount = {15, 15, 15}; // hip, thigh, calf
    double maxTorque = 35.0;
};

void ParseArgs(int argc, char **argv, Config &cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        if (eq == std::string::npos) {
            std::cerr << "[StaticCollector] Ignoring argument (missing '='): " << arg << std::endl;
            continue;
        }
        std::string key = arg.substr(0, eq);
        std::string value = arg.substr(eq + 1);
        auto parseDouble = [&](const std::string &text, double &dest) {
            try {
                dest = std::stod(text);
            } catch (const std::exception &ex) {
                std::cerr << "[StaticCollector] Failed to parse value for " << key << ": " << ex.what() << std::endl;
            }
        };
        auto parseInt = [&](const std::string &text, int &dest) {
            try {
                dest = std::stoi(text);
            } catch (const std::exception &ex) {
                std::cerr << "[StaticCollector] Failed to parse value for " << key << ": " << ex.what() << std::endl;
            }
        };

        if (key == "--kp-hip") parseDouble(value, cfg.kp[0]);
        else if (key == "--kp-thigh") parseDouble(value, cfg.kp[1]);
        else if (key == "--kp-calf") parseDouble(value, cfg.kp[2]);
        else if (key == "--kd-hip") parseDouble(value, cfg.kd[0]);
        else if (key == "--kd-thigh") parseDouble(value, cfg.kd[1]);
        else if (key == "--kd-calf") parseDouble(value, cfg.kd[2]);
        else if (key == "--max-torque") parseDouble(value, cfg.maxTorque);
        else if (key == "--hip-count") parseInt(value, cfg.targetCount[0]);
        else if (key == "--thigh-count") parseInt(value, cfg.targetCount[1]);
        else if (key == "--calf-count") parseInt(value, cfg.targetCount[2]);
        else if (key == "--margin") parseDouble(value, cfg.limitMargin);
        else if (key == "--pos-eps") parseDouble(value, cfg.posEps);
        else if (key == "--vel-eps") parseDouble(value, cfg.velEps);
        else if (key == "--hold-min") parseDouble(value, cfg.holdMin);
        else if (key == "--move-duration") parseDouble(value, cfg.moveDuration);
        else if (key == "--dwell") parseDouble(value, cfg.dwellDuration);
        else if (key == "--cooldown") parseDouble(value, cfg.cooldownDuration);
        else {
            std::cerr << "[StaticCollector] Unknown argument: " << key << std::endl;
        }
    }
}

double ClampToRange(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

class StaticCollector {
public:
    explicit StaticCollector(const Config &cfg);
    ~StaticCollector();

    void UDPRecv();
    void UDPSend();
    void RobotControl();

private:
    void buildJointInfo();
    void buildTargets();
    void ensureHeader();
    void writeLog(double simTime);
    void updateStateMachine(double simTime);
    void computeDesiredPositions(double simTime);
    void applyCommands();
    bool checkStaticConditions() const;
    void checkJointLimits(double simTime);
    void advancePlan(double simTime);
    void setPhase(Phase newPhase, double simTime);
    void resetCommands();

    Config config_;
    Safety safe_;
    UDP udp_;
    LowCmd cmd_;
    LowState state_;

    int motiontime_;
    Phase phase_;
    double phaseEnterTime_;
    double stableStartTime_;

    int currentTargetIdx_;
    int activeJointIndex_;
    std::string activeJointName_;
    double activeTarget_;
    double rampStart_;
    double activeRawLower_;
    double activeRawUpper_;
    double activeNarrowLower_;
    double activeNarrowUpper_;

    std::array<float, 12> desiredPositions_;
    std::array<float, 12> holdPositions_;
    std::array<float, 12> torqueCmd_;

    std::ofstream dataFile_;
    bool headerWritten_;

    std::vector<JointInfo> jointInfos_;
    std::vector<TargetPoint> targets_;
};

StaticCollector::StaticCollector(const Config &cfg)
    : config_(cfg),
      safe_(LeggedType::Aliengo),
      udp_(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH),
      motiontime_(0),
      phase_(Phase::INIT),
      phaseEnterTime_(0.0),
      stableStartTime_(-1.0),
      currentTargetIdx_(-1),
      activeJointIndex_(-1),
      activeJointName_("NONE"),
      activeTarget_(0.0),
      rampStart_(0.0),
      activeRawLower_(0.0),
      activeRawUpper_(0.0),
      activeNarrowLower_(0.0),
      activeNarrowUpper_(0.0),
      desiredPositions_{},
      holdPositions_{},
      torqueCmd_{},
      dataFile_(),
      headerWritten_(false),
      jointInfos_(),
      targets_() {
    udp_.InitCmdData(cmd_);
    cmd_.levelFlag = LOWLEVEL;

    buildJointInfo();

    const std::array<float, 3> defaultHold = {0.0f, 1.2f, -2.4f};
    for (size_t i = 0; i < jointInfos_.size(); ++i) {
        const auto &info = jointInfos_[i];
        float hold = defaultHold[static_cast<int>(info.type)];
        hold = static_cast<float>(ClampToRange(static_cast<double>(hold), info.narrowLower, info.narrowUpper));
        holdPositions_[i] = hold;
        desiredPositions_[i] = hold;
    }

    buildTargets();

    dataFile_.open("static_sweep.csv", std::ios::out | std::ios::trunc);
    if (!dataFile_.is_open()) {
        std::cerr << "[StaticCollector] Error: unable to open static_sweep.csv for writing." << std::endl;
    } else {
        dataFile_ << std::fixed << std::setprecision(6);
    }

    std::cout << "[StaticCollector] Torque limit set to " << config_.maxTorque << " N·m. Targets planned: "
              << targets_.size() << std::endl;
}

StaticCollector::~StaticCollector() {
    resetCommands();
    udp_.SetSend(cmd_);
    if (dataFile_.is_open()) {
        dataFile_.close();
    }
}

void StaticCollector::buildJointInfo() {
    jointInfos_.clear();
    jointInfos_.reserve(kMotorOrder.size());
    for (size_t i = 0; i < kMotorOrder.size(); ++i) {
        JointType type = JointTypeForIndex(i);
        double rawLower = 0.0;
        double rawUpper = 0.0;
        switch (type) {
            case JointType::Hip:
                rawLower = aliengo_Hip_min;
                rawUpper = aliengo_Hip_max;
                break;
            case JointType::Thigh:
                rawLower = aliengo_Thigh_min;
                rawUpper = aliengo_Thigh_max;
                break;
            case JointType::Calf:
                rawLower = aliengo_Calf_min;
                rawUpper = aliengo_Calf_max;
                break;
        }
        double margin = config_.limitMargin;
        double narrowLower = rawLower + margin;
        double narrowUpper = rawUpper - margin;
        while (narrowLower >= narrowUpper && margin > 1e-4) {
            margin *= 0.5;
            narrowLower = rawLower + margin;
            narrowUpper = rawUpper - margin;
        }
        if (narrowLower >= narrowUpper) {
            narrowLower = rawLower;
            narrowUpper = rawUpper;
        }
        jointInfos_.push_back({type, kMotorLabels[i], rawLower, rawUpper, narrowLower, narrowUpper});
    }
}

void StaticCollector::buildTargets() {
    targets_.clear();
    for (size_t leg = 0; leg < 4; ++leg) {
        for (size_t joint = 0; joint < 3; ++joint) {
            size_t index = leg * 3 + joint;
            const auto &info = jointInfos_[index];
            int count = config_.targetCount[static_cast<int>(joint)];
            if (count <= 0) {
                continue;
            }
            std::vector<double> values;
            if (count <= 1) {
                values.push_back(ClampToRange(holdPositions_[index], info.narrowLower, info.narrowUpper));
            } else {
                for (int k = 0; k < count; ++k) {
                    double ratio = static_cast<double>(k) / static_cast<double>(count - 1);
                    double value = info.narrowLower + (info.narrowUpper - info.narrowLower) * ratio;
                    values.push_back(value);
                }
            }

            double hold = ClampToRange(static_cast<double>(holdPositions_[index]), info.narrowLower, info.narrowUpper);
            int closestIdx = 0;
            double minDiff = std::abs(values[0] - hold);
            for (size_t k = 1; k < values.size(); ++k) {
                double diff = std::abs(values[k] - hold);
                if (diff < minDiff) {
                    minDiff = diff;
                    closestIdx = static_cast<int>(k);
                }
            }
            values[closestIdx] = hold;

            std::vector<int> order(values.size());
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) {
                double da = std::abs(values[a] - hold);
                double db = std::abs(values[b] - hold);
                if (std::abs(da - db) > 1e-6) {
                    return da < db;
                }
                return values[a] < values[b];
            });

            for (int idxOrder : order) {
                double value = values[idxOrder];
                targets_.push_back({static_cast<int>(index), info.name, value, info.narrowLower, info.narrowUpper});
            }
        }
    }
}

void StaticCollector::UDPRecv() {
    udp_.Recv();
}

void StaticCollector::UDPSend() {
    udp_.Send();
}

void StaticCollector::resetCommands() {
    for (size_t i = 0; i < kMotorOrder.size(); ++i) {
        int motor = kMotorOrder[i];
        cmd_.motorCmd[motor].q = PosStopF;
        cmd_.motorCmd[motor].dq = VelStopF;
        cmd_.motorCmd[motor].Kp = 0;
        cmd_.motorCmd[motor].Kd = 0;
        cmd_.motorCmd[motor].tau = 0.0f;
    }
}

void StaticCollector::setPhase(Phase newPhase, double simTime) {
    if (phase_ == newPhase) {
        phaseEnterTime_ = simTime;
        return;
    }
    phase_ = newPhase;
    phaseEnterTime_ = simTime;
    if (newPhase == Phase::SETTLE) {
        stableStartTime_ = -1.0;
    }
    std::cout << "[StaticCollector] Phase -> " << PhaseToString(newPhase);
    if (activeJointIndex_ >= 0) {
        std::cout << " (" << activeJointName_ << ")";
    }
    std::cout << std::endl;
}

void StaticCollector::advancePlan(double simTime) {
    currentTargetIdx_++;
    if (currentTargetIdx_ >= static_cast<int>(targets_.size())) {
        activeJointIndex_ = -1;
        activeJointName_ = "NONE";
        setPhase(Phase::DONE, simTime);
        std::cout << "[StaticCollector] All targets processed. Holding nominal pose." << std::endl;
        return;
    }
    const auto &target = targets_[currentTargetIdx_];
    activeJointIndex_ = target.jointIndex;
    activeJointName_ = target.name;
    activeTarget_ = target.target;
    activeRawLower_ = jointInfos_[activeJointIndex_].rawLower;
    activeRawUpper_ = jointInfos_[activeJointIndex_].rawUpper;
    activeNarrowLower_ = target.lower;
    activeNarrowUpper_ = target.upper;
    int motor = kMotorOrder[activeJointIndex_];
    rampStart_ = state_.motorState[motor].q;
    activeTarget_ = ClampToRange(activeTarget_, activeNarrowLower_, activeNarrowUpper_);
    targets_[currentTargetIdx_].target = activeTarget_;
    std::cout << "[StaticCollector] Target " << activeJointName_ << " -> "
              << activeTarget_ << " rad (range [" << activeNarrowLower_ << ", "
              << activeNarrowUpper_ << "])" << std::endl;
    setPhase(Phase::MOVE, simTime);
}

bool StaticCollector::checkStaticConditions() const {
    if (activeJointIndex_ < 0) {
        return false;
    }
    int motor = kMotorOrder[activeJointIndex_];
    const auto &ms = state_.motorState[motor];
    double posError = std::abs(ms.q - activeTarget_);
    double velAbs = std::abs(ms.dq);
    return posError <= config_.posEps && velAbs <= config_.velEps;
}

void StaticCollector::checkJointLimits(double simTime) {
    if (activeJointIndex_ < 0) {
        return;
    }
    int motor = kMotorOrder[activeJointIndex_];
    double q = state_.motorState[motor].q;
    if (q <= activeRawLower_ + config_.limitGuard ||
        q >= activeRawUpper_ - config_.limitGuard) {
        std::cerr << "[WARN] " << activeJointName_ << " near mechanical limit (q="
                  << q << ", raw bounds [" << activeRawLower_ << ", "
                  << activeRawUpper_ << "]). Retargeting inward." << std::endl;
        double safetyLower = std::min(activeNarrowUpper_, std::max(activeNarrowLower_, activeRawLower_ + 2.0 * config_.limitGuard));
        double safetyUpper = std::max(activeNarrowLower_, std::min(activeNarrowUpper_, activeRawUpper_ - 2.0 * config_.limitGuard));
        if (safetyLower >= safetyUpper) {
            safetyLower = activeNarrowLower_;
            safetyUpper = activeNarrowUpper_;
        }
        double safeMid = 0.5 * (safetyLower + safetyUpper);
        activeTarget_ = ClampToRange(activeTarget_, safetyLower, safetyUpper);
        targets_[currentTargetIdx_].target = activeTarget_;
        rampStart_ = q;
        std::cout << "[StaticCollector] Adjusted target for " << activeJointName_
                  << " to " << activeTarget_ << " rad (safe mid " << safeMid << ")." << std::endl;
        setPhase(Phase::MOVE, simTime);
    }
}

void StaticCollector::updateStateMachine(double simTime) {
    switch (phase_) {
        case Phase::INIT:
            if (simTime - phaseEnterTime_ >= config_.initDuration) {
                advancePlan(simTime);
            }
            break;
        case Phase::MOVE:
            if (simTime - phaseEnterTime_ >= config_.moveDuration) {
                setPhase(Phase::SETTLE, simTime);
            }
            break;
        case Phase::SETTLE: {
            if (activeJointIndex_ < 0) {
                setPhase(Phase::DONE, simTime);
                return;
            }
            bool stable = checkStaticConditions();
            if (stable) {
                if (stableStartTime_ < 0.0) {
                    stableStartTime_ = simTime;
                    std::cout << "[StaticCollector] SETTLE stable window started for "
                              << activeJointName_ << std::endl;
                }
                if (simTime - stableStartTime_ >= config_.holdMin) {
                    setPhase(Phase::DWELL_LOG, simTime);
                }
            } else {
                if (stableStartTime_ >= 0.0) {
                    std::cout << "[StaticCollector] SETTLE reset (joint "
                              << activeJointName_ << ") due to motion."
                              << std::endl;
                }
                stableStartTime_ = -1.0;
            }
            break;
        }
        case Phase::DWELL_LOG:
            if (simTime - phaseEnterTime_ >= config_.dwellDuration) {
                setPhase(Phase::NEXT, simTime);
            }
            break;
        case Phase::NEXT:
            if (simTime - phaseEnterTime_ >= config_.cooldownDuration) {
                advancePlan(simTime);
            }
            break;
        case Phase::DONE:
            // Hold nominal pose.
            break;
    }
}

void StaticCollector::computeDesiredPositions(double simTime) {
    for (size_t i = 0; i < desiredPositions_.size(); ++i) {
        desiredPositions_[i] = holdPositions_[i];
    }
    if (activeJointIndex_ < 0) {
        return;
    }
    double reference = activeTarget_;
    if (phase_ == Phase::MOVE) {
        double elapsed = simTime - phaseEnterTime_;
        double alpha = std::clamp(elapsed / config_.moveDuration, 0.0, 1.0);
        reference = rampStart_ + (activeTarget_ - rampStart_) * alpha;
    }
    reference = ClampToRange(reference, activeNarrowLower_, activeNarrowUpper_);
    desiredPositions_[activeJointIndex_] = static_cast<float>(reference);
}

void StaticCollector::applyCommands() {
    for (size_t i = 0; i < kMotorOrder.size(); ++i) {
        int motor = kMotorOrder[i];
        const auto &info = jointInfos_[i];
        double kp = config_.kp[static_cast<int>(info.type)];
        double kd = config_.kd[static_cast<int>(info.type)];
        double desired = desiredPositions_[i];
        double posError = desired - state_.motorState[motor].q;
        double velError = -state_.motorState[motor].dq;
        double torque = kp * posError + kd * velError + kGravityComp[i];
        torque = ClampToRange(torque, -config_.maxTorque, config_.maxTorque);

        float commanded = static_cast<float>(torque);
        torqueCmd_[i] = commanded;

        cmd_.motorCmd[motor].q = PosStopF;
        cmd_.motorCmd[motor].dq = VelStopF;
        cmd_.motorCmd[motor].Kp = 0;
        cmd_.motorCmd[motor].Kd = 0;
        cmd_.motorCmd[motor].tau = commanded;
    }
}

void StaticCollector::ensureHeader() {
    if (headerWritten_ || !dataFile_.is_open()) {
        return;
    }
    dataFile_ << "Timestamp_ms,phase,active_joint,target_q,is_static";
    for (const auto *label : kMotorLabels) {
        dataFile_ << "," << label << "_q"
                  << "," << label << "_dq"
                  << "," << label << "_tauEst"
                  << "," << label << "_temp"
                  << "," << label << "_tauCmd";
    }
    dataFile_ << ",imu_quat_w,imu_quat_x,imu_quat_y,imu_quat_z";
    dataFile_ << ",imu_gyro_x,imu_gyro_y,imu_gyro_z";
    dataFile_ << ",imu_acc_x,imu_acc_y,imu_acc_z\n";
    dataFile_.flush();
    headerWritten_ = true;
}

void StaticCollector::writeLog(double simTime) {
    if (!dataFile_.is_open()) {
        return;
    }
    ensureHeader();

    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    const char *phaseName = PhaseToString(phase_);
    double target = (activeJointIndex_ >= 0) ? activeTarget_ : 0.0;
    int isStatic = (phase_ == Phase::DWELL_LOG) ? 1 : 0;

    dataFile_ << ms << "," << phaseName << "," << activeJointName_ << ","
              << target << "," << isStatic;

    for (size_t i = 0; i < kMotorOrder.size(); ++i) {
        int motor = kMotorOrder[i];
        const auto &msState = state_.motorState[motor];
        dataFile_ << "," << msState.q
                  << "," << msState.dq
                  << "," << msState.tauEst
                  << "," << static_cast<int>(msState.temperature)
                  << "," << torqueCmd_[i];
    }

    const auto &imu = state_.imu;
    dataFile_ << "," << imu.quaternion[0]
              << "," << imu.quaternion[1]
              << "," << imu.quaternion[2]
              << "," << imu.quaternion[3];
    dataFile_ << "," << imu.gyroscope[0]
              << "," << imu.gyroscope[1]
              << "," << imu.gyroscope[2];
    dataFile_ << "," << imu.accelerometer[0]
              << "," << imu.accelerometer[1]
              << "," << imu.accelerometer[2]
              << "\n";
    dataFile_.flush();
}

void StaticCollector::RobotControl() {
    motiontime_++;
    udp_.GetRecv(state_);

    double simTime = motiontime_ * config_.dt;

    checkJointLimits(simTime);
    updateStateMachine(simTime);
    computeDesiredPositions(simTime);
    applyCommands();
    writeLog(simTime);

    udp_.SetSend(cmd_);
}

} // namespace

int main(int argc, char **argv) {
    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Robot must be hung up. Ensure hardware E-stop is ready." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    Config config;
    ParseArgs(argc, argv, config);

    StaticCollector collector(config);
    InitEnvironment();

    LoopFunc loop_control("control_loop", config.dt,    boost::bind(&StaticCollector::RobotControl, &collector));
    LoopFunc loop_udpSend("udp_send",     config.dt, 3, boost::bind(&StaticCollector::UDPSend,     &collector));
    LoopFunc loop_udpRecv("udp_recv",     config.dt, 3, boost::bind(&StaticCollector::UDPRecv,     &collector));

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();

    while (true) {
        sleep(10);
    }

    return 0;
}
