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
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

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
constexpr std::array<const char *, 4> kLegTags = {"FR", "FL", "RR", "RL"};

// Light gravity compensation offsets (same as в статическом сборщике).
constexpr std::array<float, 12> kGravityComp = {
    -1.6f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
     1.0f, 0.0f, 0.0f};

constexpr double kTwoPi = 6.28318530717958647692;

enum class JointType {
    Hip = 0,
    Thigh = 1,
    Calf = 2
};

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

struct WaveSpec {
    int jointIndex;   // 0..11
    double base;
    double amplitude;
    double freqHz;
    double phase;
};

struct PlanStep {
    std::string name;
    double duration;
    std::vector<WaveSpec> waves;
    std::string activeJoints;
};

struct Config {
    double dt = 0.002;           // 500 Hz
    double initDuration = 1.0;   // удержание в базовой стойке перед стартом
    double limitMargin = 0.1;
    double limitGuard = 0.02;
    std::array<double, 3> kp = {17.0, 17.0, 13.5};
    std::array<double, 3> kd = {1.5, 1.3, 1.7};
    double maxTorque = 25.0;     // Н·м

    // Базовая стойка для удержания и центровки синусов.
    std::array<double, 3> basePose = {0.0, 0.8, -1.7};  // hip, thigh, calf
    std::array<double, 3> singleAmp = {0.4, 0.6, 0.4};
    double pairAmpScale = 0.8;    // 10–20% меньше, чем single
    double pairAmplitudeReduce = 0.8; // доп. снижение амплитуды для стадии 2 на 20%
    double fullAmpScale = 0.85;
    double coverageFraction = 0.9; // доля доступного диапазона, которую хотим пройти на медленной волне
    double maxDesiredVel = 3.0;    // целевой |dq| (рад/с) для подбора амплитуды (не хард-клип)

    std::array<double, 3> singleFreqs = {0.25, 0.5, 0.8};
    std::array<std::array<double, 2>, 2> pairFreqs = {{
        {0.4, 0.7},
        {0.3, 0.6}
    }};
    std::array<double, 3> fullBodyFreqs = {0.3, 0.5, 0.8};

    double singleDuration = 20.0;  // сек на одну частоту одного DOF
    double pairDuration = 20.0;    // сек на одну комбинацию частот пары
    double fullBodyDuration = 45.0;
    bool runSingle = true;
    bool runPair = true;
    bool runFull = true;
};

double ClampToRange(double value, double lower, double upper) {
    return std::max(lower, std::min(value, upper));
}

void ParseArgs(int argc, char **argv, Config &cfg) {
    bool modesProvided = false;
    auto enableMode = [&](int m) {
        switch (m) {
            case 1: cfg.runSingle = true; break;
            case 2: cfg.runPair = true; break;
            case 3: cfg.runFull = true; break;
            default:
                std::cerr << "[DynamicCollector] Unknown mode value: " << m << " (expected 1,2,3)" << std::endl;
                break;
        }
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        auto eq = arg.find('=');
        std::string key = (eq == std::string::npos) ? arg : arg.substr(0, eq);
        std::string value = (eq == std::string::npos) ? "" : arg.substr(eq + 1);

        if (key == "--mode") {
            if (!modesProvided) {
                cfg.runSingle = cfg.runPair = cfg.runFull = false;
                modesProvided = true;
            }
            std::replace(value.begin(), value.end(), ',', ' ');
            std::stringstream ss(value);
            int m = 0;
            while (ss >> m) {
                enableMode(m);
            }
        } else {
            std::cerr << "[DynamicCollector] Unknown argument: " << arg << std::endl;
        }
    }

    if (modesProvided == false) {
        // keep defaults (all enabled)
    }
}

class DynamicCollector {
public:
    explicit DynamicCollector(const Config &cfg);
    ~DynamicCollector();

    void UDPRecv();
    void UDPSend();
    void RobotControl();

private:
    void buildJointInfo();
    void buildPlan();
    double allowedAmplitude(int jointIndex, double base, double desiredAmp) const;
    WaveSpec makeWaveSpec(int jointIndex, double freqHz, double ampScale, double phase) const;
    void computeDesiredPositions(double simTime);
    void applyCommands();
    void updateStep(double simTime);
    void ensureHeader();
    void writeLog(double simTime);
    void resetCommands();
    void checkJointLimits(double simTime);

    Config config_;
    Safety safe_;
    UDP udp_;
    LowCmd cmd_;
    LowState state_;

    int motiontime_;
    size_t stepIndex_;
    double stepEnterTime_;
    std::string activeStep_;

    std::array<float, 12> desiredPositions_;
    std::array<float, 12> holdPositions_;
    std::array<float, 12> torqueCmd_;

    std::vector<JointInfo> jointInfos_;
    std::vector<PlanStep> plan_;

    std::ofstream dataFile_;
    bool headerWritten_;
    std::string activeJoints_;
};

DynamicCollector::DynamicCollector(const Config &cfg)
    : config_(cfg),
      safe_(LeggedType::Aliengo),
      udp_(LOCAL_PORT, TARGET_IP, TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH),
      motiontime_(0),
      stepIndex_(0),
      stepEnterTime_(-1.0),
      activeStep_("INIT_HOLD"),
      desiredPositions_{},
      holdPositions_{},
      torqueCmd_{},
      jointInfos_(),
      plan_(),
      dataFile_(),
      headerWritten_(false),
      activeJoints_("") {
    udp_.InitCmdData(cmd_);
    cmd_.levelFlag = LOWLEVEL;

    buildJointInfo();

    for (size_t i = 0; i < kMotorOrder.size(); ++i) {
        JointType type = jointInfos_[i].type;
        double base = config_.basePose[static_cast<int>(type)];
        base = ClampToRange(base, jointInfos_[i].narrowLower, jointInfos_[i].narrowUpper);
        holdPositions_[i] = static_cast<float>(base);
        desiredPositions_[i] = holdPositions_[i];
    }

    buildPlan();

    dataFile_.open("dynamic_sweep.csv", std::ios::out | std::ios::trunc);
    if (!dataFile_.is_open()) {
        std::cerr << "[DynamicCollector] Error: unable to open dynamic_sweep.csv for writing." << std::endl;
    } else {
        dataFile_ << std::fixed << std::setprecision(6);
    }

    std::cout << "[DynamicCollector] Plan steps: " << plan_.size()
              << " | torque limit " << config_.maxTorque << " N·m" << std::endl;
}

DynamicCollector::~DynamicCollector() {
    resetCommands();
    udp_.SetSend(cmd_);
    if (dataFile_.is_open()) {
        dataFile_.close();
    }
}

void DynamicCollector::buildJointInfo() {
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

double DynamicCollector::allowedAmplitude(int jointIndex, double base, double desiredAmp) const {
    const auto &info = jointInfos_[jointIndex];
    double lowerSpace = base - info.narrowLower;
    double upperSpace = info.narrowUpper - base;
    double maxAmp = std::max(0.0, std::min(std::abs(lowerSpace), std::abs(upperSpace)));
    return ClampToRange(desiredAmp, 0.0, maxAmp);
}

WaveSpec DynamicCollector::makeWaveSpec(int jointIndex, double freqHz, double ampScale, double phase) const {
    const auto &info = jointInfos_[jointIndex];
    double base = 0.5 * (info.narrowLower + info.narrowUpper); // центр доступного диапазона

    double lowerSpace = base - info.narrowLower;
    double upperSpace = info.narrowUpper - base;
    double symmetricSpace = std::max(0.0, std::min(lowerSpace, upperSpace));

    double desiredAmp = config_.singleAmp[static_cast<int>(info.type)] * ampScale;
    double coverageAmp = symmetricSpace * config_.coverageFraction;
    double amp = std::max(desiredAmp, coverageAmp);

    if (config_.maxDesiredVel > 1e-6 && freqHz > 1e-6) {
        double velCap = config_.maxDesiredVel / (kTwoPi * freqHz);
        amp = std::min(amp, velCap);
    }

    amp = allowedAmplitude(jointIndex, base, amp);
    return {jointIndex, base, amp, freqHz, phase};
}

void DynamicCollector::buildPlan() {
    plan_.clear();

    auto buildJointList = [this](const std::vector<WaveSpec> &waves) {
        std::string joined;
        for (size_t i = 0; i < waves.size(); ++i) {
            if (i > 0) joined += "+";
            joined += jointInfos_[waves[i].jointIndex].name;
        }
        return joined;
    };

    // 2A: одиночные DOF
    if (config_.runSingle) {
        for (size_t idx = 0; idx < kMotorOrder.size(); ++idx) {
            for (double freq : config_.singleFreqs) {
                PlanStep step;
                step.name = "1DOF_" + jointInfos_[idx].name + "_f" + std::to_string(freq);
                step.duration = config_.singleDuration;
                step.waves.push_back(makeWaveSpec(static_cast<int>(idx), freq, 1.0, 0.0));
                step.activeJoints = buildJointList(step.waves);
                plan_.push_back(step);
            }
        }
    }

    // 2B: парные синусоиды на одной ноге
    auto addPairStep = [&](size_t legIdx, int jointA, int jointB, const std::array<double, 2> &freqs, const std::string &label) {
        PlanStep step;
        std::string legTag = (legIdx < kLegTags.size()) ? kLegTags[legIdx] : ("leg" + std::to_string(legIdx));
        step.name = "PAIR_" + legTag + "_" + label +
                    "_f" + std::to_string(freqs[0]) + "-" + std::to_string(freqs[1]);
        step.duration = config_.pairDuration;

        step.waves.push_back(makeWaveSpec(jointA, freqs[0], config_.pairAmpScale, 0.0));
        step.waves.push_back(makeWaveSpec(jointB, freqs[1], config_.pairAmpScale, 0.0));
        for (auto &w : step.waves) {
            w.amplitude *= config_.pairAmplitudeReduce;
        }
        step.activeJoints = buildJointList(step.waves);
        plan_.push_back(step);
    };

    if (config_.runPair) {
        for (size_t leg = 0; leg < 4; ++leg) {
            int hip = static_cast<int>(leg * 3 + 0);
            int thigh = static_cast<int>(leg * 3 + 1);
            int calf = static_cast<int>(leg * 3 + 2);
            for (const auto &freqPair : config_.pairFreqs) {
                addPairStep(leg, hip, thigh, freqPair, "hip-thigh");
                addPairStep(leg, thigh, calf, freqPair, "thigh-calf");
            }
        }
    }

    // 2C: лёгкая full-body лиссажу (опционально)
    if (config_.runFull) {
        PlanStep full;
        full.name = "FULL_BODY";
        full.duration = config_.fullBodyDuration;

        const std::array<double, 4> phaseOffsets = {0.0, 1.0, 2.0, 3.0};

        for (size_t leg = 0; leg < 4; ++leg) {
            int hip = static_cast<int>(leg * 3 + 0);
            int thigh = static_cast<int>(leg * 3 + 1);
            int calf = static_cast<int>(leg * 3 + 2);
            double phase = phaseOffsets[leg];

            full.waves.push_back(makeWaveSpec(hip,   config_.fullBodyFreqs[0], config_.fullAmpScale, phase));
            full.waves.push_back(makeWaveSpec(thigh, config_.fullBodyFreqs[1], config_.fullAmpScale, phase + 0.5));
            full.waves.push_back(makeWaveSpec(calf,  config_.fullBodyFreqs[2], config_.fullAmpScale, phase + 1.0));
            full.activeJoints = buildJointList(full.waves);
        }
        plan_.push_back(full);
    }
}

void DynamicCollector::UDPRecv() {
    udp_.Recv();
}

void DynamicCollector::UDPSend() {
    udp_.Send();
}

void DynamicCollector::resetCommands() {
    for (size_t i = 0; i < kMotorOrder.size(); ++i) {
        int motor = kMotorOrder[i];
        cmd_.motorCmd[motor].q = PosStopF;
        cmd_.motorCmd[motor].dq = VelStopF;
        cmd_.motorCmd[motor].Kp = 0;
        cmd_.motorCmd[motor].Kd = 0;
        cmd_.motorCmd[motor].tau = 0.0f;
    }
}

void DynamicCollector::applyCommands() {
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

void DynamicCollector::checkJointLimits(double /*simTime*/) {
    for (size_t i = 0; i < kMotorOrder.size(); ++i) {
        int motor = kMotorOrder[i];
        double q = state_.motorState[motor].q;
        if (q <= jointInfos_[i].rawLower + config_.limitGuard ||
            q >= jointInfos_[i].rawUpper - config_.limitGuard) {
            std::cerr << "[WARN] " << jointInfos_[i].name << " near limit (q="
                      << q << ", bounds [" << jointInfos_[i].rawLower
                      << ", " << jointInfos_[i].rawUpper << "])" << std::endl;
        }
    }
}

void DynamicCollector::updateStep(double simTime) {
    if (simTime < config_.initDuration) {
        activeStep_ = "INIT_HOLD";
        activeJoints_.clear();
        stepEnterTime_ = -1.0;
        return;
    }

    if (stepIndex_ >= plan_.size()) {
        activeStep_ = "DONE_HOLD";
        activeJoints_.clear();
        stepEnterTime_ = -1.0;
        return;
    }

    if (stepEnterTime_ < 0.0) {
        stepEnterTime_ = simTime;
        activeStep_ = plan_[stepIndex_].name;
        activeJoints_ = plan_[stepIndex_].activeJoints;
        std::cout << "[DynamicCollector] Step " << (stepIndex_ + 1) << "/"
                  << plan_.size() << " -> " << activeStep_ << std::endl;
        return;
    }

    double elapsed = simTime - stepEnterTime_;
    if (elapsed >= plan_[stepIndex_].duration) {
        stepIndex_++;
        stepEnterTime_ = -1.0;
    }
}

void DynamicCollector::computeDesiredPositions(double simTime) {
    for (size_t i = 0; i < desiredPositions_.size(); ++i) {
        desiredPositions_[i] = holdPositions_[i];
    }

    if (stepIndex_ >= plan_.size()) {
        return;
    }
    if (stepEnterTime_ < 0.0) {
        return;
    }

    const auto &step = plan_[stepIndex_];
    double tLocal = simTime - stepEnterTime_;
    for (const auto &wave : step.waves) {
        double value = wave.base + wave.amplitude * std::sin(kTwoPi * wave.freqHz * tLocal + wave.phase);
        value = ClampToRange(value, jointInfos_[wave.jointIndex].narrowLower, jointInfos_[wave.jointIndex].narrowUpper);
        desiredPositions_[wave.jointIndex] = static_cast<float>(value);
    }
}

void DynamicCollector::ensureHeader() {
    if (headerWritten_ || !dataFile_.is_open()) {
        return;
    }
    dataFile_ << "Timestamp_ms,step,active_joints,step_elapsed";
    for (const auto *label : kMotorLabels) {
        dataFile_ << "," << label << "_q"
                  << "," << label << "_dq"
                  << "," << label << "_tauEst"
                  << "," << label << "_tauCmd";
    }
    dataFile_ << ",imu_quat_w,imu_quat_x,imu_quat_y,imu_quat_z";
    dataFile_ << ",imu_gyro_x,imu_gyro_y,imu_gyro_z";
    dataFile_ << ",imu_acc_x,imu_acc_y,imu_acc_z\n";
    dataFile_.flush();
    headerWritten_ = true;
}

void DynamicCollector::writeLog(double simTime) {
    if (!dataFile_.is_open()) {
        return;
    }
    ensureHeader();

    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    double stepElapsed = (stepEnterTime_ >= 0.0) ? (simTime - stepEnterTime_) : 0.0;
    dataFile_ << ms << "," << activeStep_ << "," << activeJoints_ << "," << stepElapsed;

    for (size_t i = 0; i < kMotorOrder.size(); ++i) {
        int motor = kMotorOrder[i];
        const auto &msState = state_.motorState[motor];
        dataFile_ << "," << msState.q
                  << "," << msState.dq
                  << "," << msState.tauEst
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

void DynamicCollector::RobotControl() {
    motiontime_++;
    udp_.GetRecv(state_);

    double simTime = motiontime_ * config_.dt;

    checkJointLimits(simTime);
    updateStep(simTime);
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
    DynamicCollector collector(config);
    InitEnvironment();

    LoopFunc loop_control("control_loop", config.dt,    boost::bind(&DynamicCollector::RobotControl, &collector));
    LoopFunc loop_udpSend("udp_send",     config.dt, 3, boost::bind(&DynamicCollector::UDPSend,     &collector));
    LoopFunc loop_udpRecv("udp_recv",     config.dt, 3, boost::bind(&DynamicCollector::UDPRecv,     &collector));

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();

    while (true) {
        sleep(10);
    }

    return 0;
}
