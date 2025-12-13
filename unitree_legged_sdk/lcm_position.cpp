/*****************************************************************
 Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
******************************************************************/

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "unitree_legged_sdk/unitree_joystick.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <thread>
#include <fstream>

#include <lcm/lcm-cpp.hpp>

#include "state_estimator_lcmt.hpp"
#include "leg_control_data_lcmt.hpp"
#include "pd_tau_targets_lcmt.hpp"
#include "rc_command_lcmt.hpp"
#include "PositionGravityState.hpp"

// new lcm types
#include "motor_raw_state_t.hpp"
#include "motor_state_t.hpp"
#include "motor_cmd_t.hpp"
#include "foot_contact_t.hpp"
#include "imu_state_t.hpp"

// recorder
#include "./HDF5_recorder.h"
//#include "./T265_reader.h"
#include "./utils.h"

// #include "camera_lcm_msgs/PositionGravityState.hpp"

#include <csignal>  // Needed for signal

using namespace std; 
using namespace UNITREE_LEGGED_SDK;
// using namespace camera_lcm_msgs;

// low cmd
constexpr uint16_t TARGET_PORT = 8007;
constexpr uint16_t LOCAL_PORT = 8082;
constexpr char TARGET_IP[] = "192.168.123.10";   // target IP address

const int LOW_CMD_LENGTH = 610;
const int LOW_STATE_LENGTH = 771;

// const float PosStopF = 2.146e+9f;
// const float VelStopF = 16000.0f;

const float MAX_TORQUE = 15.0f;

HDF5Recorder HDF5Recorder;
//RealSensePose T265_reader;


// class PositionGravityStateHandler {

//     public:
//         std::vector<float> gravity;
//         std::vector<float> states_step;
    
//         void handleMessage(const lcm::ReceiveBuffer* rbuf, const std::string& chan, const PositionGravityState* msg) {
//             //gravity(msg->gravity, msg->gravity + 3);
//             for(int i=0; i<3; i++)
//             {
//                 gravity.push_back(msg->gravity[i]);
//             }
//             cout << "Received Gravity Vector: [" << gravity[0] << ", " << gravity[1] << ", " << gravity[2] << "]" << std::endl;

//             for(int i=0; i<18; i++)
//             {
//                 states_step.push_back(msg->data[i]);
//             }
//             //std::cout << "Received states_step vector:" << std::endl;
//         }
//     };

class Custom
{
public:
    Custom(uint8_t level) : safe(LeggedType::Aliengo),
                            udp(LOCAL_PORT, TARGET_IP,TARGET_PORT, LOW_CMD_LENGTH, LOW_STATE_LENGTH)
    {
        udp.InitCmdData(cmd);
        cmd.levelFlag = LOWLEVEL;
    }
    void UDPRecv();
    void UDPSend();
    void RobotControl();

    void handleMessageLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const PositionGravityState* msg);

    void init();
    void handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt * msg);
    void _simpleLCMThread();

    float record_state(float data, int scale);
    float record_action(float data);

    bool processBoolean(int8_t value);

    Safety safe;
    UDP udp;
    LowCmd cmd = {0};
    LowState state = {0};
    //IMU imu;
    Cartesian pose;
    //float qInit[3] = {0};
    float qInit[12] = {0};
    //float qDes[3] = {0};
    float qDes[12] = {0};
    float sin_mid_q[3] = {0.0, 1.2, -2.0};
    float Kp[3] = {0};
    float Kd[3] = {0};
    double time_consume = 0;
    int rate_count = 0;
    int sin_count = 0;
    int motiontime = 0;
    float dt = 0.001; // 0.001~0.01 for old policies set 0.005 for p-eam policies set 0.001

    lcm::LCM _simpleLCM;
    std::thread _simple_LCM_thread;
    bool _firstCommandReceived;
    bool _firstRun;
    state_estimator_lcmt body_state_simple = {0};
    leg_control_data_lcmt joint_state_simple = {0};
    pd_tau_targets_lcmt joint_command_simple = {0};
    rc_command_lcmt rc_command = {0};

    xRockerBtnDataStruct _keyData;
    int mode = 0;

    vector<float> actions_step;
    vector<float> states_step;
    int counter = 0;

    //PositionGravityStateHandler handler;
    std::vector<float> gravity;
    std::vector<float> steps;
    PositionGravityState camera_state;
};

void Custom::handleMessageLCM(const lcm::ReceiveBuffer* rbuf, const std::string& chan, const PositionGravityState* msg) {

    // std::cout<<"HELLO!!"<<std::endl;
    (void) rbuf;
    (void) chan;

    gravity.clear();
    steps.clear();

    //gravity(msg->gravity, msg->gravity + 3);
    for(int i=0; i<3; i++)
    {
        gravity.push_back(msg->gravity[i]);
    }
    // cout << "Received Gravity Vector: [" << gravity[0] << ", " << gravity[1] << ", " << gravity[2] << "]" << std::endl;

    for(int i=0; i<18; i++)
    {
        steps.push_back(msg->data[i]);
    }
    // std::cout << "Received states_step vector: [";
    // for (size_t i = 0; i < steps.size(); ++i) {
    //     std::cout << steps[i];
    //     if (i < steps.size() - 1) {
    //         std::cout << ", ";
    //     }
    // }
    // std::cout << "]" << std::endl;
}

void Custom::init()
{
    
    _simpleLCM.subscribe("POSITION_GRAVITY_STATE", &Custom::handleMessageLCM, this);
    _simpleLCM.subscribe("pd_plustau_targets", &Custom::handleActionLCM, this);
    _simple_LCM_thread = std::thread(&Custom::_simpleLCMThread, this);

    _firstCommandReceived = false;
    _firstRun = true;

    // set nominal pose

    for(int i = 0; i < 12; i++)
    {
        joint_command_simple.qd_des[i] = 0;
        joint_command_simple.tau_ff[i] = 0;
    }

    // best Kp/Kd from EAM identification (eam_out_4/eam_ident_results_all_joints.json)
    // const float kp_best[12] = {
    //     49.9356f, 46.4355f, 49.8013f,  // FL0, FL1, FL2
    //     49.9134f, 46.2557f, 49.7901f,  // FR0, FR1, FR2
    //     49.8681f, 48.9686f, 49.4983f,  // RL0, RL1, RL2
    //     49.8657f, 48.6259f, 49.4709f   // RR0, RR1, RR2
    // };
    // const float kd_best[12] = {
    //     1.9765f, 2.0284f, 2.0021f,
    //     1.9731f, 1.9973f, 1.9990f,
    //     1.9943f, 2.2234f, 2.0541f,
    //     2.0046f, 2.0856f, 2.0709f
    // };


    for (int i = 0; i < 12; i++) {
        joint_command_simple.kp[i] = 40.0;
        joint_command_simple.kd[i] = 1.2;
    }

    joint_command_simple.q_des[0] = -0.3;
    joint_command_simple.q_des[1] = 1.2;
    joint_command_simple.q_des[2] = -2.721;
    joint_command_simple.q_des[3] = 0.3;
    joint_command_simple.q_des[4] = 1.2;
    joint_command_simple.q_des[5] = -2.721;
    joint_command_simple.q_des[6] = -0.3;
    joint_command_simple.q_des[7] = 1.2;
    joint_command_simple.q_des[8] = -2.721;
    joint_command_simple.q_des[9] = 0.3;
    joint_command_simple.q_des[10] = 1.2;
    joint_command_simple.q_des[11] = -2.721;

    joint_command_simple.calibrated = 0;

    printf("SET NOMINAL POSE");
}

void Custom::UDPRecv()
{
    udp.Recv();
}

void Custom::UDPSend()
{
    udp.Send();
}

double jointLinearInterpolation(double initPos, double targetPos, double rate)
{
    double p;
    rate = std::min(std::max(rate, 0.0), 1.0);
    p = initPos * (1 - rate) + targetPos * rate;
    return p;
}

void Custom::handleActionLCM(const lcm::ReceiveBuffer *rbuf, const std::string & chan, const pd_tau_targets_lcmt * msg)
{
    (void) rbuf;
    (void) chan;

    joint_command_simple = *msg;
    _firstCommandReceived = true;

}

void Custom::_simpleLCMThread()
{
    while(true)
    {
        _simpleLCM.handle();
    }
}

float Custom::record_state(float data, int scale = 1)
{
    states_step.push_back(data*scale);
    return data;
}

float Custom::record_action(float data)
{   
    actions_step.push_back(data);
    return data;
}

bool Custom::processBoolean(int8_t value) {
    bool booleanValue = (value != 0);
    return booleanValue;
}

void Custom::RobotControl()
{
    states_step.clear();
    actions_step.clear();
    
    udp.GetRecv(state);

    int64_t tnow = (int64_t)state.tick * 1000000LL; // ns

    motor_raw_state_t  mraw{};
    motor_state_t      mstate{};
    motor_cmd_t        mcmd{};
    foot_contact_t     fct{};
    imu_state_t        imu{};

    mraw.utime = mstate.utime = mcmd.utime = fct.utime = imu.utime = tnow;

    // ---- 12 моторов ----
    for (int i = 0; i < 12; ++i) {
    const auto& ms = state.motorState[i];
    const auto& mc = cmd.motorCmd[i];

    mraw.q_raw[i]    = ms.q_raw;
    mraw.dq_raw[i]   = ms.dq_raw;
    mraw.ddq_raw[i]  = ms.ddq_raw;
    mraw.temperature[i] = ms.temperature;  // int8_t

    mstate.q[i]      = ms.q;
    mstate.dq[i]     = ms.dq;
    mstate.ddq[i]    = ms.ddq;
    mstate.tau_est[i]= ms.tauEst;

    mcmd.q[i]  = mc.q;
    mcmd.dq[i] = mc.dq;
    mcmd.tau[i]= mc.tau;
    mcmd.kp[i] = mc.Kp;
    mcmd.kd[i] = mc.Kd;
    }

    // ---- Контакты (4) ----
    for (int k = 0; k < 4; ++k) {
    fct.force[k]     = state.footForce[k];
    fct.force_est[k] = state.footForceEst[k];
    }

    // ---- IMU (3) ----
    for (int k = 0; k < 3; ++k) {
    imu.rpy[k]   = state.imu.rpy[k];
    imu.omega[k] = state.imu.gyroscope[k];
    imu.accel[k] = state.imu.accelerometer[k];
    }

    // alpha = d(omega)/dt (оценка)
    static float   prev_omg[3] = {0.f,0.f,0.f};
    static int64_t prev_t = 0;
    if (prev_t > 0) {
    double dt = (tnow - prev_t) * 1e-9;
    if (dt > 0) {
        for (int k = 0; k < 3; ++k)
        imu.alpha[k] = (imu.omega[k] - prev_omg[k]) / dt;
    }
    }
    for (int k = 0; k < 3; ++k) prev_omg[k] = imu.omega[k];
    prev_t = tnow;

    // ---- публикация (5 отдельных топиков) ----
    _simpleLCM.publish("motor_raw_state", &mraw);
    _simpleLCM.publish("motor_state",     &mstate);
    _simpleLCM.publish("motor_cmd",       &mcmd);
    _simpleLCM.publish("foot_contact",    &fct);
    _simpleLCM.publish("imu_state",       &imu);


    memcpy(&_keyData, &state.wirelessRemote[0], 40);
    if (_keyData.btn.components.R1 && !rc_command.right_upper_switch)
    {
        // std::cout<<"Pressed"<<std::endl;
        HDF5Recorder.new_episode();
        // right_upper_switch_pressed = false;
    }
    rc_command.left_stick[0] = record_state(_keyData.lx)/0.6;//y speed
    rc_command.left_stick[1] = record_state(_keyData.ly)/3.5;//x speed
    // std::cout<<"cmd x:"<<_keyData.ly<<"   "<<"cmd y:"<<_keyData.lx<<std::endl;
    // std::cout<<"cmd x2:"<<record_state(_keyData.ly)<<"   "<<"cmd y2:"<<record_state(_keyData.lx)<<std::endl;
    rc_command.right_stick[1] = record_state(_keyData.ry, -1)/5.0;//yaw speed
    rc_command.right_stick[0] = record_state(_keyData.rx);//z speed
    rc_command.right_lower_right_switch = _keyData.btn.components.R2;
    rc_command.right_upper_switch = _keyData.btn.components.R1;
    rc_command.left_lower_left_switch = _keyData.btn.components.L2;
    rc_command.left_upper_switch = _keyData.btn.components.L1;


    if(_keyData.btn.components.A > 0)
    {
        mode = 0;
    } 
    else if(_keyData.btn.components.B > 0)
    {
        mode = 1;
    }
    else if(_keyData.btn.components.X > 0)
    {
        mode = 2;
    }
    else if(_keyData.btn.components.Y > 0)
    {
        mode = 3;
    }
    else if(_keyData.btn.components.up > 0)
    {
        mode = 4;
    }
    else if(_keyData.btn.components.right > 0)
    {
        mode = 5;
    }
    else if(_keyData.btn.components.down > 0)
    {
        mode = 6;    
    }
    else if(_keyData.btn.components.left > 0)
    {
        mode = 7;
    }

    rc_command.mode = mode;

    // publish state to LCM
    for(int i = 0; i < 12; i++)
    {
        // joint_state_simple.q[i] = state.motorState[i].q;
        // joint_state_simple.qd[i] = state.motorState[i].dq;
        // joint_state_simple.tau_est[i] = state.motorState[i].tauEst;

        joint_state_simple.q[i] = record_state(state.motorState[i].q);
    }

    for(int i = 0; i < 12; i++)
    {
        joint_state_simple.qd[i] = record_state(state.motorState[i].dq);
        joint_state_simple.tau_est[i] = state.motorState[i].tauEst;
    }
    
    // record_action(state.motorState[i].tauEst);
    
    for(int i = 0; i < 4; i++)
    {
        body_state_simple.quat[i] = record_state(state.imu.quaternion[i]);
    }

    for(int i = 0; i < 3; i++)
    {
        body_state_simple.rpy[i] = record_state(state.imu.rpy[i]);
        body_state_simple.aBody[i] = record_state(state.imu.accelerometer[i]);
        body_state_simple.omegaBody[i] = record_state(state.imu.gyroscope[i]);
    }

    for(int i = 0; i < 4; i++)
    {
        body_state_simple.contact_estimate[i] = record_state(state.footForce[i]);
    }

    // std::vector<float> euler = T265_reader.appendPoseData(states_step);

    // std::vector<float> gravity = compute_gravity_vector(euler[0], euler[1], euler[2]);
    // std::cout << "Gravity in robot frame: [" << gravity[0] << ", " << gravity[1] << ", " << gravity[2] << "]" << std::endl;
    // record_state(gravity[0]);
    // record_state(gravity[1]);
    // record_state(gravity[2]);
    
    //lcm::LCM lcm("udpm://239.255.76.67:7667?ttl=255");

    //PositionGravityStateHandler handler;
    // lcm.subscribe("POSITION_GRAVITY_STATE", &PositionGravityStateHandler::handleMessage, &handler);

    //_simpleLCM.subscribe("POSITION_GRAVITY_STATE", &PositionGravityStateHandler::handleMessage, &handler);

    if (steps.size()==18)
    {
        for (int i = 0; i < 18; i++) {
            states_step.push_back(steps[i]);
            camera_state.data[i] = steps[i];
        }
        cout << "steps: " << steps[2] <<" "<< steps[9];
    }
    
    if (gravity.size()==3)
    {
        record_state(gravity[0]);
        record_state(gravity[1]);
        record_state(gravity[2]);
        for (int i = 0; i < 3; i++) {
            
            camera_state.gravity[i] = gravity[i];
        }
        cout << " gravity: " << gravity[0];
    }

    _simpleLCM.publish("state_estimator_data", &body_state_simple);
    _simpleLCM.publish("leg_control_data", &joint_state_simple);
    _simpleLCM.publish("rc_command", &rc_command);
    _simpleLCM.publish("camera_python", &camera_state);

    if(_firstRun && joint_state_simple.q[0] != 0)
    {
        for(int i = 0; i < 12; i++)
        {
            joint_command_simple.q_des[i] = joint_state_simple.q[i];
        }
        _firstRun = false;
    }

    for(int i = 0; i < 12; i++)
    {
        record_action(joint_command_simple.qd_des[i]);
    }

    for(int i = 0; i < 12; i++){
       
        /*Torque Mode*/ 

        // cmd.motorCmd[i].mode = 1;
        cmd.motorCmd[i].q = record_action(joint_command_simple.q_des[i]); // 2.146E+9f
        cmd.motorCmd[i].dq = 0; // 16000.0f
        cmd.motorCmd[i].Kp = joint_command_simple.kp[i];
        cmd.motorCmd[i].Kd = joint_command_simple.kd[i];
        cmd.motorCmd[i].tau = 0;

        //cout << joint_command_simple.q_des[i] << ",";
    }
    //ccout<<endl;
    
    // if(!processBoolean(joint_command_simple.calibrated))
    // {
    //     safe.PositionLimit(cmd);
    //     // int res1 = safe.PowerProtect(cmd, state, 9);
    //     safe.PowerProtect(cmd, state, 9);
        
    //     udp.SetSend(cmd);
    // }

    safe.PositionLimit(cmd);
    safe.PowerProtect(cmd, state, 9);
        
    udp.SetSend(cmd);
    
    if(counter >= 10)
    {
        HDF5Recorder.record_step(actions_step, states_step);
        counter = 0;
    }
    else
    {
        counter++;
    }

    // CSV logging similar to aliengo_gait_controller.cpp
    static std::ofstream csv_log;
    static bool csv_initialized = false;
    if (!csv_initialized) {
        csv_log.open("lcm_position_log.csv", std::ios::out | std::ios::trunc);
        if (csv_log.is_open()) {
            csv_log << "timestep_ms";
            const char* prefixes[] = {"q_des_", "q_", "qd_", "tau_"};
            const char* joint_names[] = {
                "FR0", "FR1", "FR2", "FL0", "FL1", "FL2",
                "RR0", "RR1", "RR2", "RL0", "RL1", "RL2"};
            for (const char* prefix : prefixes) {
                for (const char* name : joint_names) {
                    csv_log << ',' << prefix << name;
                }
            }
            csv_log << '\n';
            csv_initialized = true;
        }
    }
    if (csv_initialized) {
        double t_ms = motiontime * dt * 1000.0;
        csv_log << t_ms;
        for (int i = 0; i < 12; ++i) {
            csv_log << ',' << joint_command_simple.q_des[i];
        }
        for (int i = 0; i < 12; ++i) {
            csv_log << ',' << state.motorState[i].q;
        }
        for (int i = 0; i < 12; ++i) {
            csv_log << ',' << state.motorState[i].dq;
        }
        for (int i = 0; i < 12; ++i) {
            csv_log << ',' << state.motorState[i].tauEst;
        }
        csv_log << '\n';
    }

    // for(auto vel: states_step){
    //     std::cout<<vel<< "  ";
    // }
    // for(int i = 0; i<4; i++){
    //     std::cout<<states_step[i]<<"  ";
    // }
    // std::cout<<std::endl;
}


int main(void)
{
    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    Custom custom(LOWLEVEL);
    custom.init();
    // InitEnvironment();
    LoopFunc loop_control("control_loop", custom.dt,    boost::bind(&Custom::RobotControl, &custom));
    LoopFunc loop_udpSend("udp_send",     custom.dt, 3, boost::bind(&Custom::UDPSend,      &custom));
    LoopFunc loop_udpRecv("udp_recv",     custom.dt, 3, boost::bind(&Custom::UDPRecv,      &custom));

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();

    while(1){
        sleep(10);
    };

    return 0;
}
