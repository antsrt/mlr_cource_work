Grouped LCM message types (first 12 motors only).

Files:
  - motor_raw_state_t.lcm : q_raw, dq_raw, ddq_raw, temperature
  - motor_state_t.lcm     : q, dq, ddq, tau_est
  - motor_cmd_t.lcm       : desired q, dq, tau, kp, kd
  - foot_contact_t.lcm    : force, force_est (4 feet)
  - imu_state_t.lcm       : rpy, omega, accel, alpha

Generate C++:
  cd lcm_types_grouped
  lcm-gen -x *.lcm

Suggested topics:
  - motor_raw_state
  - motor_state
  - motor_cmd
  - foot_contact
  - imu_state
