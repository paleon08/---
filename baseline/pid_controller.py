import numpy as np

class PIDController:
    def __init__(self, kp_track=0.015, ki_track=0.0001, kd_track=0.08,
                 kp_heading=0.7, ki_heading=0.0005, kd_heading=0.15):
        # 1. 항로 오차(Cross-track Error) PID 게인
        self.kp_track = kp_track
        self.ki_track = ki_track
        self.kd_track = kd_track

        # 2. 각도 오차(Heading Error) PID 게인
        self.kp_heading = kp_heading
        self.ki_heading = ki_heading
        self.kd_heading = kd_heading

        # 오차 누적 및 이전 오차 변수 초기화
        self.reset()

    def reset(self):
        """에피소드 reset 시 제어기 내부 오차 상태를 초기화합니다."""
        self.integral_track = 0.0
        self.prev_track_error = 0.0
        self.integral_heading = 0.0
        self.prev_heading_error = 0.0

    def get_action(self, state):
        """
        Input state: [e_track, e_heading, u, v, r, delta_current]
        Output: delta_target (범위 [-1, 1] 로 정규화된 타각 명령)
        """
        e_track = state[0]
        e_heading = state[1]

        # 항로 오차(e_track) PID 연산
        self.integral_track += e_track
        derivative_track = e_track - self.prev_track_error
        self.prev_track_error = e_track

        u_track = (self.kp_track * e_track + 
                   self.ki_track * self.integral_track + 
                   self.kd_track * derivative_track)

        # 각도 오차(e_heading) PID 연산
        self.integral_heading += e_heading
        derivative_heading = e_heading - self.prev_heading_error
        self.prev_heading_error = e_heading

        u_heading = (self.kp_heading * e_heading + 
                     self.ki_heading * self.integral_heading + 
                     self.kd_heading * derivative_heading)

        # 제어 명령 합산
        steering_cmd = u_track + u_heading

        # DDPG 액션 범위와 동일하게 [-1.0, 1.0]으로 제어값 클리핑
        action = np.clip(steering_cmd, -1.0, 1.0)
        return np.array([action], dtype=np.float32)