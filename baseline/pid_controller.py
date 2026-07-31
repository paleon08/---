import numpy as np

class PIDController:
    # 👇 여기서 kp, ki, kd를 외부에서 받을 수 있게 세팅!
    def __init__(self, kp=1.5, ki=0.01, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.max_steering = 1.0 # 정규화된 타각 [-1, 1]

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def get_action(self, state, explore=False):
        # state[0] = 정규화된 항로 오차 (e_track)
        # state[1] = 정규화된 각도 오차 (e_heading)
        
        # 단순화를 위해 위치 오차와 각도 오차를 결합하여 제어 에러 계산
        error = -state[0] - state[1] 
        
        self.integral += error
        derivative = error - self.prev_error
        
        # PID 제어 수식 계산
        action = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        
        # Action을 배열 형태로 묶어서 반환 (DDPG와 형태를 맞추기 위함)
        return np.array([np.clip(action, -self.max_steering, self.max_steering)], dtype=np.float32)