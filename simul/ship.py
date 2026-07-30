import numpy as np

class ShipDynamics:
    """질량, 관성, 유체 항력을 고려한 3자유도 선박 동역학 모델"""
    def __init__(self, L=10.0, B=3.0, m=2000.0, I_z=15000.0, A_R=1.5):
        # 선박 제원 파라미터 
        self.L = L
        self.B = B
        self.m = m
        self.I_z = I_z
        self.A_R = A_R
        
        self.max_steering = np.radians(35.0)
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x, y, hdg, u_init=5.0):
        # 위치 및 상태 초기화
        self.x, self.y, self.hdg = x, y, hdg
        self.u = u_init  # 전진 속도 (Surge) 
        self.v = 0.0     # 측면 밀림 속도 (Sway) 
        self.r = 0.0     # 회전 속도 (Yaw Rate) 
        self.delta_current = 0.0 # 현재 실제 타각 

    def update_state(self, delta_target, V_c, psi_c, dt=0.5):
        # 1. 조타기 지연 현상 (명령을 내려도 타가 서서히 꺾임)
        T_e = 2.0 # 조타 시정수
        self.delta_current += (delta_target - self.delta_current) / T_e * dt

        # 2. 간략화된 3자유도 운동 방정식 (가속도 계산)
        # 실제 해양공학의 복잡한 수식을 RL 훈련 속도에 맞춰 경량화한 모델
        du = (-0.1 * self.u + 0.5) * dt  # 일정한 엔진 추력 가정
        # 조류 방향(psi_c)과 배 방향(hdg)의 차이로 측면 밀림(Sway) 발생
        dv = (-0.5 * self.v - self.u * self.r + 0.1 * V_c * np.sin(psi_c - self.hdg)) * dt
        # 타각(delta)에 의한 회전력과 물의 저항(감쇠)
        dr = (-0.5 * self.r + 0.05 * self.delta_current * (self.u**2)) * dt

        # 속도 업데이트
        self.u += du
        self.v += dv
        self.r += dr

        # 3. 지구 고정 좌표계(위/경도)로 이동량 계산 (조류 속도 합성)
        current_u = V_c * np.cos(psi_c - self.hdg)
        current_v = V_c * np.sin(psi_c - self.hdg)

        U_ground = (self.u + current_u) * np.cos(self.hdg) - (self.v + current_v) * np.sin(self.hdg)
        V_ground = (self.u + current_u) * np.sin(self.hdg) + (self.v + current_v) * np.cos(self.hdg)

        # 최종 위치 및 선수각 업데이트
        self.x += U_ground * dt
        self.y += V_ground * dt
        self.hdg = (self.hdg + self.r * dt) % (2 * np.pi)

        return self.x, self.y, self.hdg, self.u, self.v, self.r

    def update_spec(self, length, mass):
        """환경(env)에서 에피소드마다 배의 제원을 변경할 때 호출되는 함수"""
        self.L = length
        self.m = mass
        
        # 💡 물리 엔진 꿀팁: 배의 무게(m)와 길이(L)가 변하면 회전 관성(I_z)도 변해야 현실적이야!
        # 직육면체 관성 모멘트 공식(1/12 * m * (L^2 + B^2))을 적용해서 AI가 더 정교한 물리법칙을 학습하게 해줄게.
        self.I_z = (1.0 / 12.0) * self.m * (self.L**2 + self.B**2)