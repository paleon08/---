import numpy as np
import random
from simul.ship import ShipDynamics
from simul.ocean_current_generator import OceanCurrentField

class MarineEnv:
    """에이전트와 물리엔진을 연결하는 체육관(Gym) 환경"""
    def __init__(self):
        self.ship = ShipDynamics()
        self.ocean = OceanCurrentField()
        # 가상의 항로선 (시작점 -> 목적지)
        self.target_wp1 = np.array([0.0, 0.0])
        self.target_wp2 = np.array([800.0, 600.0])
        self.max_steps = 1000
        self.current_step = 0
        self.prev_action = 0.0

    def reset(self, current_episode=1):
        """에피소드 진행도(current_episode)에 따른 난이도 제어 (Curriculum Learning)"""
        
        # 1단계: 초기 학습 (1~100 에피소드) - 고정 제원, 약한 조류
        if current_episode <= 100:
            ship_length = 15.0                     # L (m)
            ship_mass = 2000.0                     # m (kg)
            v_c = random.uniform(0.2, 0.8)         # V_c (m/s) 약한 조류
            psi_c = random.uniform(0, 360)         # psi_c (degree)

        # 2단계: 중급 학습 (101~300 에피소드) - 제원 소폭 무작위화, 중간 조류
        elif current_episode <= 300:
            ship_length = random.uniform(12.0, 18.0)
            ship_mass = random.uniform(1500.0, 2500.0)
            v_c = random.uniform(0.5, 1.8)         # V_c 중간 조류
            psi_c = random.uniform(0, 360)

        # 3단계: 고급/극한 학습 (301 에피소드 이상) - 전 범위 무작위화
        else:
            ship_length = random.uniform(5.0, 25.0) # L
            ship_mass = random.uniform(800.0, 5000.0) # m
            v_c = random.uniform(0.5, 3.0)          # V_c 강한 조류
            psi_c = random.uniform(0, 360)

        # 결정된 파라미터를 ShipDynamics 및 OceanCurrentField에 주입
        self.ship.update_spec(length=ship_length, mass=ship_mass)
        self.ocean.set_current(speed=v_c, direction=np.radians(psi_c))

        return self._get_state()

    def _get_state(self):
        # 가야 할 항로선(Vector) 계산
        path_vector = self.target_wp2 - self.target_wp1
        ship_vector = np.array([self.ship.x, self.ship.y]) - self.target_wp1
        
        # 1. 항로 이탈 거리 (e_track) 계산 (Cross-product 활용) 
        cross_prod = path_vector[0]*ship_vector[1] - path_vector[1]*ship_vector[0]
        e_track = cross_prod / (np.linalg.norm(path_vector) + 1e-6)
        
        # 2. 각도 오차 (e_heading) 계산 
        path_angle = np.atan2(path_vector[1], path_vector[0])
        e_heading = path_angle - self.ship.hdg
        e_heading = (e_heading + np.pi) % (2 * np.pi) - np.pi # -pi ~ pi 정규화

        # 3. AI에게 전달할 최종 상태 
        state = np.array([
            e_track / 100.0,       # 정규화된 이탈 거리
            e_heading / np.pi,     # 정규화된 각도 오차
            self.ship.u / 10.0,    # 전진 속도
            self.ship.v / 5.0,     # 밀림 속도
            self.ship.r / 1.0,     # 회전 속도
            self.ship.delta_current / self.ship.max_steering # 현재 방향타 꺾임
        ], dtype=np.float32)
        
        return state

    def step(self, action):
        self.current_step += 1
        
        # Action: AI가 내린 목표 타각 명령 [-1, 1] [cite: 2510]
        delta_target = action[0] * self.ship.max_steering

        delta_diff = abs(action[0] - self.prev_action)
        self.prev_action = action[0] # 다음 스텝 비교를 위해 현재 액션 저장
        
        # 환경 업데이트
        V_c, psi_c = self.ocean.get_current(self.ship.x, self.ship.y)
        self.ship.update_state(delta_target, V_c, psi_c)
        
        # 보상 및 다음 상태 도출
        next_state = self._get_state()
        reward, done = self._calculate_reward(next_state, delta_diff)
        
        if self.current_step >= self.max_steps:
            done = True
            
        return next_state, reward, done, {}

    def _calculate_reward(self, state, delta_diff):
        e_track_real = state[0] * 100.0 # 실제 미터 단위 복원
        e_heading_real = state[1] * np.pi
        
        dist_to_target = np.linalg.norm(np.array([self.ship.x, self.ship.y]) - self.target_wp2)
        
        reward = 0.0
        done = False
        
        # 보상 설계 (APF 결합 시 수정 가능)
        if dist_to_target < 20.0:
            reward = 100.0 # 목적지 도달!
            done = True
        elif abs(e_track_real) > 200.0: 
            reward = -50.0 # 항로를 너무 크게 이탈함
            done = True
        else:
            # 항로 중심에 붙을수록, 덜 흔들릴(r) 수록 좋은 점수 부여
            path_reward = -abs(e_track_real)*0.01 - abs(e_heading_real)*0.1 - abs(self.ship.r)*0.5
            # 타각 변화량(delta_diff)이 클수록 강력한 감점 부여 (가중치 5.0은 학습하며 조절 가능)
            smoothness_penalty = delta_diff * 5.0  
            
            # 최종 보상 = 기본 항로 유지 보상 - 타각 꺾임 패널티
            reward = path_reward - smoothness_penalty
        return reward, done