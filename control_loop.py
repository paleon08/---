import numpy as np
from simul.ocean_current_generator import OceanCurrentField
from simul.ship import ShipDynamics

class MarineEnv:
    """시뮬레이션 루프, 상태 공간 제공, 보상을 판정하는 강화학습 메인 환경"""
    def __init__(self, current_mode="extreme"):
        self.current_field = OceanCurrentField(mode=current_mode)
        self.target_x = 700.0
        self.target_y = 500.0
        self.step_count = 0
        self.reset()

    def reset(self):
        self.ship = ShipDynamics(x_init=100.0, y_init=100.0, hdg_init=90.0)
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        # 현재 위치의 조류 파악
        cx, cy = self.current_field.get_current(self.ship.x, self.ship.y, self.step_count)
        
        # 에이전트에게 제공할 확장된 상태 공간 (Input 데이터 구조화) [cite: 24]
        # [배 좌표, 선수각, 대지침로, 대수속력, 대지속력, 조류벡터X, 조류벡터Y, 목적지와의 각도/거리]
        dist_to_target = np.sqrt((self.target_x - self.ship.x)**2 + (self.target_y - self.ship.y)**2)
        angle_to_target = np.atan2(self.target_y - self.ship.y, self.target_x - self.ship.x) - self.ship.hdg
        
        state = [
            self.ship.x / 800.0, self.ship.y / 600.0,  # 맵 크기 기준 정규화 [cite: 441, 484]
            self.ship.hdg / np.pi, self.ship.cog / np.pi,
            self.ship.stw / self.ship.max_stw, self.ship.sog / self.ship.max_stw,
            cx / 5.0, cy / 5.0,  # 조류 속도 정규화
            dist_to_target / 1000.0, np.sin(angle_to_target)
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        # 1. 조류 가져오기
        cx, cy = self.current_field.get_current(self.ship.x, self.ship.y, self.step_count)
        
        # 2. 선박 물리 업데이트
        self.ship.update_state(action, cx, cy)
        
        # 3. 다음 상태 및 보상 계산
        next_state = self._get_state()
        reward, done = self._calculate_reward()
        
        return next_state, reward, done, {}

    def _calculate_reward(self):
        # 기본 논문 보상 설계식 베이스 + 조류 패널티 커스텀 가능 구역 [cite: 437]
        # 목적지 도달 시 큰 보상, 이탈/장애물 충돌 시 패널티 [cite: 437]
        done = False
        reward = 0.0
        
        dist = np.sqrt((self.target_x - self.ship.x)**2 + (self.target_y - self.ship.y)**2)
        if dist < 20.0: # 목적지 반경 도달 [cite: 437]
            reward = 2.0 [cite: 437]
            done = True
        elif self.ship.x < 0 or self.ship.x > 800 or self.ship.y < 0 or self.ship.y > 600: # 이탈 [cite: 497]
            reward = -1.0 [cite: 438]
            done = True
        else:
            # 항로 유지 효율 및 시간 패널티 [cite: 437, 444]
            reward = -0.001 
            
        return reward, done