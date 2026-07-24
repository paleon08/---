import numpy as np
from tensorflow.keras import layers, models, optimizers



class ShipDynamics:
    """선박의 추진 물리 및 조류 외란을 계산하는 물리 엔진 클래스"""
    def __init__(self, x_init, y_init, hdg_init=0.0):
        # 선박 위치 및 선수각(Heading)
        self.x = x_init
        self.y = y_init
        self.hdg = np.radians(hdg_init) # 선수각 
        
        # 선박 고유 제원 및 성능 제약
        self.max_steering = np.radians(35.0)  # 최대 타각 35도 [cite: 354]
        self.max_stw = 28.0                  # 최대 대수속속력 (knots) [cite: 589]
        self.stw = 10.0                      # 현재 대수속력 (물에 대한 선박 자체 속력)
        
        # 조류 영향으로 계산될 대지 침로 정보 
        self.sog = 10.0                      # SOG 
        self.cog = self.hdg                  # COG 

    def update_state(self, action, current_cx, current_cy, dt=1.0):
        """Actor의 제어 명령과 현재 위치의 조류 벡터를 받아 다음 물리 상태로 업데이트"""
        # action -> [steering_cmd, shifting_cmd] [-1, 1] 범위 수신 [cite: 353, 356]
        steer_cmd = action[0] * self.max_steering
        shift_cmd = action[1]  # 0~1 사이의 엔진 출력 비율 (Sigmoid 대응) [cite: 504]
        
        # 1. 선수각(HDG) 업데이트
        self.hdg = (self.hdg + steer_cmd * dt) % (2 * np.pi)
        
        # 2. 선박 자체의 대수 속력(STW) 업데이트
        self.stw = shift_cmd * self.max_stw
        
        # 3. 선박 자체의 추진 벡터 계산 (대수 속도 벡터)
        ship_vx = self.stw * np.cos(self.hdg)
        ship_vy = self.stw * np.sin(self.hdg)
        
        # 4. 외란 합성: 선박 추진 벡터 + 조류 외란 벡터 = 실제 이동 벡터 (대지 속도 벡터)
        sog_vx = ship_vx + current_cx
        sog_vy = ship_vy + current_cy
        
        # 5. 합성된 벡터로 실제 대지 속력(SOG) 및 대지 침로(COG) 도출 
        self.sog = np.sqrt(sog_vx**2 + sog_vy**2)
        self.cog = np.atan2(sog_vy, sog_vx) % (2 * np.pi)
        
        # 6. 실제 좌표 이동 (SOG 벡터 기준 이동) [cite: 11]
        self.x += sog_vx * dt
        self.y += sog_vy * dt
        
        return self.x, self.y, self.hdg, self.cog, self.sog, self.stw