import numpy as np

class OceanCurrentField:
    """시뮬레이션 맵 내의 조류 정보를 제공하는 클래스"""
    def __init__(self, mode="static", V_c=1.5, psi_c=45.0):
        self.mode = mode
        self.V_c = V_c                  # 조류의 속도 (m/s) 
        self.psi_c = np.radians(psi_c)  # 조류 절대 각도 (rad) 

    def get_current(self, x, y):
        # 향후 위치(x,y)에 따라 유속이 미친듯이 빨라지는 '태풍 구역' 함수를 여기에 추가 가능
        return self.V_c, self.psi_c
    def set_current(self, speed, direction):
        """환경(env)에서 에피소드마다 조류의 속도와 방향을 업데이트하는 함수"""
        self.V_c = speed
        self.psi_c = direction