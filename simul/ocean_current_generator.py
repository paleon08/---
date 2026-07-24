import numpy as np

class OceanCurrentField:
    """시뮬레이션 맵 내의 조류(방향, 유속)를 정의하는 클래스"""
    def __init__(self, mode="static", base_speed=1.5, base_direction=45.0):
        self.mode = mode  # "static", "dynamic", "extreme" (태풍/급류)
        self.base_speed = base_speed          # 기본 유속 (knots)
        self.base_direction = np.radians(base_direction) # 기본 조류 방향 (rad)

    def get_current(self, x, y, step=0):
        """특정 위치(x, y)와 시간(step)에서의 조류 속도 및 방향 벡터 반환"""
        if self.mode == "static":
            # 맵 전체에 일정한 조류가 흐르는 기본 상태
            speed = self.base_speed
            direction = self.base_direction
            
        elif self.mode == "extreme":
            # 극한 상황: 특정 구역(예: 소용돌이 또는 중심부)으로 갈수록 조류가 급격히 강해짐
            # x, y 좌표를 기반으로 변화하는 수식을 자유롭게 추가 가능
            speed = self.base_speed * (1.0 + 3.0 * np.sin(x / 100.0)) 
            direction = self.base_direction + np.radians(20.0 * np.cos(y / 100.0))
        else:
            speed, direction = self.base_speed, self.base_direction
            
        # 조류의 X, Y 성분 벡터 분할
        current_cx = speed * np.cos(direction)
        current_cy = speed * np.sin(direction)
        return current_cx, current_cy