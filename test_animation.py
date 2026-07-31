import numpy as np
import matplotlib.subplots as subplots
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

# 캡틴의 프로젝트 구조에 맞게 환경과 모델을 불러옵니다.
# from env.marine_env import MarineEnv
# from model.DDPG import DDPGAgent

def run_interactive_simulation(env, agent=None):
    # 1. 화면 및 UI 영역 설정
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.2) # 하단에 리셋 버튼이 들어갈 공간 확보
    
    ax.set_title("자율주행배 실시간 시뮬레이션 (단축키: R 초기화)", fontsize=14, fontweight='bold')
    ax.set_xlim(-10, 100) 
    ax.set_ylim(-10, 100)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 2. 고정된 배경 그래픽 (항로, 조류 화살표)
    ax.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Target Path')
    ax.quiver(10, 80, np.cos(np.radians(45)), np.sin(np.radians(45)), 
              color='cyan', alpha=0.5, scale=10, label='Ocean Current')
    
    # 3. 실시간으로 변할 배와 궤적 객체
    trajectory, = ax.plot([], [], 'b-', alpha=0.5, label='Ship Trajectory')
    ship_polygon = plt.Polygon([[0,0], [0,0], [0,0]], color='blue', alpha=0.8, label='Ship')
    ax.add_patch(ship_polygon)
    
    ax.legend(loc='upper left')

    # 상태 관리를 위한 딕셔너리 (애니메이션 내부에서 값을 수정하기 위함)
    sim_state = {
        'state': env.reset(),
        'done': False,
        'history_x': [],
        'history_y': []
    }

    # --- 💡 상호작용(초기화) 핵심 로직 ---
    def reset_simulation(event=None):
        """환경을 초기화하고 화면의 궤적을 지웁니다."""
        print("🔄 시뮬레이션 상황을 초기화합니다!")
        sim_state['state'] = env.reset()
        sim_state['done'] = False
        sim_state['history_x'].clear()
        sim_state['history_y'].clear()
        trajectory.set_data([], [])

    # GUI 버튼 생성 (위치: [left, bottom, width, height])
    ax_reset = plt.axes([0.4, 0.05, 0.2, 0.075])
    btn_reset = Button(ax_reset, 'RESET', color='lightgoldenrodyellow', hovercolor='0.975')
    btn_reset.on_clicked(reset_simulation) # 클릭 시 reset_simulation 함수 실행
    
    # 키보드 이벤트 연결 ('R' 키를 누르면 리셋)
    def on_key_press(event):
        if event.key.lower() == 'r':
            reset_simulation()
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    # -----------------------------------

    # 4. 프레임마다 실시간으로 환경(Env)과 상호작용하는 업데이트 함수
    def update(frame):
        # 이미 목적지에 도달했거나 범위를 이탈했다면 화면을 멈춤
        if sim_state['done']:
            return ship_polygon, trajectory
            
        current_s = sim_state['state']
        
        # 모델 행동 결정
        if agent:
            action = agent.select_action(current_s)
        else:
            action = [0.0] # 모델 없을 땐 직진
            
        # 💡 시뮬레이터에서 1 Step 진행
        next_s, reward, done, _ = env.step(action)
        sim_state['state'] = next_s
        sim_state['done'] = done
        
        # env.ship 내부의 절대 좌표 가져오기 (변수명 동기화 필수!)
        x = env.ship.x 
        y = env.ship.y
        yaw = env.ship.heading
        
        # 궤적 업데이트
        sim_state['history_x'].append(x)
        sim_state['history_y'].append(y)
        
        # 배 모양 다각형(Polygon) 렌더링 연산
        L, B = 4.0, 2.0 
        pt_bow = [x + L/2 * np.cos(yaw), y + L/2 * np.sin(yaw)]
        pt_port = [x - L/2 * np.cos(yaw) - B/2 * np.sin(yaw), y - L/2 * np.sin(yaw) + B/2 * np.cos(yaw)]
        pt_stbd = [x - L/2 * np.cos(yaw) + B/2 * np.sin(yaw), y - L/2 * np.sin(yaw) - B/2 * np.cos(yaw)]
        
        ship_polygon.set_xy([pt_bow, pt_port, pt_stbd])
        trajectory.set_data(sim_state['history_x'], sim_state['history_y'])
        
        return ship_polygon, trajectory

    # 애니메이션 실행 (실시간 연산이므로 frames=None으로 무한 반복)
    ani = animation.FuncAnimation(fig, update, frames=None, interval=50, blit=True, save_count=100)
    
    # 주의: plt.show()를 호출할 때 객체(btn_reset, ani)가 메모리에서 지워지지 않도록 유지해야 합니다.
    plt.show()

if __name__ == "__main__":
    # --- 실행 테스트용 코드 ---
    # env = MarineEnv()
    # agent = DDPGAgent(state_dim=6, action_dim=1)
    
    # run_interactive_simulation(env, agent)
    print("실시간 상호작용 UI가 탑재된 애니메이션 모듈 준비 완료!")