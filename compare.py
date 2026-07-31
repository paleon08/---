import time
import numpy as np
import matplotlib.pyplot as plt
from env.marine_env import MarineEnv
from model.DDPG import DDPGAgent
from baseline.pid_controller import PIDController

def evaluate_controller(env, controller, is_ai=True, eval_episodes=50, difficulty="hard"):
    """특정 난이도에서 여러 에피소드를 돌며 통계와 궤적을 수집하는 함수"""
    total_track_errors = []
    rudder_variations = []
    inference_latencies = []
    
    # 시각화를 위한 궤적 저장 (마지막 에피소드 기준)
    trajectory_x = []
    trajectory_y = []

    for ep in range(eval_episodes):
        state = env.reset()
        
        # 💡 강제 난이도 주입 (curriculum 우회)
        if difficulty == "hard":
            # 무거운 배 + 강한 측면 조류 (가장 가혹한 조건)
            env.ship.update_spec(length=20.0, mass=4000.0)
            env.ocean.set_current(speed=2.5, direction=np.radians(90)) 
        
        if hasattr(controller, 'reset'):
            controller.reset()

        done = False
        prev_action = 0.0

        # 궤적 초기화 (마지막 에피소드만 저장)
        if ep == eval_episodes - 1:
            trajectory_x = [env.ship.x]
            trajectory_y = [env.ship.y]

        while not done:
            # 1. 연산 지연 측정
            start_time = time.perf_counter()
            if is_ai:
                action = controller.get_action(state, explore=False)
            else:
                action = controller.get_action(state)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            inference_latencies.append(latency_ms)

            # 2. 환경 진행
            next_state, reward, done, info = env.step(action)

            # 3. 데이터 수집
            e_track = abs(state[0]) * 100.0 # m 단위로 복원
            delta_action = abs(action[0] - prev_action)

            total_track_errors.append(e_track)
            rudder_variations.append(delta_action)
            prev_action = action[0]
            state = next_state
            
            # 마지막 에피소드 궤적 저장
            if ep == eval_episodes - 1:
                trajectory_x.append(env.ship.x)
                trajectory_y.append(env.ship.y)

    return (np.mean(total_track_errors), np.std(total_track_errors),
            np.mean(rudder_variations), np.mean(inference_latencies),
            trajectory_x, trajectory_y)

def plot_trajectories(ai_x, ai_y, pid_x, pid_y, target_wp1, target_wp2):
    """보고서용 궤적 비교 그래프 생성"""
    plt.figure(figsize=(10, 6))
    plt.plot([target_wp1[0], target_wp2[0]], [target_wp1[1], target_wp2[1]], 
             'k--', label='Target Path (목표 항로)', linewidth=2)
    
    plt.plot(ai_x, ai_y, 'b-', label='DDPG AI (게걸음 조종)', linewidth=2.5)
    plt.plot(pid_x, pid_y, 'r-', label='PID Controller', linewidth=2.5, alpha=0.7)
    
    plt.title('Ship Trajectory Comparison under Strong Cross-Current (2.5m/s)', fontsize=14)
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 그래프를 파일로 저장
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    print("📸 [System] 궤적 비교 그래프가 'trajectory_comparison.png'로 저장되었습니다.")
    # plt.show() # 화면에 띄우고 싶다면 주석 해제

def main():
    env = MarineEnv()
    ai_agent = DDPGAgent(state_dim=6, action_dim=1)
    has_loaded = ai_agent.load_models(save_dir="checkpoints/best")
    pid_controller = PIDController(kp=2.0, ki=0.01, kd=1.0) # 튜닝된 PID

    print("\n" + "=" * 70)
    print("⚓ [자율주행배] 가혹 조건(측면 강조류) 50회 몬테카를로 통계 검증")
    print("=" * 70)

    # 50번씩 가혹 조건(hard)에서 돌리며 통계 추출
    ai_err_mean, ai_err_std, ai_smooth, ai_latency, ai_x, ai_y = evaluate_controller(env, ai_agent, is_ai=True, eval_episodes=50, difficulty="hard")
    pid_err_mean, pid_err_std, pid_smooth, pid_latency, pid_x, pid_y = evaluate_controller(env, pid_controller, is_ai=False, eval_episodes=50, difficulty="hard")

    print(f"\n📊 [50회 반복 검증 결과 분석]")
    print(f"{'평가 지표 (Metric)':<28} | {'DDPG (AI)':<15} | {'PID (전통 제어)':<15}")
    print("-" * 70)
    print(f"{'항로 오차 평균 (m)':<25} | {ai_err_mean:<15.2f} | {pid_err_mean:<15.2f}")
    print(f"{'항로 오차 표준편차 (안정성)':<23} | {ai_err_std:<15.2f} | {pid_err_std:<15.2f}")
    print(f"{'조타 변동량 (Smoothness)':<25} | {ai_smooth:<15.4f} | {pid_smooth:<15.4f}")
    print(f"{'단일 연산 지연 (ms)':<25} | {ai_latency:<15.4f} | {pid_latency:<15.4f}")
    print("=" * 70 + "\n")

    # 시각화 실행
    plot_trajectories(ai_x, ai_y, pid_x, pid_y, env.target_wp1, env.target_wp2)

if __name__ == "__main__":
    main()