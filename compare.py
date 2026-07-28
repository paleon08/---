import time
import numpy as np
from env.marine_env import MarineEnv
from model.DDPG import DDPGAgent
from baseline.pid_controller import PIDController

def evaluate_controller(env, controller, is_ai=True, eval_episodes=10):
    total_track_errors = []
    rudder_variations = []
    inference_latencies = []

    for ep in range(eval_episodes):
        state = env.reset()
        if hasattr(controller, 'reset'):
            controller.reset()

        done = False
        prev_action = 0.0

        while not done:
            # 1. 단일 추론 연산 지연 시간(Inference Latency, ms) 정밀 측정
            start_time = time.perf_counter()
            if is_ai:
                action = controller.get_action(state, explore=False)  # AI 추론 (노이즈 제외)
            else:
                action = controller.get_action(state)                # PID 수식 연산
            
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            inference_latencies.append(latency_ms)

            # 2. 환경 진행
            next_state, reward, done, info = env.step(action)

            # 3. 성능 평가 데이터 수집 (항로 오차, 조타 변동량)
            e_track = abs(state[0])
            delta_action = abs(action[0] - prev_action)

            total_track_errors.append(e_track)
            rudder_variations.append(delta_action)

            prev_action = action[0]
            state = next_state

    # 평균 지표 산출
    avg_track_error = np.mean(total_track_errors)
    avg_rudder_var = np.mean(rudder_variations)
    avg_latency = np.mean(inference_latencies)

    return avg_track_error, avg_rudder_var, avg_latency

def main():
    env = MarineEnv()

    # 1. DDPG AI 에이전트 생성 및 학습된 가중치 불러오기
    ai_agent = DDPGAgent(state_dim=6, action_dim=1)
    has_loaded = ai_agent.load_models(save_dir="checkpoints")
    
    if not has_loaded:
        print("[Warning] 학습된 체크포인트가 없습니다. 미학습 초기 네트워크로 평가를 진행합니다.")

    # 2. 전통 제어 PID 컨트롤러 생성
    pid_controller = PIDController()

    print("\n" + "=" * 65)
    print("⚓ [자율주행배] DDPG vs PID 성능 비교 및 연산 지연 검증")
    print("=" * 65)

    # 평가 구동
    ai_e_track, ai_smooth, ai_latency = evaluate_controller(env, ai_agent, is_ai=True)
    pid_e_track, pid_smooth, pid_latency = evaluate_controller(env, pid_controller, is_ai=False)

    # 결과 출력
    print(f"\n📊 [최종 검증 결과 분석]")
    print(f"{'평가 지표 (Metric)':<28} | {'DDPG (AI)':<15} | {'PID (전통 제어)':<15}")
    print("-" * 65)
    print(f"{'평균 항로 오차 e_track (m)':<25} | {ai_e_track:<15.4f} | {pid_e_track:<15.4f}")
    print(f"{'조타 변동량 (Smoothness)':<25} | {ai_smooth:<15.4f} | {pid_smooth:<15.4f}")
    print(f"{'단일 연산 지연 Latency (ms)':<23} | {ai_latency:<15.4f} | {pid_latency:<15.4f}")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()