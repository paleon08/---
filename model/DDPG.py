import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
from collections import deque
import torch
import os

# 3.3.3절 행동 탐색 전략: OU 노이즈 [cite: 386, 393]
class OUNoise:
    def __init__(self, action_dimension, mu=0.0, theta=0.6, sigma=0.3):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        # dx = theta * (mu - x) * dt + sigma * dW 
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# 3.3.1절 경험 재플레이 메모리 [cite: 328]
class ReplayBuffer:
    def __init__(self, capacity=5000): # 논문 스펙 크기 5000 [cite: 571]
        self.buffer = deque(maxlen=capacity)
        
    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size=64): # 논문 스펙 배치 크기 64 [cite: 571]
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32), 
                np.array(actions, dtype=np.float32), 
                np.array(rewards, dtype=np.float32).reshape(-1, 1), 
                np.array(next_states, dtype=np.float32), 
                np.array(dones, dtype=np.float32).reshape(-1, 1))
                
    def __len__(self):
        return len(self.buffer)
    

# 3.3.2절 Actor 네트워크 객체
class ActorNetwork(tf.keras.Model):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        # 은닉층 300, 600 유닛
        self.fc1 = layers.Dense(300, activation='relu')
        self.fc2 = layers.Dense(600, activation='relu')
        
        # 현재는 1차원 제어(조타)만 수행하므로 속도(shifting) 출력은 제거
        self.steering_output = layers.Dense(1, activation='tanh') # 조타 [-1, 1]

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        # 단일 행동(조타 각도)만 반환
        return self.steering_output(x)
    
# 3.3.2절 Critic 네트워크 객체 [cite: 339]
class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        # 은닉층 200, 200 유닛 [cite: 364, 570]
        self.fc1 = layers.Dense(200, activation='relu')
        self.fc2 = layers.Dense(200, activation='relu')
        self.q_output = layers.Dense(1, activation=None) # 최종 Q값은 활성화 함수 없음 [cite: 367]

    def call(self, state, action):
        # 상태와 행동을 입력 파라미터로 받아 결합 [cite: 366]
        inputs = tf.concat([state, action], axis=-1)
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.q_output(x)
    

class DDPGAgent:
    """Actor, Critic, 메모리를 통합 관리하고 훈련(Gradient Update)을 수행하는 메인 두뇌"""
    def __init__(self, state_dim=6, action_dim=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99 
        self.tau = 0.001 

        self.memory = ReplayBuffer(capacity=5000)
        self.noise = OUNoise(action_dimension=action_dim)

        self.actor = ActorNetwork()
        self.target_actor = ActorNetwork()
        self.critic = CriticNetwork()
        self.target_critic = CriticNetwork()

        # 💡 [핵심 해결책] 텐서플로우가 나중에 헷갈리지 않게, 태어나자마자 강제로 뼈대를 완성시킵니다!
        dummy_state = tf.zeros((1, self.state_dim))
        dummy_action = tf.zeros((1, self.action_dim))
        
        self.actor(dummy_state)
        self.target_actor(dummy_state)
        self.critic(dummy_state, dummy_action)
        self.target_critic(dummy_state, dummy_action)
        # -----------------------------------------------------------------

        # 이제 뼈대가 완벽하게 굳어졌으니, 가중치 초기화도 안전하게 진행됩니다.
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer = optimizers.Adam(learning_rate=0.001)

    def get_action(self, state, explore=True):
        """현재 상태를 보고 타각 명령을 출력"""
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        
        # Actor 네트워크를 통과시켜 행동 도출
        action = self.actor(state_tensor)[0].numpy()
        
        # 탐색(Exploration)을 위한 노이즈 추가
        if explore:
            # 1차원 조타에만 OU 노이즈를 추가
            noise = self.noise.sample()
            action[0] += noise[0] 
            
        # 조타 범위 [-1.0, 1.0] 클리핑
        return np.clip(action, -1.0, 1.0)

    def train(self, batch_size=64):
        """메모리에서 배치를 뽑아 Actor와 Critic 가중치 업데이트"""
        if len(self.memory) < batch_size:
            return  # 메모리에 데이터가 충분히 쌓일 때까지 학습 대기

        # 1. 배치 샘플링
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # --------------------------------------------------------
        # [1] Critic (크리틱) 네트워크 업데이트
        # --------------------------------------------------------
        with tf.GradientTape() as tape:
            # 타겟 액터가 다음 상태에서 내릴 행동 예측
            target_actions = self.target_actor(next_states)
            # 타겟 크리틱이 예측한 미래 Q값
            target_q = self.target_critic(next_states, target_actions)
            # 벨만 방정식: 현재 보상 + (미래 보상 * gamma)
            y = rewards + self.gamma * target_q * (1.0 - dones)
            
            # 현재 크리틱이 예측한 Q값
            current_q = self.critic(states, actions)
            
            # MSE 손실 함수 계산
            critic_loss = tf.reduce_mean(tf.square(y - current_q))

        # 크리틱 가중치 업데이트
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # --------------------------------------------------------
        # [2] Actor (액터) 네트워크 업데이트
        # --------------------------------------------------------
        with tf.GradientTape() as tape:
            # 현재 상태에서 액터가 새로운 행동 결정
            new_actions = self.actor(states)
            # 크리틱이 그 행동을 평가 (Q값이 높을수록 좋음)
            actor_q_values = self.critic(states, new_actions)
            
            # Q값을 최대화해야 하므로, 평균에 마이너스(-)를 붙여 손실 함수로 만듦
            actor_loss = -tf.reduce_mean(actor_q_values)

        # 액터 가중치 업데이트
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # --------------------------------------------------------
        # [3] 타겟 네트워크 소프트 업데이트 (Soft Update)
        # --------------------------------------------------------
        self._update_target_networks(self.tau)

    def _update_target_networks(self, tau):
        """Polyak Averaging을 사용해 타겟 네트워크를 천천히 업데이트"""
        # Actor
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)

        # Critic
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)

    # 텐서플로우 버전에 맞게 수정된 저장/불러오기 함수
    def save_models(self, save_dir="checkpoints"):
        """학습된 Actor 및 Critic 신경망 가중치를 파일로 저장합니다."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # TensorFlow Keras의 save_weights 메서드 사용
        self.actor.save_weights(os.path.join(save_dir, "actor.weights.h5"))
        self.critic.save_weights(os.path.join(save_dir, "critic.weights.h5"))
        self.target_actor.save_weights(os.path.join(save_dir, "target_actor.weights.h5"))
        self.target_critic.save_weights(os.path.join(save_dir, "target_critic.weights.h5"))
        print(f" 💾 [System] 성공적으로 가중치를 저장했습니다: {save_dir}/")

    def load_models(self, save_dir="checkpoints"):
        """저장된 신경망 가중치를 불러와 에이전트에 적용합니다."""
        actor_path = os.path.join(save_dir, "actor.weights.h5")
        critic_path = os.path.join(save_dir, "critic.weights.h5")
        target_actor_path = os.path.join(save_dir, "target_actor.weights.h5")
        target_critic_path = os.path.join(save_dir, "target_critic.weights.h5")
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            
            if os.path.exists(target_actor_path):
                self.target_actor.load_weights(target_actor_path)
            if os.path.exists(target_critic_path):
                self.target_critic.load_weights(target_critic_path)
                
            print(f" 📂 [System] 성공적으로 가중치를 불러왔습니다: {save_dir}/")
            return True
        else:
            print(f" ⚠️ [Warning] 저장된 가중치 파일을 찾을 수 없습니다: {save_dir}/")
            return False