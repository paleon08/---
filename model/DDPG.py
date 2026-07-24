import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import random
from collections import deque

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
    

# 3.3.2절 Actor 네트워크 객체 [cite: 338]
class ActorNetwork(tf.keras.Model):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        # 은닉층 300, 600 유닛 [cite: 359]
        self.fc1 = layers.Dense(300, activation='relu')
        self.fc2 = layers.Dense(600, activation='relu')
        
        # 출력층 분리 (논문 사양에 맞춰 활성화 함수 분리) [cite: 567]
        self.steering_output = layers.Dense(1, activation='tanh') # 조타 [-1, 1] [cite: 377, 568]
        self.shifting_output = layers.Dense(1, activation='sigmoid') # 속도증감 [0, 1] [cite: 569]

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        steering = self.steering_output(x)
        shifting = self.shifting_output(x)
        # 최종 두 행동을 결합하여 출력 [cite: 567]
        return tf.concat([steering, shifting], axis=-1)

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
    

