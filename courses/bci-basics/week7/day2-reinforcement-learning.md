---
title: "W7D2 - Reinforcement Learning"
subtitle: "강화 학습과 보상 기반 학습"
---

# W7D2: Reinforcement Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W7D2_ReinforcementLearning.ipynb)

---

## 📋 Overview

**핵심 질문**: 뇌는 어떻게 보상 신호로부터 학습하는가?

**강화 학습**은 보상을 최대화하는 행동을 시행착오를 통해 학습합니다.

```{mermaid}
flowchart LR
    A[에이전트<br/>Agent] --> |행동 a| E[환경<br/>Environment]
    E --> |상태 s| A
    E --> |보상 r| A
    
    style A fill:#3498db
    style E fill:#2ecc71
```

---

## 🎯 Learning Objectives

1. **강화 학습 프레임워크** 이해
2. **TD 학습**과 도파민 신호의 연결
3. **Q-러닝** 구현
4. **Actor-Critic** 모델 이해

---

## 1. 강화 학습 기초

### 1.1 핵심 개념

| 개념 | 기호 | 설명 |
|------|------|------|
| **상태** | $s$ | 환경의 현재 상황 |
| **행동** | $a$ | 에이전트의 선택 |
| **보상** | $r$ | 즉각적 피드백 |
| **정책** | $\pi(a|s)$ | 행동 선택 규칙 |
| **가치 함수** | $V(s)$ | 상태의 기대 보상 |
| **행동 가치** | $Q(s,a)$ | 행동의 기대 보상 |

### 1.2 목표

$$\max_\pi \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

$\gamma$: 할인 계수 (미래 보상의 가치)

---

## 2. TD 학습과 도파민

### 2.1 Temporal Difference (TD) 학습

$$V(s_t) \leftarrow V(s_t) + \alpha \cdot \underbrace{[r_t + \gamma V(s_{t+1}) - V(s_t)]}_{\text{TD Error } \delta}$$

### 2.2 도파민 = TD Error?

```{mermaid}
flowchart TB
    subgraph 신경과학
        DA[도파민 뉴런<br/>VTA/SNc]
        REW[보상 예측 오차<br/>Reward Prediction Error]
    end
    
    subgraph 강화학습
        TD[TD Error<br/>δ = r + γV' - V]
    end
    
    DA --> REW
    REW <--> |동일?| TD
```

**Schultz et al. (1997)**: 도파민 뉴런이 TD error를 인코딩

| 상황 | TD Error | 도파민 반응 |
|------|----------|-------------|
| 예상치 못한 보상 | δ > 0 | 버스트 발화 ↑ |
| 예상된 보상 | δ ≈ 0 | 변화 없음 |
| 보상 누락 | δ < 0 | 발화 억제 ↓ |

### 2.3 TD 학습 구현

```python
import numpy as np
import matplotlib.pyplot as plt

def td_learning(n_states=5, n_episodes=100, alpha=0.1, gamma=0.9):
    """
    단순 TD(0) 학습
    
    환경: 1D 격자, 오른쪽 끝이 목표 (보상 1)
    """
    V = np.zeros(n_states)
    V_history = [V.copy()]
    td_errors = []
    
    for episode in range(n_episodes):
        s = 0  # 시작 상태
        episode_errors = []
        
        while s < n_states - 1:
            # 행동: 오른쪽으로 이동
            s_next = s + 1
            
            # 보상: 목표 도달 시 1
            r = 1 if s_next == n_states - 1 else 0
            
            # TD Error
            delta = r + gamma * V[s_next] - V[s]
            episode_errors.append(delta)
            
            # 가치 업데이트
            V[s] = V[s] + alpha * delta
            
            s = s_next
        
        V_history.append(V.copy())
        td_errors.append(episode_errors)
    
    return V, np.array(V_history), td_errors

# 학습
V, V_history, td_errors = td_learning(n_states=5, n_episodes=50)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 가치 함수 변화
for i, ep in enumerate([0, 5, 10, 49]):
    axes[0, 0].plot(V_history[ep], 'o-', label=f'Episode {ep}')
axes[0, 0].set_xlabel('State')
axes[0, 0].set_ylabel('Value V(s)')
axes[0, 0].set_title('Value Function Learning')
axes[0, 0].legend()

# 최종 가치 함수
axes[0, 1].bar(range(5), V, color='steelblue', edgecolor='black')
axes[0, 1].set_xlabel('State')
axes[0, 1].set_ylabel('Value V(s)')
axes[0, 1].set_title('Final Value Function')

# TD Error 변화 (도파민 반응 시뮬레이션)
early_errors = td_errors[0]  # 초기
late_errors = td_errors[-1]   # 학습 후

axes[1, 0].bar(range(len(early_errors)), early_errors, alpha=0.7, label='Early')
axes[1, 0].axhline(y=0, color='gray', linestyle='--')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('TD Error (δ)')
axes[1, 0].set_title('TD Error = "Dopamine Signal"')
axes[1, 0].legend()

# 보상 시점 변화
axes[1, 1].bar(np.arange(len(late_errors)) - 0.2, early_errors, 0.4, 
               alpha=0.7, label='Before learning', color='blue')
axes[1, 1].bar(np.arange(len(late_errors)) + 0.2, late_errors, 0.4,
               alpha=0.7, label='After learning', color='green')
axes[1, 1].axhline(y=0, color='gray', linestyle='--')
axes[1, 1].set_xlabel('Step')
axes[1, 1].set_ylabel('TD Error')
axes[1, 1].set_title('Prediction Error Shift')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

---

## 3. Q-러닝

### 3.1 Q-함수

$$Q(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q(s', a') | s, a]$$

### 3.2 Q-러닝 업데이트

$$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

### 3.3 구현 (GridWorld)

```python
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.goal = (size-1, size-1)
        self.reset()
    
    def reset(self):
        self.pos = (0, 0)
        return self.pos
    
    def step(self, action):
        # 행동: 0=상, 1=하, 2=좌, 3=우
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_pos = (
            np.clip(self.pos[0] + moves[action][0], 0, self.size-1),
            np.clip(self.pos[1] + moves[action][1], 0, self.size-1)
        )
        self.pos = new_pos
        
        if self.pos == self.goal:
            return self.pos, 1, True  # 목표 도달
        return self.pos, -0.01, False  # 작은 패널티

def q_learning(env, episodes=500, alpha=0.1, gamma=0.95, epsilon=0.1):
    """Q-러닝"""
    Q = np.zeros((env.size, env.size, 4))
    rewards_history = []
    
    for episode in range(episodes):
        s = env.reset()
        total_reward = 0
        
        for step in range(100):
            # ε-greedy 정책
            if np.random.rand() < epsilon:
                a = np.random.randint(4)
            else:
                a = np.argmax(Q[s[0], s[1]])
            
            s_next, r, done = env.step(a)
            total_reward += r
            
            # Q 업데이트
            Q[s[0], s[1], a] += alpha * (
                r + gamma * np.max(Q[s_next[0], s_next[1]]) - Q[s[0], s[1], a]
            )
            
            s = s_next
            if done:
                break
        
        rewards_history.append(total_reward)
    
    return Q, rewards_history

# Q-러닝 실행
env = GridWorld(size=4)
Q, rewards = q_learning(env, episodes=300)

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# 학습 곡선
window = 20
smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
axes[0].plot(rewards, alpha=0.3, color='blue')
axes[0].plot(range(window-1, len(rewards)), smoothed, 'b-', linewidth=2)
axes[0].set_xlabel('Episode')
axes[0].set_ylabel('Total Reward')
axes[0].set_title('Learning Curve')

# 가치 함수
V = np.max(Q, axis=2)
im = axes[1].imshow(V, cmap='viridis')
axes[1].set_title('Value Function max Q(s,a)')
plt.colorbar(im, ax=axes[1])
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')

# 정책 (화살표)
policy = np.argmax(Q, axis=2)
arrows = ['↑', '↓', '←', '→']

for i in range(env.size):
    for j in range(env.size):
        if (i, j) != env.goal:
            axes[2].text(j, i, arrows[policy[i, j]], ha='center', va='center', fontsize=16)
        else:
            axes[2].text(j, i, '★', ha='center', va='center', fontsize=20, color='gold')

axes[2].set_xlim(-0.5, env.size-0.5)
axes[2].set_ylim(env.size-0.5, -0.5)
axes[2].set_title('Learned Policy')
axes[2].grid(True)

plt.tight_layout()
plt.show()
```

---

## 4. Actor-Critic

### 4.1 구조

```{mermaid}
flowchart TB
    S[상태 s] --> A[Actor<br/>정책 π Δ a|s Δ]
    S --> C[Critic<br/>가치 V Δ s Δ]
    
    A --> |행동| ENV[환경]
    ENV --> |보상| C
    C --> |TD Error δ| A
    
    style A fill:#e74c3c
    style C fill:#3498db
```

### 4.2 생물학적 해석

| 구성요소 | 역할 | 뇌 영역 |
|----------|------|---------|
| **Actor** | 행동 선택 | 선조체 (Striatum) |
| **Critic** | 가치 평가 | 복측 선조체 |
| **TD Error** | 학습 신호 | 도파민 (VTA) |

---

## 5. 보상 기반 시냅스 가소성

### 5.1 3요소 규칙 (Three-Factor Rule)

$$\Delta w = \eta \cdot \underbrace{(\text{pre})}_{\text{시냅스전}} \cdot \underbrace{(\text{post})}_{\text{시냅스후}} \cdot \underbrace{(\text{reward})}_{\text{조절신호}}$$

```python
def three_factor_learning():
    """3요소 학습 규칙 시뮬레이션"""
    
    np.random.seed(42)
    n_trials = 100
    n_inputs = 10
    
    # 가중치 초기화
    w = np.random.rand(n_inputs) * 0.5
    
    # 정답 패턴 (일부 입력만 보상과 연관)
    target = np.zeros(n_inputs)
    target[[2, 5, 7]] = 1
    
    w_history = [w.copy()]
    
    for trial in range(n_trials):
        # 랜덤 입력
        pre = (np.random.rand(n_inputs) > 0.5).astype(float)
        
        # 출력 (가중 합)
        post = 1 / (1 + np.exp(-np.dot(w, pre) + 2))
        
        # 보상 (정답 패턴과 유사할수록 높음)
        reward = np.corrcoef(pre, target)[0, 1]
        reward = max(0, reward)  # 양수만
        
        # 3요소 업데이트
        dw = 0.1 * pre * post * reward
        w = w + dw
        w = np.clip(w, 0, 2)
        
        w_history.append(w.copy())
    
    return np.array(w_history), target

w_history, target = three_factor_learning()

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 가중치 변화
for i in range(10):
    style = '-' if target[i] == 1 else '--'
    color = 'red' if target[i] == 1 else 'blue'
    axes[0].plot(w_history[:, i], style, color=color, alpha=0.7)

axes[0].set_xlabel('Trial')
axes[0].set_ylabel('Weight')
axes[0].set_title('Weight Evolution\n(Red=target, Blue=non-target)')

# 최종 가중치
colors = ['red' if t == 1 else 'blue' for t in target]
axes[1].bar(range(10), w_history[-1], color=colors, edgecolor='black')
axes[1].set_xlabel('Input')
axes[1].set_ylabel('Final Weight')
axes[1].set_title('Learned Weights')

plt.tight_layout()
plt.show()
```

---

## 📝 실습 문제

### 문제 1: SARSA
On-policy 알고리즘인 SARSA를 구현하고 Q-러닝과 비교하세요.

### 문제 2: 도파민 시뮬레이션
고전적 조건형성 실험의 도파민 반응을 TD 모델로 재현하세요.

### 문제 3: 탐색-활용
ε-greedy 외의 탐색 전략 (UCB, Thompson Sampling)을 구현하세요.

---

## 🔗 관련 개념

- [Reinforcement Learning](../../concepts/reinforcement-learning)
- [Hebbian Learning](../../concepts/hebbian-learning)
- [STDP](../../concepts/stdp)

---

## 📚 참고 자료

- Sutton & Barto, "Reinforcement Learning: An Introduction"
- Schultz et al. (1997): Dopamine neurons
- Dayan & Abbott, Chapter 9

---

## ⏭️ Next

```{button-ref} ../week8/day1-bci-systems
:color: primary

다음: W8D1 - BCI Systems →
```
