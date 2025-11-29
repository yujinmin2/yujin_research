---
title: "W6D2 - Network Models"
subtitle: "신경망 네트워크 모델과 동역학"
---

# W6D2: Network Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W6D2_NetworkModels.ipynb)

---

## 📋 Overview

**핵심 질문**: 뉴런들이 연결되면 어떤 집단적 동역학이 나타나는가?

개별 뉴런의 동역학을 넘어 **네트워크 수준**의 현상을 이해합니다.

```{mermaid}
flowchart TB
    subgraph 네트워크유형
        FF[피드포워드<br/>Feedforward]
        RC[순환<br/>Recurrent]
        INH[억제-흥분<br/>E-I Balance]
    end
    
    subgraph 현상
        SYNC[동기화<br/>Synchronization]
        OSC[진동<br/>Oscillations]
        ATT[어트랙터<br/>Attractors]
    end
    
    FF --> SYNC
    RC --> OSC
    INH --> ATT
```

---

## 🎯 Learning Objectives

1. **연결 행렬**로 네트워크 구조 표현
2. **피드포워드 네트워크** 구현
3. **순환 네트워크**와 어트랙터 동역학
4. **E-I 균형** 네트워크 이해

---

## 1. 네트워크 구조

### 1.1 연결 행렬 (Connectivity Matrix)

$$W_{ij} = \text{뉴런 } j \text{에서 뉴런 } i \text{로의 시냅스 가중치}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def create_connectivity(N, p, w_mean=1.0, topology='random'):
    """
    네트워크 연결 행렬 생성
    
    N : 뉴런 수
    p : 연결 확률
    topology : 'random', 'ring', 'small_world'
    """
    if topology == 'random':
        W = np.random.randn(N, N) * w_mean
        W = W * (np.random.rand(N, N) < p)
        np.fill_diagonal(W, 0)
        
    elif topology == 'ring':
        W = np.zeros((N, N))
        for i in range(N):
            for j in range(1, int(N*p/2) + 1):
                W[i, (i+j) % N] = w_mean
                W[i, (i-j) % N] = w_mean
    
    return W

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

topologies = ['random', 'ring', 'random']
N = 50

for ax, topo in zip(axes, topologies):
    W = create_connectivity(N, p=0.2, topology=topo)
    im = ax.imshow(W, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax.set_xlabel('Pre-synaptic')
    ax.set_ylabel('Post-synaptic')
    ax.set_title(f'{topo.capitalize()} Network\n(N={N}, p=0.2)')

plt.colorbar(im, ax=axes[-1], label='Weight')
plt.tight_layout()
plt.show()
```

### 1.2 네트워크 통계

| 지표 | 정의 | 의미 |
|------|------|------|
| **연결 확률** | $p$ | 두 뉴런이 연결될 확률 |
| **평균 차수** | $k = pN$ | 평균 연결 수 |
| **클러스터링 계수** | $C$ | 이웃 간 연결 밀도 |
| **경로 길이** | $L$ | 평균 최단 경로 |

---

## 2. 피드포워드 네트워크

### 2.1 개념

입력에서 출력으로 단방향 전파:

```{mermaid}
flowchart LR
    subgraph Input
        I1[●]
        I2[●]
        I3[●]
    end
    
    subgraph Hidden
        H1[●]
        H2[●]
    end
    
    subgraph Output
        O1[●]
    end
    
    I1 --> H1
    I1 --> H2
    I2 --> H1
    I2 --> H2
    I3 --> H1
    I3 --> H2
    H1 --> O1
    H2 --> O1
```

### 2.2 스파이킹 피드포워드 네트워크

```python
def simulate_feedforward_snn(input_spikes, W, duration=100, dt=0.1,
                              tau_m=20, V_th=-55, V_reset=-70, E_L=-70):
    """
    스파이킹 피드포워드 네트워크
    
    input_spikes : (N_in, T) 입력 스파이크 배열
    W : (N_out, N_in) 가중치 행렬
    """
    N_out, N_in = W.shape
    T = int(duration / dt)
    
    V = np.ones((N_out, T)) * E_L
    spikes = np.zeros((N_out, T))
    
    for t in range(1, T):
        # 시냅스 입력
        I_syn = W @ input_spikes[:, min(t, input_spikes.shape[1]-1)]
        
        # LIF 동역학
        dV = (-(V[:, t-1] - E_L) + I_syn * 10) / tau_m * dt
        V[:, t] = V[:, t-1] + dV
        
        # 발화 체크
        fired = V[:, t] >= V_th
        spikes[fired, t] = 1
        V[fired, t] = V_reset
    
    return V, spikes

# 예시: 3층 네트워크
N_layers = [10, 20, 5]
np.random.seed(42)

# 입력 스파이크 생성
T = 500
input_spikes = (np.random.rand(N_layers[0], T) < 0.05).astype(float)

# 가중치
W1 = np.random.randn(N_layers[1], N_layers[0]) * 0.5
W2 = np.random.randn(N_layers[2], N_layers[1]) * 0.5

# 순차 시뮬레이션
V1, spikes1 = simulate_feedforward_snn(input_spikes, W1)
V2, spikes2 = simulate_feedforward_snn(spikes1, W2)

# 래스터 플롯
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, (spk, name, N) in enumerate([(input_spikes, 'Input', N_layers[0]),
                                      (spikes1, 'Hidden', N_layers[1]),
                                      (spikes2, 'Output', N_layers[2])]):
    for n in range(min(N, 20)):
        spike_times = np.where(spk[n])[0] * 0.1
        axes[i].scatter(spike_times, np.ones_like(spike_times) * n, 
                       s=2, c='black')
    axes[i].set_ylabel(f'{name}\n(N={N})')
    axes[i].set_ylim(-0.5, min(N, 20) - 0.5)

axes[-1].set_xlabel('Time (ms)')
plt.suptitle('Feedforward Spiking Network', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 3. 순환 네트워크 (Recurrent Networks)

### 3.1 Rate-based 순환 네트워크

$$\tau \frac{dr_i}{dt} = -r_i + f\left(\sum_j W_{ij} r_j + I_i\right)$$

### 3.2 어트랙터 동역학

```python
def simulate_rate_rnn(W, I_ext, duration=500, dt=1, tau=20):
    """
    Rate-based 순환 신경망
    """
    N = W.shape[0]
    T = int(duration / dt)
    
    r = np.zeros((N, T))
    r[:, 0] = np.random.rand(N) * 0.1
    
    for t in range(1, T):
        # 총 입력
        h = W @ r[:, t-1] + I_ext
        # 활성화 함수 (시그모이드)
        r_inf = 1 / (1 + np.exp(-h))
        # 동역학
        dr = (-r[:, t-1] + r_inf) / tau * dt
        r[:, t] = r[:, t-1] + dr
    
    return r

# Hopfield 네트워크 (어트랙터)
N = 100
n_patterns = 3

# 저장할 패턴 (binary)
patterns = np.sign(np.random.randn(n_patterns, N))

# Hebbian 학습으로 가중치 설정
W = np.zeros((N, N))
for p in patterns:
    W += np.outer(p, p)
W = W / n_patterns
np.fill_diagonal(W, 0)

# 손상된 패턴으로 시작
noisy_pattern = patterns[0].copy()
noise_idx = np.random.choice(N, size=int(N*0.3), replace=False)
noisy_pattern[noise_idx] *= -1

# 시뮬레이션
I_ext = noisy_pattern * 0.5
r = simulate_rate_rnn(W * 0.1, I_ext, duration=200)

# 패턴 복원 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 원래 패턴
axes[0, 0].imshow(patterns[0].reshape(10, 10), cmap='RdBu_r', aspect='auto')
axes[0, 0].set_title('Original Pattern')
axes[0, 0].axis('off')

# 손상된 패턴
axes[0, 1].imshow(noisy_pattern.reshape(10, 10), cmap='RdBu_r', aspect='auto')
axes[0, 1].set_title('Noisy Input (30% corrupted)')
axes[0, 1].axis('off')

# 복원된 패턴
recovered = np.sign(r[:, -1] - 0.5)
axes[1, 0].imshow(recovered.reshape(10, 10), cmap='RdBu_r', aspect='auto')
axes[1, 0].set_title('Recovered Pattern')
axes[1, 0].axis('off')

# 활동 변화
overlap = [np.dot(patterns[0], r[:, t]) / N for t in range(r.shape[1])]
axes[1, 1].plot(overlap, 'b-', linewidth=2)
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Overlap with Pattern')
axes[1, 1].set_title('Pattern Retrieval Dynamics')
axes[1, 1].axhline(y=1, color='red', linestyle='--', label='Perfect')
axes[1, 1].legend()

plt.suptitle('Hopfield Network: Attractor Dynamics', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 4. E-I 균형 네트워크

### 4.1 개념

흥분성(E)과 억제성(I) 뉴런의 **균형**이 네트워크 동역학의 핵심

```{mermaid}
flowchart TB
    E[흥분성 뉴런<br/>Excitatory] <--> |흥분| I[억제성 뉴런<br/>Inhibitory]
    I --> |억제| E
    E --> |흥분| E
    
    style E fill:#3498db
    style I fill:#e74c3c
```

### 4.2 E-I 네트워크 시뮬레이션

```python
def simulate_ei_network(N_E=80, N_I=20, duration=500, dt=0.1):
    """
    E-I 균형 네트워크
    """
    N = N_E + N_I
    T = int(duration / dt)
    
    # 연결 가중치
    W = np.zeros((N, N))
    
    # E→E, E→I (흥분성)
    W[:, :N_E] = np.random.rand(N, N_E) * 0.3 * (np.random.rand(N, N_E) < 0.2)
    
    # I→E, I→I (억제성)
    W[:, N_E:] = -np.random.rand(N, N_I) * 1.0 * (np.random.rand(N, N_I) < 0.5)
    
    np.fill_diagonal(W, 0)
    
    # LIF 파라미터
    tau_m = 20
    V_th = -55
    V_reset = -70
    E_L = -70
    
    V = np.ones((N, T)) * E_L
    spikes = np.zeros((N, T))
    
    for t in range(1, T):
        # 외부 입력 (포아송)
        I_ext = (np.random.rand(N) < 0.01 / dt) * 20
        
        # 시냅스 입력
        I_syn = W @ spikes[:, t-1] * 50
        
        # LIF 동역학
        dV = (-(V[:, t-1] - E_L) + I_ext + I_syn) / tau_m * dt
        V[:, t] = V[:, t-1] + dV
        
        # 발화
        fired = V[:, t] >= V_th
        spikes[fired, t] = 1
        V[fired, t] = V_reset
    
    return spikes[:N_E], spikes[N_E:], V

# 시뮬레이션
spikes_E, spikes_I, V = simulate_ei_network()

# 시각화
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# E 뉴런 래스터
for n in range(min(30, spikes_E.shape[0])):
    spike_times = np.where(spikes_E[n])[0] * 0.1
    axes[0].scatter(spike_times, np.ones_like(spike_times) * n, s=1, c='blue')
axes[0].set_ylabel('E neurons')
axes[0].set_title('E-I Balanced Network')

# I 뉴런 래스터
for n in range(min(15, spikes_I.shape[0])):
    spike_times = np.where(spikes_I[n])[0] * 0.1
    axes[1].scatter(spike_times, np.ones_like(spike_times) * n, s=1, c='red')
axes[1].set_ylabel('I neurons')

# Population rate
window = 50  # 5ms
rate_E = np.convolve(spikes_E.sum(axis=0), np.ones(window)/window, mode='same') / 0.08 * 1000
rate_I = np.convolve(spikes_I.sum(axis=0), np.ones(window)/window, mode='same') / 0.02 * 1000

t = np.arange(len(rate_E)) * 0.1
axes[2].plot(t, rate_E, 'b-', label='E rate', alpha=0.7)
axes[2].plot(t, rate_I, 'r-', label='I rate', alpha=0.7)
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('Rate (Hz)')
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## 5. 네트워크 진동 (Oscillations)

### 5.1 감마 진동 (30-80 Hz)

E-I 상호작용에서 발생하는 빠른 진동

```python
# 주파수 분석
from scipy.signal import welch

rate_total = rate_E + rate_I * 0.2
f, psd = welch(rate_total, fs=10000, nperseg=2048)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(t, rate_total, 'k-', linewidth=0.5)
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Population Rate')
axes[0].set_title('Network Activity')
axes[0].set_xlim(200, 400)

axes[1].semilogy(f, psd, 'b-')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Power')
axes[1].set_title('Power Spectrum')
axes[1].set_xlim(0, 200)
axes[1].axvspan(30, 80, alpha=0.3, color='yellow', label='Gamma band')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## 📝 실습 문제

### 문제 1: 동기화
네트워크 연결 강도에 따른 동기화 정도를 분석하세요.

### 문제 2: Small-World 네트워크
Small-world 토폴로지가 정보 전파에 미치는 영향을 분석하세요.

### 문제 3: E-I 비율
E:I 비율이 네트워크 안정성에 미치는 영향을 분석하세요.

---

## 🔗 관련 개념

- [Spiking Neural Networks](../../concepts/spiking-nn)
- [Recurrent Networks](../../concepts/recurrent-networks)
- [STDP](../../concepts/stdp)

---

## 📚 참고 자료

- Gerstner & Kistler, Chapter 12: Networks
- Dayan & Abbott, Chapter 7
- Brunel (2000): E-I Networks

---

## ⏭️ Next

```{button-ref} ../week7/day1-supervised-learning
:color: primary

다음: W7D1 - Supervised Learning →
```
