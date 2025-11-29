---
title: "W6D1 - Synaptic Models"
subtitle: "시냅스 전달과 가소성 모델링"
---

# W6D1: Synaptic Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W6D1_SynapticModels.ipynb)

---

## 📋 Overview

**핵심 질문**: 뉴런 간의 시냅스 연결을 어떻게 모델링하는가?

시냅스는 뉴런 간 정보 전달의 핵심이며, **가소성**을 통해 학습이 일어납니다.

```{mermaid}
flowchart LR
    PRE[시냅스전 뉴런<br/>Spike!] --> |신경전달물질| SYN[시냅스]
    SYN --> |전류 주입| POST[시냅스후 뉴런<br/>EPSP/IPSP]
    
    style PRE fill:#e74c3c
    style SYN fill:#f39c12
    style POST fill:#3498db
```

---

## 🎯 Learning Objectives

1. **화학 시냅스**의 기본 모델 이해
2. **EPSP/IPSP** 시뮬레이션
3. **단기 가소성** (STD/STF) 구현
4. **STDP** (장기 가소성) 구현

---

## 1. 시냅스 전류 모델

### 1.1 델타 시냅스 (가장 단순)

스파이크 시 즉각적인 전류 주입:

$$I_{syn}(t) = w \cdot \sum_k \delta(t - t_k)$$

### 1.2 지수 시냅스

더 현실적: 시간에 따라 감쇠

$$\tau_s \frac{dI_{syn}}{dt} = -I_{syn}$$

스파이크 시: $I_{syn} \leftarrow I_{syn} + w$

### 1.3 이중 지수 시냅스

상승과 하강 시간 분리:

$$I_{syn}(t) = \bar{g} \cdot \frac{\tau_d}{\tau_d - \tau_r} \left( e^{-t/\tau_d} - e^{-t/\tau_r} \right)$$

```python
import numpy as np
import matplotlib.pyplot as plt

def alpha_synapse(t, tau=5):
    """알파 함수 시냅스"""
    return (t / tau) * np.exp(1 - t / tau) * (t >= 0)

def double_exp_synapse(t, tau_r=1, tau_d=5):
    """이중 지수 시냅스"""
    if tau_d == tau_r:
        tau_d = tau_r + 0.1
    norm = tau_d / (tau_d - tau_r)
    return norm * (np.exp(-t / tau_d) - np.exp(-t / tau_r)) * (t >= 0)

# 시각화
t = np.linspace(0, 50, 500)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 다양한 시냅스 커널
axes[0].plot(t, alpha_synapse(t, tau=2), label='τ=2ms (fast)')
axes[0].plot(t, alpha_synapse(t, tau=5), label='τ=5ms')
axes[0].plot(t, alpha_synapse(t, tau=10), label='τ=10ms (slow)')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Synaptic Current')
axes[0].set_title('Alpha Synapse')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# AMPA vs NMDA vs GABA
axes[1].plot(t, double_exp_synapse(t, 0.5, 3), 'b-', label='AMPA (fast)')
axes[1].plot(t, double_exp_synapse(t, 5, 100), 'g-', label='NMDA (slow)')
axes[1].plot(t, -double_exp_synapse(t, 1, 7), 'r-', label='GABA (inhibitory)')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Synaptic Current')
axes[1].set_title('Different Synapse Types')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='gray', linestyle='--')

plt.tight_layout()
plt.show()
```

---

## 2. 시냅스 전달 시뮬레이션

### 2.1 시냅스후 전위 (PSP)

```python
def simulate_synapse_lif(spike_times, w=5, tau_s=5, tau_m=20, 
                          E_L=-70, V_th=-55, duration=100, dt=0.1):
    """
    시냅스 입력을 받는 LIF 뉴런
    """
    t = np.arange(0, duration, dt)
    V = np.zeros(len(t))
    I_syn = np.zeros(len(t))
    V[0] = E_L
    
    output_spikes = []
    
    for i in range(1, len(t)):
        # 시냅스 전류 감쇠
        dI = -I_syn[i-1] / tau_s * dt
        I_syn[i] = I_syn[i-1] + dI
        
        # 입력 스파이크 체크
        if any(abs(t[i] - st) < dt for st in spike_times):
            I_syn[i] += w
        
        # LIF 동역학
        dV = (-(V[i-1] - E_L) + I_syn[i]) / tau_m * dt
        V[i] = V[i-1] + dV
        
        if V[i] >= V_th:
            output_spikes.append(t[i])
            V[i] = E_L
    
    return t, V, I_syn, output_spikes

# 단일 EPSP
spike_times = [20]
t, V, I_syn, _ = simulate_synapse_lif(spike_times, w=10, duration=80)

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# 입력 스파이크
axes[0].eventplot(spike_times, linewidths=2)
axes[0].set_ylabel('Pre-synaptic')
axes[0].set_title('Single Synaptic Input → EPSP')

# 시냅스 전류
axes[1].plot(t, I_syn, 'g-', linewidth=2)
axes[1].set_ylabel('I_syn')

# 막전위
axes[2].plot(t, V, 'b-', linewidth=2)
axes[2].axhline(y=-55, color='red', linestyle='--', label='Threshold')
axes[2].set_xlabel('Time (ms)')
axes[2].set_ylabel('V (mV)')
axes[2].legend()

plt.tight_layout()
plt.show()
```

### 2.2 시간적 합산 (Temporal Summation)

```python
# 빈도에 따른 합산
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

frequencies = [20, 50, 100, 200]  # Hz

for ax, freq in zip(axes.flat, frequencies):
    isi = 1000 / freq  # ms
    spike_times = np.arange(10, 90, isi)
    
    t, V, I_syn, output = simulate_synapse_lif(spike_times, w=3, duration=100)
    
    ax.plot(t, V, 'b-', linewidth=1.5)
    ax.axhline(y=-55, color='red', linestyle='--', alpha=0.5)
    ax.eventplot(spike_times, lineoffsets=-80, linelengths=5, colors='green')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title(f'{freq} Hz input → {len(output)} output spikes')
    ax.set_ylim(-85, -50)

plt.suptitle('Temporal Summation', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 3. 단기 가소성 (Short-Term Plasticity)

### 3.1 STD (Short-Term Depression)

반복 자극 시 시냅스 강도 **감소**

$$\frac{dx}{dt} = \frac{1-x}{\tau_d} - u \cdot x \cdot \delta(t - t_{spike})$$

### 3.2 STF (Short-Term Facilitation)

반복 자극 시 시냅스 강도 **증가**

$$\frac{du}{dt} = \frac{U-u}{\tau_f} + U(1-u) \cdot \delta(t - t_{spike})$$

### 3.3 구현 (Tsodyks-Markram Model)

```python
def simulate_stp(spike_times, U=0.2, tau_d=200, tau_f=50, 
                 A=1.0, duration=500, dt=0.1):
    """
    Short-Term Plasticity (Tsodyks-Markram 모델)
    
    U : 기본 방출 확률
    tau_d : depression 시간상수
    tau_f : facilitation 시간상수
    """
    t = np.arange(0, duration, dt)
    
    x = np.ones(len(t))  # 가용 자원 (depression)
    u = np.ones(len(t)) * U  # 방출 확률 (facilitation)
    PSP = np.zeros(len(t))
    
    for i in range(1, len(t)):
        # 회복 동역학
        dx = (1 - x[i-1]) / tau_d * dt
        du = (U - u[i-1]) / tau_f * dt
        
        x[i] = x[i-1] + dx
        u[i] = u[i-1] + du
        
        # 스파이크 체크
        if any(abs(t[i] - st) < dt for st in spike_times):
            PSP[i] = A * u[i] * x[i]
            
            # 업데이트 (스파이크 후)
            x[i] = x[i] * (1 - u[i])
            u[i] = u[i] + U * (1 - u[i])
    
    return t, x, u, PSP

# STD vs STF 비교
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

spike_times = np.arange(50, 450, 50)  # 20Hz

# STD dominant
t, x_d, u_d, psp_d = simulate_stp(spike_times, U=0.5, tau_d=200, tau_f=20)

# STF dominant  
t, x_f, u_f, psp_f = simulate_stp(spike_times, U=0.1, tau_d=50, tau_f=200)

# 왼쪽: STD
axes[0, 0].eventplot(spike_times, linewidths=1.5)
axes[0, 0].set_title('STD Dominant (U=0.5, τd=200, τf=20)')
axes[0, 0].set_ylabel('Spikes')

axes[1, 0].plot(t, x_d, 'b-', label='x (resources)')
axes[1, 0].plot(t, u_d, 'r-', label='u (release prob)')
axes[1, 0].legend()
axes[1, 0].set_ylabel('Variables')

axes[2, 0].stem(spike_times, [psp_d[int(st/0.1)] for st in spike_times], 
                basefmt=' ', linefmt='g-', markerfmt='go')
axes[2, 0].set_xlabel('Time (ms)')
axes[2, 0].set_ylabel('PSP')

# 오른쪽: STF
axes[0, 1].eventplot(spike_times, linewidths=1.5)
axes[0, 1].set_title('STF Dominant (U=0.1, τd=50, τf=200)')

axes[1, 1].plot(t, x_f, 'b-', label='x (resources)')
axes[1, 1].plot(t, u_f, 'r-', label='u (release prob)')
axes[1, 1].legend()

axes[2, 1].stem(spike_times, [psp_f[int(st/0.1)] for st in spike_times],
                basefmt=' ', linefmt='g-', markerfmt='go')
axes[2, 1].set_xlabel('Time (ms)')

plt.tight_layout()
plt.show()
```

---

## 4. STDP (Spike-Timing-Dependent Plasticity)

### 4.1 개념

```{mermaid}
flowchart LR
    subgraph LTP["Pre → Post (LTP)"]
        PRE1[Pre spike] --> |Δt > 0| POST1[Post spike]
        POST1 --> W_UP[가중치 ↑]
    end
    
    subgraph LTD["Post → Pre (LTD)"]
        POST2[Post spike] --> |Δt < 0| PRE2[Pre spike]
        PRE2 --> W_DN[가중치 ↓]
    end
```

### 4.2 STDP 규칙

$$\Delta w = \begin{cases} 
A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \text{ (LTP)} \\
-A_- e^{\Delta t / \tau_-} & \text{if } \Delta t < 0 \text{ (LTD)}
\end{cases}$$

### 4.3 구현

```python
def stdp_window(dt, A_plus=0.01, A_minus=0.012, tau_plus=20, tau_minus=20):
    """STDP 학습 창"""
    if dt > 0:
        return A_plus * np.exp(-dt / tau_plus)
    else:
        return -A_minus * np.exp(dt / tau_minus)

# STDP 창 시각화
dt_range = np.linspace(-50, 50, 200)
dw = [stdp_window(dt) for dt in dt_range]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(dt_range, dw, 'b-', linewidth=2)
axes[0].axhline(y=0, color='gray', linestyle='--')
axes[0].axvline(x=0, color='gray', linestyle='--')
axes[0].fill_between(dt_range, dw, where=np.array(dw)>0, alpha=0.3, color='green', label='LTP')
axes[0].fill_between(dt_range, dw, where=np.array(dw)<0, alpha=0.3, color='red', label='LTD')
axes[0].set_xlabel('Δt = t_post - t_pre (ms)')
axes[0].set_ylabel('Δw')
axes[0].set_title('STDP Learning Window')
axes[0].legend()

# STDP 시뮬레이션
def simulate_stdp_learning(pre_rate=20, post_rate=20, correlation=0.5, 
                           duration=1000, w_init=0.5):
    """상관관계에 따른 STDP 학습"""
    dt = 0.1
    t = np.arange(0, duration, dt)
    
    # 스파이크 생성
    np.random.seed(42)
    pre_spikes = np.random.rand(len(t)) < (pre_rate / 1000 * dt)
    
    # 상관관계 있는 post 스파이크
    post_spikes = np.zeros(len(t), dtype=bool)
    delay = int(10 / dt)  # 10ms delay
    
    for i in range(delay, len(t)):
        if pre_spikes[i - delay]:
            if np.random.rand() < correlation:
                post_spikes[i] = True
        if np.random.rand() < ((1-correlation) * post_rate / 1000 * dt):
            post_spikes[i] = True
    
    # STDP 학습
    w = w_init
    w_history = [w]
    
    pre_times = t[pre_spikes]
    post_times = t[post_spikes]
    
    for t_post in post_times:
        for t_pre in pre_times:
            dt_spike = t_post - t_pre
            if abs(dt_spike) < 50:
                w += stdp_window(dt_spike)
                w = np.clip(w, 0, 1)
        w_history.append(w)
    
    return w_history

# 다양한 상관관계
correlations = [0.0, 0.3, 0.6, 0.9]
for corr in correlations:
    w_hist = simulate_stdp_learning(correlation=corr)
    axes[1].plot(w_hist[:100], label=f'corr={corr}')

axes[1].set_xlabel('Post-synaptic spikes')
axes[1].set_ylabel('Synaptic Weight')
axes[1].set_title('STDP Learning vs Correlation')
axes[1].legend()
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()
```

---

## 📝 실습 문제

### 문제 1: NMDA 전압 의존성
Mg²⁺ 블록을 포함한 NMDA 시냅스를 구현하세요.

### 문제 2: STP 필터링
STD 시냅스가 저주파 통과 필터 역할을 하는지 확인하세요.

### 문제 3: STDP 패턴 학습
STDP로 입력 패턴을 학습하는 뉴런을 구현하세요.

---

## 🔗 관련 개념

- [시냅스](../../concepts/synapse)
- [STDP](../../concepts/stdp)
- [Hebbian Learning](../../concepts/hebbian-learning)

---

## 📚 참고 자료

- Dayan & Abbott, Chapter 5: Synaptic Conductance
- Gerstner & Kistler, Chapter 11: Synaptic Plasticity
- Bi & Poo (1998): STDP 원논문

---

## ⏭️ Next

```{button-ref} day2-network-models
:color: primary

다음: W6D2 - Network Models →
```
