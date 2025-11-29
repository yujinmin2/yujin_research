---
title: "W5D2 - Simplified Neuron Models"
subtitle: "간소화된 뉴런 모델: LIF, IF, 그리고 그 너머"
---

# W5D2: Simplified Neuron Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W5D2_NeuronModels.ipynb)

---

## 📋 Overview

**핵심 질문**: H-H 모델보다 간단하면서도 핵심 동역학을 포착하는 모델은?

실제 신경망 시뮬레이션에서는 계산 효율성을 위해 **간소화된 모델**을 사용합니다.

```{mermaid}
flowchart LR
    subgraph 복잡도
        HH[Hodgkin-Huxley<br/>4 변수]
        LIF[Leaky IF<br/>1 변수]
        IF[Integrate-Fire<br/>1 변수]
        IZH[Izhikevich<br/>2 변수]
    end
    
    HH --> |단순화| LIF
    LIF --> |단순화| IF
    HH --> |근사| IZH
    
    style HH fill:#e74c3c
    style LIF fill:#f39c12
    style IF fill:#2ecc71
    style IZH fill:#3498db
```

---

## 🎯 Learning Objectives

1. **Integrate-and-Fire (IF)** 모델 이해
2. **Leaky Integrate-and-Fire (LIF)** 모델 구현
3. **Izhikevich 모델**로 다양한 발화 패턴 재현
4. 모델 간 **트레이드오프** 이해

---

## 1. Integrate-and-Fire (IF) 모델

### 1.1 개념

가장 단순한 뉴런 모델:
- 입력 전류를 **적분**
- 역치 도달 시 **발화** 후 리셋

```{mermaid}
flowchart LR
    I[입력 전류 I] --> INT[적분<br/>dV/dt = I/C]
    INT --> V{V > Vth?}
    V -->|Yes| SPIKE[스파이크!]
    V -->|No| INT
    SPIKE --> RESET[V = Vreset]
    RESET --> INT
```

### 1.2 수식

$$C \frac{dV}{dt} = I$$

**발화 조건**: $V \geq V_{th}$ → 스파이크 발생, $V \rightarrow V_{reset}$

### 1.3 구현

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_if(I_ext, duration=100, dt=0.1, C=1, V_th=-55, V_reset=-70, V_init=-65):
    """
    Integrate-and-Fire 모델
    """
    t = np.arange(0, duration, dt)
    V = np.zeros(len(t))
    V[0] = V_init
    spikes = []
    
    for i in range(1, len(t)):
        # 적분
        dV = I_ext / C * dt
        V[i] = V[i-1] + dV
        
        # 발화 체크
        if V[i] >= V_th:
            spikes.append(t[i])
            V[i] = V_reset
    
    return t, V, spikes

# 시뮬레이션
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

I_values = [10, 15, 20]
colors = ['blue', 'green', 'red']

for I, color in zip(I_values, colors):
    t, V, spikes = simulate_if(I, duration=100)
    rate = len(spikes) / 0.1  # Hz
    axes[0].plot(t, V, color=color, linewidth=1, label=f'I={I}, rate={rate:.0f}Hz')

axes[0].axhline(y=-55, color='gray', linestyle='--', label='Threshold')
axes[0].set_ylabel('V (mV)')
axes[0].set_title('Integrate-and-Fire Model')
axes[0].legend()

# F-I 곡선
I_range = np.linspace(5, 30, 20)
rates = []
for I in I_range:
    _, _, spikes = simulate_if(I, duration=500)
    rates.append(len(spikes) / 0.5)

axes[1].plot(I_range, rates, 'ko-', linewidth=2)
axes[1].set_xlabel('Current (pA)')
axes[1].set_ylabel('Firing Rate (Hz)')
axes[1].set_title('F-I Curve')

plt.tight_layout()
plt.show()
```

---

## 2. Leaky Integrate-and-Fire (LIF) 모델

### 2.1 개념

IF에 **막 누출**을 추가:

$$\tau_m \frac{dV}{dt} = -(V - E_L) + R_m I$$

- $\tau_m$: 막 시간 상수
- $E_L$: 누출 역전위 (휴지 전위)
- $R_m$: 막 저항

### 2.2 특성

| 특성 | IF | LIF |
|------|-----|-----|
| 휴지 전위 | 없음 | $E_L$ |
| 시간 상수 | 없음 | $\tau_m$ |
| F-I 관계 | 선형 | 비선형 |
| 생물학적 타당성 | 낮음 | 중간 |

### 2.3 구현

```python
def simulate_lif(I_ext, duration=100, dt=0.1, 
                 tau_m=20, E_L=-70, R_m=10, 
                 V_th=-55, V_reset=-70, t_ref=2):
    """
    Leaky Integrate-and-Fire 모델
    
    Parameters:
    -----------
    tau_m : float - 막 시간 상수 (ms)
    E_L : float - 누출 역전위 (mV)
    R_m : float - 막 저항 (MΩ)
    t_ref : float - 불응기 (ms)
    """
    t = np.arange(0, duration, dt)
    V = np.zeros(len(t))
    V[0] = E_L
    spikes = []
    ref_counter = 0
    
    for i in range(1, len(t)):
        if ref_counter > 0:
            # 불응기
            V[i] = V_reset
            ref_counter -= dt
        else:
            # LIF 동역학
            dV = (-(V[i-1] - E_L) + R_m * I_ext) / tau_m * dt
            V[i] = V[i-1] + dV
            
            if V[i] >= V_th:
                spikes.append(t[i])
                V[i] = V_reset
                ref_counter = t_ref
    
    return t, V, spikes

# IF vs LIF 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 시간 응답
I = 3  # pA
t_if, V_if, sp_if = simulate_if(I, duration=100)
t_lif, V_lif, sp_lif = simulate_lif(I, duration=100)

axes[0, 0].plot(t_if, V_if, 'b-', linewidth=1.5, label='IF')
axes[0, 0].plot(t_lif, V_lif, 'r-', linewidth=1.5, label='LIF')
axes[0, 0].axhline(y=-55, color='gray', linestyle='--')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('V (mV)')
axes[0, 0].set_title(f'Subthreshold Response (I={I})')
axes[0, 0].legend()

# 발화 응답
I = 15
t_if, V_if, sp_if = simulate_if(I, duration=100)
t_lif, V_lif, sp_lif = simulate_lif(I, duration=100)

axes[0, 1].plot(t_if, V_if, 'b-', linewidth=1, label='IF')
axes[0, 1].plot(t_lif, V_lif, 'r-', linewidth=1, label='LIF')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('V (mV)')
axes[0, 1].set_title(f'Spiking Response (I={I})')
axes[0, 1].legend()

# F-I 곡선 비교
I_range = np.linspace(0, 30, 30)
rates_if, rates_lif = [], []

for I in I_range:
    _, _, sp = simulate_if(I, duration=500)
    rates_if.append(len(sp) / 0.5)
    _, _, sp = simulate_lif(I, duration=500)
    rates_lif.append(len(sp) / 0.5)

axes[1, 0].plot(I_range, rates_if, 'b-', linewidth=2, label='IF')
axes[1, 0].plot(I_range, rates_lif, 'r-', linewidth=2, label='LIF')
axes[1, 0].set_xlabel('Current (pA)')
axes[1, 0].set_ylabel('Firing Rate (Hz)')
axes[1, 0].set_title('F-I Curves')
axes[1, 0].legend()

# tau_m의 영향
tau_values = [10, 20, 40]
for tau in tau_values:
    rates = []
    for I in I_range:
        _, _, sp = simulate_lif(I, duration=500, tau_m=tau)
        rates.append(len(sp) / 0.5)
    axes[1, 1].plot(I_range, rates, linewidth=2, label=f'τm={tau}ms')

axes[1, 1].set_xlabel('Current (pA)')
axes[1, 1].set_ylabel('Firing Rate (Hz)')
axes[1, 1].set_title('Effect of Membrane Time Constant')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

---

## 3. Izhikevich 모델

### 3.1 개념

**2개의 변수**로 H-H의 다양한 발화 패턴을 재현:

$$\frac{dv}{dt} = 0.04v^2 + 5v + 140 - u + I$$
$$\frac{du}{dt} = a(bv - u)$$

**발화 조건**: $v \geq 30$ → $v \leftarrow c$, $u \leftarrow u + d$

### 3.2 파라미터와 발화 패턴

| 패턴 | a | b | c | d |
|------|---|---|---|---|
| Regular Spiking (RS) | 0.02 | 0.2 | -65 | 8 |
| Intrinsic Bursting (IB) | 0.02 | 0.2 | -55 | 4 |
| Chattering (CH) | 0.02 | 0.2 | -50 | 2 |
| Fast Spiking (FS) | 0.1 | 0.2 | -65 | 2 |
| Low-threshold (LTS) | 0.02 | 0.25 | -65 | 2 |

### 3.3 구현

```python
def simulate_izhikevich(I_ext, duration=200, dt=0.1,
                        a=0.02, b=0.2, c=-65, d=8):
    """
    Izhikevich 모델
    """
    t = np.arange(0, duration, dt)
    v = np.zeros(len(t))
    u = np.zeros(len(t))
    v[0] = c
    u[0] = b * c
    spikes = []
    
    for i in range(1, len(t)):
        # 동역학
        dv = (0.04 * v[i-1]**2 + 5 * v[i-1] + 140 - u[i-1] + I_ext) * dt
        du = a * (b * v[i-1] - u[i-1]) * dt
        
        v[i] = v[i-1] + dv
        u[i] = u[i-1] + du
        
        # 발화
        if v[i] >= 30:
            spikes.append(t[i])
            v[i] = c
            u[i] = u[i] + d
    
    return t, v, u, spikes

# 다양한 발화 패턴
patterns = [
    ('Regular Spiking', 0.02, 0.2, -65, 8),
    ('Intrinsic Bursting', 0.02, 0.2, -55, 4),
    ('Chattering', 0.02, 0.2, -50, 2),
    ('Fast Spiking', 0.1, 0.2, -65, 2),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for ax, (name, a, b, c, d) in zip(axes.flat, patterns):
    t, v, u, spikes = simulate_izhikevich(10, duration=200, a=a, b=b, c=c, d=d)
    
    # v가 30을 넘으면 시각적으로 스파이크 표시
    v_plot = v.copy()
    for i, spike_t in enumerate(spikes):
        idx = int(spike_t / 0.1)
        if idx < len(v_plot):
            v_plot[idx] = 30
    
    ax.plot(t, v_plot, 'b-', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('v (mV)')
    ax.set_title(f'{name}\na={a}, b={b}, c={c}, d={d}')
    ax.set_ylim(-80, 40)

plt.suptitle('Izhikevich Model: Different Firing Patterns', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 4. 모델 비교

### 4.1 트레이드오프

```{mermaid}
flowchart TB
    subgraph 복잡도축
        direction LR
        SIMPLE[단순] --> COMPLEX[복잡]
    end
    
    subgraph 모델들
        IF[IF<br/>1var, 1eq]
        LIF[LIF<br/>1var, 1eq]
        IZH[Izhikevich<br/>2var, 2eq]
        HH[H-H<br/>4var, 4eq]
    end
    
    IF --> LIF --> IZH --> HH
```

### 4.2 정량적 비교

| 모델 | 변수 수 | 계산 비용 | 생물학적 정확도 | 발화 패턴 다양성 |
|------|--------|----------|----------------|-----------------|
| IF | 1 | ⭐ | ⭐ | ⭐ |
| LIF | 1 | ⭐ | ⭐⭐ | ⭐ |
| Izhikevich | 2 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| H-H | 4 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 📝 실습 문제

### 문제 1: LIF 해석해
LIF의 정상상태 전압과 발화 시작 전류(rheobase)를 유도하세요.

### 문제 2: Adaptive LIF
적응(adaptation)을 추가한 LIF 모델을 구현하세요.

### 문제 3: 모델 피팅
실제 뉴런 데이터에 Izhikevich 파라미터를 피팅하세요.

---

## 🔗 관련 개념

- [Hodgkin-Huxley 모델](../../concepts/hodgkin-huxley)
- [활동전위](../../concepts/action-potential)
- [스파이크 트레인](../../concepts/spike-train)

---

## 📚 참고 자료

- Gerstner & Kistler, "Spiking Neuron Models"
- Izhikevich (2003), "Simple Model of Spiking Neurons"
- Dayan & Abbott, Chapter 5

---

## ⏭️ Next

```{button-ref} ../week6/day1-synaptic-models
:color: primary

다음: W6D1 - Synaptic Models →
```
