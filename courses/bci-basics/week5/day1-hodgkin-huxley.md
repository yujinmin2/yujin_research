---
title: "W5D1 - Hodgkin-Huxley Model"
subtitle: "활동전위의 생물물리학적 모델"
---

# W5D1: Hodgkin-Huxley Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W5D1_HodgkinHuxley.ipynb)

---

## 📋 Overview

**핵심 질문**: 뉴런은 어떻게 전기 신호(활동전위)를 생성하는가?

1952년 **Hodgkin & Huxley**는 오징어 거대 축삭에서 활동전위의 이온 메커니즘을 수학적으로 모델링했습니다. (노벨상 수상)

```{mermaid}
flowchart LR
    subgraph 세포막
        CM[막 정전용량<br/>Cm]
        NA[Na⁺ 채널<br/>gNa]
        K[K⁺ 채널<br/>gK]
        L[누출 채널<br/>gL]
    end
    
    I[외부 전류] --> CM
    I --> NA
    I --> K
    I --> L
    
    CM --> V[막전위 V]
    NA --> V
    K --> V
    L --> V
```

---

## 🎯 Learning Objectives

1. **H-H 모델의 등가 회로** 이해
2. **이온 채널 동역학** (게이팅 변수) 이해
3. H-H 모델 **시뮬레이션** 구현
4. **활동전위의 메커니즘** 분석

---

## 1. 등가 회로 모델

### 1.1 세포막의 전기적 특성

| 구성요소 | 기호 | 역할 |
|----------|------|------|
| **막 정전용량** | $C_m$ | 전하 저장 |
| **Na⁺ 컨덕턴스** | $g_{Na}$ | 탈분극 |
| **K⁺ 컨덕턴스** | $g_K$ | 재분극 |
| **누출 컨덕턴스** | $g_L$ | 휴지 전위 유지 |

### 1.2 핵심 방정식

$$C_m \frac{dV}{dt} = I - I_{Na} - I_K - I_L$$

각 이온 전류:

$$I_{Na} = g_{Na} \cdot m^3 h \cdot (V - E_{Na})$$
$$I_K = g_K \cdot n^4 \cdot (V - E_K)$$
$$I_L = g_L \cdot (V - E_L)$$

```{mermaid}
flowchart TB
    subgraph 전류흐름
        I_EXT[외부 전류 I]
        I_CAP[정전용량 전류<br/>Cm dV/dt]
        I_NA[Na⁺ 전류<br/>gNa m³h ΔV-ENa Δ]
        I_K[K⁺ 전류<br/>gK n⁴ ΔV-EK Δ]
        I_L[누출 전류<br/>gL ΔV-EL Δ]
    end
    
    I_EXT --> I_CAP
    I_EXT --> I_NA
    I_EXT --> I_K
    I_EXT --> I_L
```

---

## 2. 게이팅 변수 (Gating Variables)

### 2.1 개념

이온 채널은 **게이트**에 의해 열리고 닫힙니다.

| 변수 | 채널 | 역할 |
|------|------|------|
| **m** | Na⁺ | 활성화 (빠름) |
| **h** | Na⁺ | 비활성화 (느림) |
| **n** | K⁺ | 활성화 (중간) |

### 2.2 게이팅 동역학

$$\frac{dx}{dt} = \alpha_x(V)(1-x) - \beta_x(V)x$$

또는 동등하게:

$$\tau_x(V) \frac{dx}{dt} = x_\infty(V) - x$$

여기서:
- $x_\infty(V) = \frac{\alpha_x}{\alpha_x + \beta_x}$ : 정상상태 값
- $\tau_x(V) = \frac{1}{\alpha_x + \beta_x}$ : 시간 상수

### 2.3 rate 함수

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# H-H 파라미터 (오징어 거대 축삭, 6.3°C)
C_m = 1.0      # μF/cm²
g_Na = 120.0   # mS/cm²
g_K = 36.0     # mS/cm²
g_L = 0.3      # mS/cm²
E_Na = 50.0    # mV
E_K = -77.0    # mV
E_L = -54.4    # mV

def alpha_m(V):
    """Na⁺ 활성화 rate"""
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)

def alpha_h(V):
    """Na⁺ 비활성화 rate"""
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V):
    return 1.0 / (1 + np.exp(-(V + 35) / 10))

def alpha_n(V):
    """K⁺ 활성화 rate"""
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)

# 정상상태 및 시간상수
def steady_state(V):
    m_inf = alpha_m(V) / (alpha_m(V) + beta_m(V))
    h_inf = alpha_h(V) / (alpha_h(V) + beta_h(V))
    n_inf = alpha_n(V) / (alpha_n(V) + beta_n(V))
    return m_inf, h_inf, n_inf

def time_constants(V):
    tau_m = 1 / (alpha_m(V) + beta_m(V))
    tau_h = 1 / (alpha_h(V) + beta_h(V))
    tau_n = 1 / (alpha_n(V) + beta_n(V))
    return tau_m, tau_h, tau_n

# 시각화
V_range = np.linspace(-80, 50, 200)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 정상상태 값
m_inf, h_inf, n_inf = [], [], []
for V in V_range:
    m, h, n = steady_state(V)
    m_inf.append(m)
    h_inf.append(h)
    n_inf.append(n)

axes[0, 0].plot(V_range, m_inf, 'r-', linewidth=2, label='m∞ (Na⁺ act)')
axes[0, 0].plot(V_range, h_inf, 'r--', linewidth=2, label='h∞ (Na⁺ inact)')
axes[0, 0].plot(V_range, n_inf, 'b-', linewidth=2, label='n∞ (K⁺ act)')
axes[0, 0].set_xlabel('Membrane Potential (mV)')
axes[0, 0].set_ylabel('Steady-state value')
axes[0, 0].set_title('Gating Variable Steady States')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=-65, color='gray', linestyle=':', label='Rest')

# 시간 상수
tau_m, tau_h, tau_n = [], [], []
for V in V_range:
    tm, th, tn = time_constants(V)
    tau_m.append(tm)
    tau_h.append(th)
    tau_n.append(tn)

axes[0, 1].semilogy(V_range, tau_m, 'r-', linewidth=2, label='τm (Na⁺ act)')
axes[0, 1].semilogy(V_range, tau_h, 'r--', linewidth=2, label='τh (Na⁺ inact)')
axes[0, 1].semilogy(V_range, tau_n, 'b-', linewidth=2, label='τn (K⁺ act)')
axes[0, 1].set_xlabel('Membrane Potential (mV)')
axes[0, 1].set_ylabel('Time constant (ms)')
axes[0, 1].set_title('Gating Variable Time Constants')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 컨덕턴스
g_Na_eff = g_Na * np.array(m_inf)**3 * np.array(h_inf)
g_K_eff = g_K * np.array(n_inf)**4

axes[1, 0].plot(V_range, g_Na_eff, 'r-', linewidth=2, label='gNa·m³h')
axes[1, 0].plot(V_range, g_K_eff, 'b-', linewidth=2, label='gK·n⁴')
axes[1, 0].set_xlabel('Membrane Potential (mV)')
axes[1, 0].set_ylabel('Conductance (mS/cm²)')
axes[1, 0].set_title('Effective Conductances')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# I-V 관계
I_Na = g_Na_eff * (V_range - E_Na)
I_K = g_K_eff * (V_range - E_K)

axes[1, 1].plot(V_range, I_Na, 'r-', linewidth=2, label='INa')
axes[1, 1].plot(V_range, I_K, 'b-', linewidth=2, label='IK')
axes[1, 1].axhline(y=0, color='gray', linestyle='--')
axes[1, 1].set_xlabel('Membrane Potential (mV)')
axes[1, 1].set_ylabel('Current (μA/cm²)')
axes[1, 1].set_title('Ionic Currents (steady-state)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 3. H-H 모델 시뮬레이션

### 3.1 전체 구현

```python
def hodgkin_huxley(y, t, I_ext):
    """
    Hodgkin-Huxley 모델 미분방정식
    
    Parameters:
    -----------
    y : array [V, m, h, n]
    t : time
    I_ext : 외부 전류 (μA/cm²)
    """
    V, m, h, n = y
    
    # 이온 전류
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)
    
    # 막전위 변화
    dVdt = (I_ext - I_Na - I_K - I_L) / C_m
    
    # 게이팅 변수 변화
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    
    return [dVdt, dmdt, dhdt, dndt]

def simulate_hh(I_ext, duration=50, dt=0.01):
    """H-H 모델 시뮬레이션"""
    t = np.arange(0, duration, dt)
    
    # 초기 조건 (휴지 상태)
    V0 = -65.0
    m0, h0, n0 = steady_state(V0)
    y0 = [V0, m0, h0, n0]
    
    # 시뮬레이션
    solution = odeint(hodgkin_huxley, y0, t, args=(I_ext,))
    
    return t, solution

# 다양한 전류로 시뮬레이션
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

I_values = [0, 5, 10, 20]  # μA/cm²

for ax, I_ext in zip(axes.flat, I_values):
    t, sol = simulate_hh(I_ext, duration=50)
    V = sol[:, 0]
    
    ax.plot(t, V, 'b-', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')
    ax.set_title(f'I = {I_ext} μA/cm²')
    ax.set_ylim(-80, 60)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=-65, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('Hodgkin-Huxley Model: Action Potentials', fontsize=14)
plt.tight_layout()
plt.show()
```

### 3.2 활동전위 분석

```python
def analyze_action_potential():
    """활동전위의 이온 메커니즘 분석"""
    
    I_ext = 10  # 충분한 자극
    t, sol = simulate_hh(I_ext, duration=20)
    V, m, h, n = sol.T
    
    # 이온 전류 계산
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)
    
    # 컨덕턴스
    g_Na_t = g_Na * m**3 * h
    g_K_t = g_K * n**4
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # 1. 막전위
    axes[0].plot(t, V, 'k-', linewidth=2)
    axes[0].set_ylabel('V (mV)')
    axes[0].set_title('Action Potential Mechanism')
    axes[0].axhline(y=-65, color='gray', linestyle='--', alpha=0.5)
    
    # 위상 표시
    peak_idx = np.argmax(V)
    axes[0].annotate('Peak', xy=(t[peak_idx], V[peak_idx]), 
                     xytext=(t[peak_idx]+2, V[peak_idx]+5),
                     arrowprops=dict(arrowstyle='->', color='red'))
    
    # 2. 게이팅 변수
    axes[1].plot(t, m, 'r-', linewidth=2, label='m (Na⁺ act)')
    axes[1].plot(t, h, 'r--', linewidth=2, label='h (Na⁺ inact)')
    axes[1].plot(t, n, 'b-', linewidth=2, label='n (K⁺ act)')
    axes[1].set_ylabel('Gating')
    axes[1].legend(loc='right')
    axes[1].set_ylim(0, 1)
    
    # 3. 컨덕턴스
    axes[2].plot(t, g_Na_t, 'r-', linewidth=2, label='gNa')
    axes[2].plot(t, g_K_t, 'b-', linewidth=2, label='gK')
    axes[2].set_ylabel('g (mS/cm²)')
    axes[2].legend(loc='right')
    
    # 4. 이온 전류
    axes[3].plot(t, -I_Na, 'r-', linewidth=2, label='-INa (inward)')
    axes[3].plot(t, -I_K, 'b-', linewidth=2, label='-IK (outward)')
    axes[3].axhline(y=0, color='gray', linestyle='--')
    axes[3].set_xlabel('Time (ms)')
    axes[3].set_ylabel('Current (μA/cm²)')
    axes[3].legend(loc='right')
    
    plt.tight_layout()
    plt.show()

analyze_action_potential()
```

---

## 4. 활동전위 메커니즘

### 4.1 단계별 설명

```{mermaid}
flowchart TB
    subgraph 1_휴지["1. 휴지 상태"]
        R1[V = -65mV]
        R2[m↓ h↑ n↓]
    end
    
    subgraph 2_탈분극["2. 탈분극"]
        D1[자극 → V↑]
        D2[m↑ 빠르게]
        D3[Na⁺ 유입 → V↑↑]
    end
    
    subgraph 3_피크["3. 피크"]
        P1[V ≈ +40mV]
        P2[h↓ 시작]
        P3[n↑ 시작]
    end
    
    subgraph 4_재분극["4. 재분극"]
        RE1[Na⁺ 비활성화]
        RE2[K⁺ 유출]
        RE3[V↓]
    end
    
    subgraph 5_과분극["5. 과분극"]
        H1[V < -65mV]
        H2[n 아직 높음]
        H3[불응기]
    end
    
    1_휴지 --> 2_탈분극 --> 3_피크 --> 4_재분극 --> 5_과분극 --> 1_휴지
```

### 4.2 위상 평면 분석

```python
def phase_plane_analysis():
    """V-n 위상 평면 분석"""
    
    I_ext = 10
    t, sol = simulate_hh(I_ext, duration=30)
    V, m, h, n = sol.T
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # V-n 위상 평면
    axes[0].plot(V, n, 'b-', linewidth=1.5)
    axes[0].plot(V[0], n[0], 'go', markersize=10, label='Start')
    axes[0].set_xlabel('V (mV)')
    axes[0].set_ylabel('n')
    axes[0].set_title('Phase Plane: V vs n')
    axes[0].legend()
    
    # 방향 화살표
    for i in range(0, len(t), 200):
        if i + 50 < len(t):
            axes[0].annotate('', xy=(V[i+50], n[i+50]), xytext=(V[i], n[i]),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
    
    # V-m 위상 평면
    axes[1].plot(V, m**3 * h, 'r-', linewidth=1.5)
    axes[1].set_xlabel('V (mV)')
    axes[1].set_ylabel('m³h')
    axes[1].set_title('Phase Plane: V vs m³h')
    
    plt.tight_layout()
    plt.show()

phase_plane_analysis()
```

---

## 📝 실습 문제

### 문제 1: F-I 곡선
다양한 전류 값에서 발화율을 측정하여 F-I(frequency-current) 곡선을 그리세요.

### 문제 2: 온도 효과
Q10 factor를 적용하여 온도에 따른 활동전위 변화를 시뮬레이션하세요.

### 문제 3: 약물 효과
TTX(Na⁺ 채널 차단)와 TEA(K⁺ 채널 차단)의 효과를 시뮬레이션하세요.

---

## 🔗 관련 개념

- [활동전위](../../concepts/action-potential)
- [뉴런](../../concepts/neuron)
- [LIF 모델](../../concepts/lif-model)

---

## 📚 참고 자료

- Hodgkin & Huxley (1952): Original papers
- Dayan & Abbott, Chapter 5-6
- Izhikevich, "Dynamical Systems in Neuroscience"

---

## ⏭️ Next

```{button-ref} day2-neuron-models
:color: primary

다음: W5D2 - Simplified Neuron Models →
```
