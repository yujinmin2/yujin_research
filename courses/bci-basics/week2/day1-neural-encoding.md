---
title: "W2D1 - Neural Encoding Models"
subtitle: "신경 인코딩 모델: 뇌는 어떻게 정보를 표현하는가?"
---

# W2D1: Neural Encoding Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W2D1_NeuralEncoding.ipynb)

---

## 📋 Overview

**핵심 질문**: 뉴런은 외부 세계의 정보를 어떻게 표현(인코딩)하는가?

신경 인코딩(Neural Encoding)은 **자극 → 신경 반응**의 관계를 연구합니다.

```
┌─────────────┐     인코딩      ┌─────────────┐
│   자극 s    │ ─────────────▶ │   반응 r    │
│ (Stimulus)  │                │ (Response)  │
└─────────────┘                └─────────────┘
    빛, 소리,                    스파이크,
    움직임 등                     발화율 등
```

---

## 🎯 Learning Objectives

이 강의를 마치면 다음을 할 수 있습니다:

1. **Tuning curve**의 개념을 이해하고 구현할 수 있다
2. **Rate coding**과 **Temporal coding**의 차이를 설명할 수 있다
3. **Linear-Nonlinear (LN) 모델**을 이해하고 적용할 수 있다
4. **Population coding**의 원리를 이해할 수 있다

---

## 📚 배경 지식

신경과학에서 가장 기본적인 질문 중 하나는 "뇌가 어떻게 외부 세계를 표현하는가?"입니다.

예를 들어:
- 👁️ 시각: 특정 방향의 선을 볼 때 V1 뉴런이 어떻게 반응하는가?
- 👂 청각: 특정 주파수의 소리에 청각 피질이 어떻게 반응하는가?
- 🖐️ 촉각: 피부의 특정 위치 자극에 체성감각 피질이 어떻게 반응하는가?

---

## 1. Tuning Curves (튜닝 커브)

### 1.1 개념

**튜닝 커브(Tuning Curve)**는 자극의 특정 특성과 뉴런의 발화율 사이의 관계를 나타내는 함수입니다.

> 💡 **핵심 개념**: 튜닝 커브 = f(자극 특성) → 발화율. 뉴런이 "선호하는" 자극에서 가장 높은 발화율을 보입니다.

### 1.2 방향 튜닝 (Orientation Tuning)

시각 피질 V1의 뉴런은 특정 방향의 막대(bar)에 선택적으로 반응합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_tuning_curve(theta, preferred_theta, amplitude=50, width=30):
    """
    가우시안 형태의 튜닝 커브
    
    Parameters:
    -----------
    theta : array - 자극 방향 (도)
    preferred_theta : float - 뉴런이 선호하는 방향 (도)
    amplitude : float - 최대 발화율 (Hz)
    width : float - 튜닝 폭 (도)
    """
    diff = np.minimum(np.abs(theta - preferred_theta), 
                      360 - np.abs(theta - preferred_theta))
    return amplitude * np.exp(-0.5 * (diff / width)**2)

# 자극 방향 범위
theta = np.linspace(0, 360, 361)

# 4개의 뉴런, 각각 다른 선호 방향
preferred_directions = [0, 45, 90, 135]

plt.figure(figsize=(10, 5))
for pref in preferred_directions:
    rate = gaussian_tuning_curve(theta, pref)
    plt.plot(theta, rate, linewidth=2, label=f'Preferred: {pref}°')

plt.xlabel('Stimulus Direction (°)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Orientation Tuning Curves in V1')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 1.3 다양한 튜닝 커브 형태

| 유형 | 형태 | 예시 |
|------|------|------|
| **Gaussian** | 종 모양 | 시각 피질 방향 선택성 |
| **Cosine** | 코사인 | 운동 피질 방향 |
| **Sigmoid** | S자형 | 강도 인코딩 |
| **Bandpass** | 대역통과 | 청각 주파수 |

---

## 2. Rate Coding vs Temporal Coding

### 2.1 두 가지 코딩 전략

| 특성 | Rate Coding | Temporal Coding |
|------|-------------|-----------------|
| **정보 표현** | 발화율 (spikes/sec) | 스파이크 타이밍 |
| **시간 창** | 100ms ~ 1s | 1ms ~ 10ms |
| **정보량** | 상대적으로 적음 | 높은 정보량 가능 |
| **예시** | V1 방향 선택성 | 청각 위치 파악 |
| **안정성** | 노이즈에 강함 | 정밀한 타이밍 필요 |

### 2.2 Rate Coding 예시

```python
def demonstrate_rate_coding():
    """Rate Coding: 자극 강도가 발화율로 인코딩"""
    np.random.seed(42)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    
    intensities = [10, 30, 60]  # Hz
    labels = ['약한 자극 (10 Hz)', '중간 자극 (30 Hz)', '강한 자극 (60 Hz)']
    
    for ax, rate, label in zip(axes, intensities, labels):
        # 포아송 스파이크 생성
        n_spikes = np.random.poisson(rate)
        spike_times = np.sort(np.random.uniform(0, 1, n_spikes))
        
        ax.eventplot(spike_times, lineoffsets=0, linelengths=0.8, linewidths=1.5)
        ax.set_ylabel(label)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
    
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Rate Coding: 자극 강도 → 발화율')
    plt.tight_layout()
    plt.show()

demonstrate_rate_coding()
```

### 2.3 Temporal Coding 예시

```python
def demonstrate_temporal_coding():
    """같은 발화율, 다른 타이밍 = 다른 정보"""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 4))
    
    # 패턴 1: 규칙적
    spike_times_regular = np.linspace(0.05, 0.45, 10)
    
    # 패턴 2: 버스트
    spike_times_burst = np.concatenate([
        np.array([0.05, 0.06, 0.07, 0.08, 0.09]),
        np.array([0.35, 0.36, 0.37, 0.38, 0.39])
    ])
    
    axes[0].eventplot(spike_times_regular, linewidths=2)
    axes[0].set_title('Regular Pattern (10 spikes)')
    
    axes[1].eventplot(spike_times_burst, linewidths=2, colors='red')
    axes[1].set_title('Burst Pattern (10 spikes)')
    
    for ax in axes:
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle('Temporal Coding: 같은 발화율, 다른 정보', y=1.02)
    plt.tight_layout()
    plt.show()

demonstrate_temporal_coding()
```

---

## 3. Linear-Nonlinear (LN) Model

### 3.1 모델 구조

LN 모델은 신경 인코딩의 가장 기본적인 계산 모델입니다:

```
자극 s(t) → [선형 필터 k] → 필터 출력 → [비선형 함수 f] → 발화율 r(t)
```

**수식**: r(t) = f(k * s)

### 3.2 구현

```python
def ln_model_demo():
    """Linear-Nonlinear 모델 데모"""
    np.random.seed(42)
    
    dt = 0.001  # 1ms
    t = np.arange(0, 1, dt)
    
    # 1. 자극 (백색 노이즈)
    stimulus = np.random.randn(len(t))
    
    # 2. 선형 필터 (바이페이직)
    tau = np.arange(0, 0.1, dt)
    linear_filter = (tau / 0.02) * np.exp(-tau / 0.02) - \
                    0.5 * (tau / 0.04) * np.exp(-tau / 0.04)
    linear_filter /= np.max(np.abs(linear_filter))
    
    # 3. 컨볼루션
    filtered = np.convolve(stimulus, linear_filter, mode='same')
    
    # 4. 비선형 (ReLU)
    firing_rate = 50 * np.maximum(0, filtered)
    
    # 시각화
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(t, stimulus, 'k-', linewidth=0.5)
    axes[0].set_ylabel('Stimulus')
    axes[0].set_title('Linear-Nonlinear (LN) Model')
    
    axes[1].plot(tau * 1000, linear_filter, 'b-', linewidth=2)
    axes[1].set_ylabel('Filter')
    axes[1].set_xlim(0, 100)
    
    axes[2].plot(t, filtered, 'g-', linewidth=0.8)
    axes[2].set_ylabel('Filtered')
    
    axes[3].plot(t, firing_rate, 'r-', linewidth=0.8)
    axes[3].set_ylabel('Firing Rate (Hz)')
    axes[3].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

ln_model_demo()
```

---

## 4. Population Coding

### 4.1 개념

단일 뉴런보다 **뉴런 집단(population)**의 활동을 분석하면 더 정확한 정보를 추출할 수 있습니다.

> 📌 **Population Coding의 장점**:
> - 노이즈에 강건함 (averaging)
> - 높은 정보 해상도
> - 빠른 코딩 (단일 스파이크로도 정보 전달)

### 4.2 Population Vector

```python
def population_vector_demo():
    """Population Vector를 이용한 방향 디코딩"""
    np.random.seed(42)
    
    n_neurons = 8
    preferred_directions = np.linspace(0, 315, n_neurons)
    true_direction = 60  # 실제 방향
    
    # 각 뉴런의 발화율 (코사인 튜닝)
    firing_rates = []
    for pref in preferred_directions:
        diff = np.deg2rad(true_direction - pref)
        rate = 30 * (1 + np.cos(diff)) + np.random.randn() * 5
        firing_rates.append(max(0, rate))
    
    firing_rates = np.array(firing_rates)
    
    # Population Vector 계산
    pref_rad = np.deg2rad(preferred_directions)
    px = np.sum(firing_rates * np.cos(pref_rad))
    py = np.sum(firing_rates * np.sin(pref_rad))
    decoded = np.rad2deg(np.arctan2(py, px))
    if decoded < 0:
        decoded += 360
    
    print(f"실제 방향: {true_direction}°")
    print(f"디코딩된 방향: {decoded:.1f}°")
    print(f"오차: {abs(decoded - true_direction):.1f}°")

population_vector_demo()
```

---

## 📝 실습 문제

### 문제 1: 튜닝 커브 피팅
주어진 데이터에서 뉴런의 선호 방향과 튜닝 폭을 추정하세요.

```python
directions = np.array([0, 45, 90, 135, 180, 225, 270, 315])
firing_rates = np.array([10, 35, 55, 40, 15, 8, 12, 18])
# TODO: scipy.optimize.curve_fit 사용
```

### 문제 2: Population Size와 정확도
뉴런 수가 4, 8, 16, 32개일 때 디코딩 정확도를 비교하세요.

---

## 🔗 참고 자료

- Dayan & Abbott, Chapter 1-3: Neural Encoding
- Neuromatch Academy: Encoding Models
- Hubel & Wiesel (1962): V1 방향 선택성 발견

---

## ⏭️ Next

```{button-ref} day2-spike-trains
:color: primary

다음: W2D2 - Spike Trains & Neural Code →
```
