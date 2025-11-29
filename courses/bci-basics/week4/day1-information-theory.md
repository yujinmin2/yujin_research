---
title: "W4D1 - Information Theory"
subtitle: "정보 이론과 신경 코딩"
---

# W4D1: Information Theory

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W4D1_InformationTheory.ipynb)

---

## 📋 Overview

**핵심 질문**: 뉴런은 얼마나 많은 정보를 전달하는가?

**정보 이론**은 불확실성과 정보 전달을 수학적으로 정량화하는 프레임워크입니다.

```{mermaid}
flowchart LR
    S[자극 S] --> N[뉴런] --> R[반응 R]
    
    H_S["H(S)<br/>자극 엔트로피"]
    H_R["H(R)<br/>반응 엔트로피"]
    I["I(S;R)<br/>상호정보량"]
    
    S -.-> H_S
    R -.-> H_R
    S -.-> I
    R -.-> I
```

---

## 🎯 Learning Objectives

1. **엔트로피**의 개념과 계산 방법 이해
2. **상호정보량**으로 정보 전달량 측정
3. **채널 용량**의 의미 이해
4. 신경 시스템의 **정보 처리 효율** 분석

---

## 1. 엔트로피 (Entropy)

### 1.1 Shannon 엔트로피

**엔트로피**는 확률 분포의 불확실성 또는 "놀라움"의 평균을 측정합니다.

$$H(X) = -\sum_{x} P(x) \log_2 P(x)$$

단위: **bits**

```{mermaid}
flowchart LR
    subgraph 낮은엔트로피
        A[●●●●○<br/>P=0.8, 0.2]
        A --> H1["H = 0.72 bits"]
    end
    
    subgraph 높은엔트로피
        B[●●○○<br/>P=0.5, 0.5]
        B --> H2["H = 1.0 bit"]
    end
```

### 1.2 구현

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    """Shannon 엔트로피 계산"""
    p = np.array(p)
    p = p[p > 0]  # 0 제외 (log 정의)
    return -np.sum(p * np.log2(p))

# 예시: 이진 엔트로피
p_range = np.linspace(0.01, 0.99, 100)
H_binary = [entropy([p, 1-p]) for p in p_range]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(p_range, H_binary, 'b-', linewidth=2)
plt.xlabel('P(X=1)')
plt.ylabel('H(X) (bits)')
plt.title('Binary Entropy Function')
plt.axhline(y=1, color='red', linestyle='--', label='Max = 1 bit')
plt.axvline(x=0.5, color='gray', linestyle='--')
plt.legend()
plt.grid(True, alpha=0.3)

# 예시들
plt.subplot(1, 2, 2)
examples = [
    ([1.0], "확실"),
    ([0.5, 0.5], "동전"),
    ([1/6]*6, "주사위"),
    ([1/52]*52, "카드")
]

names = []
values = []
for probs, name in examples:
    H = entropy(probs)
    names.append(f"{name}\n({len(probs)}개)")
    values.append(H)

plt.bar(names, values, color=['green', 'blue', 'orange', 'red'], edgecolor='black')
plt.ylabel('Entropy (bits)')
plt.title('다양한 분포의 엔트로피')
for i, v in enumerate(values):
    plt.text(i, v + 0.1, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.show()
```

### 1.3 조건부 엔트로피

$$H(Y|X) = \sum_x P(x) H(Y|X=x)$$

"X를 알 때 Y의 남은 불확실성"

---

## 2. 상호정보량 (Mutual Information)

### 2.1 정의

**상호정보량 I(X;Y)**는 X를 알 때 Y에 대한 불확실성 감소량입니다.

$$I(X;Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)$$

또는:

$$I(X;Y) = \sum_{x,y} P(x,y) \log_2 \frac{P(x,y)}{P(x)P(y)}$$

```{mermaid}
flowchart TB
    subgraph 벤다이어그램
        HX["H(X)"]
        HY["H(Y)"]
        I["I(X;Y)<br/>공유 정보"]
    end
    
    HX --- I
    I --- HY
```

### 2.2 신경과학에서의 의미

$$I(S;R) = \text{자극 S와 반응 R이 공유하는 정보량}$$

- **높은 I(S;R)**: 반응이 자극을 잘 구분
- **낮은 I(S;R)**: 반응이 노이즈에 묻힘

### 2.3 구현

```python
def mutual_information(P_joint):
    """
    상호정보량 계산
    
    Parameters:
    -----------
    P_joint : 2D array - 결합 확률 분포 P(X,Y)
    
    Returns:
    --------
    I : float - 상호정보량 (bits)
    """
    # 주변 분포
    P_x = np.sum(P_joint, axis=1)
    P_y = np.sum(P_joint, axis=0)
    
    # 상호정보량 계산
    I = 0
    for i in range(len(P_x)):
        for j in range(len(P_y)):
            if P_joint[i, j] > 0:
                I += P_joint[i, j] * np.log2(
                    P_joint[i, j] / (P_x[i] * P_y[j])
                )
    return I

# 예시: 뉴런의 정보 전달
def simulate_neuron_channel(noise_level=0.1):
    """
    노이즈가 있는 신경 채널 시뮬레이션
    
    2개 자극 (S=0, S=1) → 2개 반응 (R=low, R=high)
    """
    # 결합 확률 P(S, R)
    # 노이즈 없으면: S=0 → R=low, S=1 → R=high
    P_joint = np.array([
        [0.5 - noise_level/2, noise_level/2],      # S=0
        [noise_level/2, 0.5 - noise_level/2]       # S=1
    ])
    
    return P_joint

# 노이즈 레벨에 따른 상호정보량
noise_levels = np.linspace(0, 0.5, 50)
MI_values = []

for noise in noise_levels:
    P_joint = simulate_neuron_channel(noise)
    MI_values.append(mutual_information(P_joint))

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(noise_levels, MI_values, 'b-', linewidth=2)
plt.xlabel('Noise Level')
plt.ylabel('I(S;R) (bits)')
plt.title('Mutual Information vs Noise')
plt.axhline(y=1, color='red', linestyle='--', label='Channel Capacity')
plt.legend()
plt.grid(True, alpha=0.3)

# 결합 분포 시각화
plt.subplot(1, 2, 2)
P_low = simulate_neuron_channel(0.1)
P_high = simulate_neuron_channel(0.4)

im = plt.imshow(P_low, cmap='Blues', aspect='auto')
plt.colorbar(im, label='P(S,R)')
plt.xticks([0, 1], ['R=low', 'R=high'])
plt.yticks([0, 1], ['S=0', 'S=1'])
plt.xlabel('Response')
plt.ylabel('Stimulus')
plt.title(f'Joint Distribution\nI(S;R) = {mutual_information(P_low):.2f} bits')

plt.tight_layout()
plt.show()
```

---

## 3. 신경 코드의 정보량

### 3.1 스파이크 트레인의 엔트로피

스파이크 트레인을 이진 시퀀스로 근사:

$$H(R) = -\sum_r P(r) \log_2 P(r)$$

```python
def spike_train_entropy(spike_trains, bin_size=0.01, duration=1.0):
    """
    스파이크 트레인의 엔트로피 추정
    
    Parameters:
    -----------
    spike_trains : list of arrays - 여러 trial의 스파이크 시간
    bin_size : float - 시간 빈 크기 (초)
    """
    n_bins = int(duration / bin_size)
    
    # 각 trial을 이진 벡터로 변환
    binary_patterns = []
    for spikes in spike_trains:
        pattern = np.zeros(n_bins, dtype=int)
        for spike in spikes:
            bin_idx = int(spike / bin_size)
            if 0 <= bin_idx < n_bins:
                pattern[bin_idx] = 1
        binary_patterns.append(tuple(pattern))
    
    # 패턴 빈도 계산
    from collections import Counter
    pattern_counts = Counter(binary_patterns)
    n_trials = len(spike_trains)
    
    # 엔트로피 계산
    H = 0
    for count in pattern_counts.values():
        p = count / n_trials
        if p > 0:
            H -= p * np.log2(p)
    
    return H

# 시뮬레이션
np.random.seed(42)
n_trials = 100
duration = 0.1
rate = 50

spike_trains = []
for _ in range(n_trials):
    n_spikes = np.random.poisson(rate * duration)
    spikes = np.sort(np.random.uniform(0, duration, n_spikes))
    spike_trains.append(spikes)

H = spike_train_entropy(spike_trains, bin_size=0.01, duration=duration)
print(f"스파이크 트레인 엔트로피: {H:.2f} bits")
print(f"최대 가능 엔트로피 (10 bins): {10:.2f} bits")
```

### 3.2 정보 전달률 (Information Rate)

$$\text{Information Rate} = \frac{I(S;R)}{T} \quad \text{(bits/s)}$$

---

## 4. 채널 용량 (Channel Capacity)

### 4.1 정의

**채널 용량**은 채널을 통해 전달할 수 있는 최대 정보량입니다.

$$C = \max_{P(X)} I(X;Y)$$

### 4.2 신경 채널의 한계

| 시스템 | 추정 용량 |
|--------|----------|
| 단일 뉴런 | ~100 bits/s |
| 시신경 | ~10 Mbits/s |
| 전체 뇌 | 제한적 (주의 집중) |

---

## 📝 실습 문제

### 문제 1: 튜닝 커브와 정보량
뉴런의 튜닝 폭이 정보 전달량에 미치는 영향을 분석하세요.

### 문제 2: 최적 코딩
동일한 정보량을 전달하는 데 필요한 최소 발화율을 계산하세요.

### 문제 3: Population 정보량
뉴런 집단의 상호정보량이 개별 뉴런의 합보다 큰지/작은지 분석하세요.

---

## 🔗 관련 개념

- [엔트로피](../../concepts/entropy)
- [상호정보량](../../concepts/mutual-information)
- [Rate Coding](../../concepts/rate-coding)

---

## 📚 참고 자료

- Cover & Thomas, "Elements of Information Theory"
- Rieke et al., "Spikes" Chapter 2
- Borst & Theunissen, "Information Theory and Neural Coding"

---

## ⏭️ Next

```{button-ref} day2-neural-coding
:color: primary

다음: W4D2 - Neural Coding →
```
