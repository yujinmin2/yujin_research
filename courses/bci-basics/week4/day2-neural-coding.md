---
title: "W4D2 - Neural Coding"
subtitle: "신경 코딩의 원리와 효율성"
---

# W4D2: Neural Coding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W4D2_NeuralCoding.ipynb)

---

## 📋 Overview

**핵심 질문**: 뇌는 어떤 코딩 전략을 사용하며, 그것은 얼마나 효율적인가?

정보 이론 도구를 사용하여 다양한 신경 코딩 전략의 효율성을 분석합니다.

```{mermaid}
flowchart TB
    subgraph 코딩전략
        RATE[Rate Coding<br/>발화율]
        TEMP[Temporal Coding<br/>타이밍]
        POP[Population Coding<br/>집단]
        SPARSE[Sparse Coding<br/>희소]
    end
    
    subgraph 평가
        INFO[정보량]
        EFF[효율성]
        ROB[강건성]
    end
    
    RATE --> INFO
    TEMP --> INFO
    POP --> INFO
    SPARSE --> EFF
```

---

## 🎯 Learning Objectives

1. **다양한 신경 코딩 전략** 비교
2. **Fisher Information**으로 코딩 정밀도 분석
3. **효율적 코딩 가설** 이해
4. **희소 코딩**의 장점 분석

---

## 1. 코딩 전략 비교

### 1.1 Rate vs Temporal Coding

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_coding_strategies():
    """Rate coding vs Temporal coding 정보량 비교"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 시뮬레이션 파라미터
    duration = 0.5  # 500ms
    
    # === Rate Coding ===
    # 발화율이 자극 강도를 인코딩
    rates = [10, 30, 50, 70]  # Hz
    
    np.random.seed(42)
    for i, rate in enumerate(rates):
        n_spikes = np.random.poisson(rate * duration)
        spike_times = np.sort(np.random.uniform(0, duration, n_spikes))
        axes[0, 0].eventplot(spike_times, lineoffsets=i, linewidths=1.5)
    
    axes[0, 0].set_yticks(range(4))
    axes[0, 0].set_yticklabels([f'{r} Hz' for r in rates])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_title('Rate Coding\n발화율 = 자극 강도')
    
    # Rate coding 정보량
    # 발화율이 4개 레벨 → log2(4) = 2 bits
    axes[0, 1].bar(['4 levels', '8 levels', '16 levels'], 
                   [2, 3, 4], color='steelblue', edgecolor='black')
    axes[0, 1].set_ylabel('Information (bits)')
    axes[0, 1].set_title('Rate Coding 정보량\nI = log2(levels)')
    
    # === Temporal Coding ===
    # 같은 발화율, 다른 패턴
    rate = 40
    patterns = [
        np.array([0.05, 0.06, 0.07, 0.3, 0.31, 0.32]),  # 버스트
        np.linspace(0.05, 0.45, 6),                       # 규칙적
        np.array([0.1, 0.15, 0.25, 0.35, 0.4, 0.45]),    # 불규칙
    ]
    labels = ['Burst', 'Regular', 'Irregular']
    
    for i, (pattern, label) in enumerate(zip(patterns, labels)):
        axes[1, 0].eventplot(pattern, lineoffsets=i, linewidths=1.5)
    
    axes[1, 0].set_yticks(range(3))
    axes[1, 0].set_yticklabels(labels)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Temporal Coding\n같은 발화율, 다른 정보')
    
    # Temporal coding 정보량
    # 시간 해상도에 따라 급격히 증가
    dt_values = [100, 50, 10, 5, 1]  # ms
    info_values = []
    for dt in dt_values:
        n_bins = int(500 / dt)  # 500ms 동안
        max_info = n_bins  # 각 빈이 0 또는 1
        info_values.append(min(max_info, 20))  # 제한
    
    axes[1, 1].plot(dt_values, info_values, 'ro-', markersize=10)
    axes[1, 1].set_xlabel('Time Resolution (ms)')
    axes[1, 1].set_ylabel('Max Information (bits)')
    axes[1, 1].set_title('Temporal Coding 정보량\n시간 해상도에 따라 증가')
    axes[1, 1].invert_xaxis()
    
    plt.tight_layout()
    plt.show()

compare_coding_strategies()
```

### 1.2 정보 전달 비교

| 코딩 전략 | 정보 용량 | 장점 | 단점 |
|----------|----------|------|------|
| **Rate** | 낮음 (~3-4 bits) | 노이즈 강건 | 느림 |
| **Temporal** | 높음 (~10+ bits) | 빠름, 고용량 | 노이즈 민감 |
| **Population** | 매우 높음 | 확장 가능 | 에너지 비용 |

---

## 2. Fisher Information

### 2.1 정의

**Fisher Information**은 추정의 정밀도를 나타내는 지표입니다.

$$J(s) = E\left[\left(\frac{\partial}{\partial s} \log P(r|s)\right)^2\right]$$

**Cramér-Rao Bound**: 어떤 추정기도 이 한계보다 좋을 수 없음

$$\text{Var}(\hat{s}) \geq \frac{1}{J(s)}$$

### 2.2 튜닝 커브와 Fisher Information

튜닝 커브가 가파를수록 Fisher Information이 높습니다.

```python
def fisher_information_demo():
    """튜닝 커브 특성과 Fisher Information"""
    
    s = np.linspace(0, 180, 200)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 넓은 튜닝 vs 좁은 튜닝
    widths = [60, 30, 15]
    colors = ['blue', 'green', 'red']
    
    for width, color in zip(widths, colors):
        # 튜닝 커브 (가우시안)
        f = 50 * np.exp(-0.5 * ((s - 90) / width)**2) + 5
        axes[0, 0].plot(s, f, color=color, linewidth=2, label=f'σ={width}°')
        
        # 튜닝 커브의 기울기 (f')
        df = np.gradient(f, s[1] - s[0])
        axes[0, 1].plot(s, df, color=color, linewidth=2)
        
        # Fisher Information J(s) = (f')^2 / f (포아송 노이즈 가정)
        J = df**2 / (f + 1e-10)
        axes[1, 0].plot(s, J, color=color, linewidth=2)
    
    axes[0, 0].set_xlabel('Stimulus (°)')
    axes[0, 0].set_ylabel('Firing Rate (Hz)')
    axes[0, 0].set_title('Tuning Curves')
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Stimulus (°)')
    axes[0, 1].set_ylabel("f'(s)")
    axes[0, 1].set_title('Tuning Curve Slope')
    axes[0, 1].axhline(y=0, color='gray', linestyle='--')
    
    axes[1, 0].set_xlabel('Stimulus (°)')
    axes[1, 0].set_ylabel('J(s)')
    axes[1, 0].set_title('Fisher Information\n좁은 튜닝 → 높은 정밀도')
    
    # Population Fisher Information
    # 여러 뉴런의 Fisher Information 합
    n_neurons = [4, 8, 16, 32]
    total_J = []
    
    for N in n_neurons:
        preferred = np.linspace(0, 180, N)
        J_sum = np.zeros_like(s)
        for pref in preferred:
            f = 50 * np.exp(-0.5 * ((s - pref) / 30)**2) + 5
            df = np.gradient(f, s[1] - s[0])
            J_sum += df**2 / (f + 1e-10)
        total_J.append(np.mean(J_sum))
    
    axes[1, 1].plot(n_neurons, total_J, 'ko-', markersize=10, linewidth=2)
    axes[1, 1].set_xlabel('Number of Neurons')
    axes[1, 1].set_ylabel('Total Fisher Information')
    axes[1, 1].set_title('Population Size vs Information\n선형 증가')
    
    plt.tight_layout()
    plt.show()

fisher_information_demo()
```

---

## 3. 효율적 코딩 가설 (Efficient Coding)

### 3.1 개념

**효율적 코딩 가설**: 감각 시스템은 자연 환경의 통계에 적응하여 정보 전달을 최대화하도록 진화했다.

```{mermaid}
flowchart LR
    ENV[자연 환경<br/>통계 구조] --> ADAPT[신경 적응]
    ADAPT --> OPT[최적 코딩<br/>정보 최대화]
    
    subgraph 예시
        RETINA[망막<br/>명암 대비]
        V1[V1<br/>방향/엣지]
        IT[IT<br/>물체]
    end
```

### 3.2 히스토그램 균등화

```python
def efficient_coding_demo():
    """효율적 코딩: 입력 분포에 적응"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 자연 영상의 밝기 분포 (비균일)
    np.random.seed(42)
    natural_dist = np.concatenate([
        np.random.normal(0.3, 0.1, 3000),
        np.random.normal(0.7, 0.15, 1000)
    ])
    natural_dist = np.clip(natural_dist, 0, 1)
    
    axes[0, 0].hist(natural_dist, bins=50, density=True, alpha=0.7, 
                    color='gray', edgecolor='black')
    axes[0, 0].set_xlabel('Luminance')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_title('Natural Image Statistics\n(비균일 분포)')
    
    # 선형 인코딩 (비효율적)
    def linear_encoding(x):
        return x
    
    linear_response = linear_encoding(natural_dist)
    axes[0, 1].hist(linear_response, bins=50, density=True, alpha=0.7,
                    color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Neural Response')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_title('Linear Encoding\n(정보 손실)')
    
    # 효율적 인코딩 (히스토그램 균등화)
    from scipy.stats import rankdata
    def efficient_encoding(x):
        return rankdata(x) / len(x)
    
    efficient_response = efficient_encoding(natural_dist)
    axes[1, 0].hist(efficient_response, bins=50, density=True, alpha=0.7,
                    color='green', edgecolor='black')
    axes[1, 0].set_xlabel('Neural Response')
    axes[1, 0].set_ylabel('Probability')
    axes[1, 0].set_title('Efficient Encoding\n(균일 분포 → 최대 엔트로피)')
    
    # 인코딩 함수 비교
    x_sorted = np.sort(natural_dist)
    axes[1, 1].plot(x_sorted, linear_encoding(x_sorted), 'b-', 
                    linewidth=2, label='Linear')
    axes[1, 1].plot(x_sorted, np.linspace(0, 1, len(x_sorted)), 'g-',
                    linewidth=2, label='Efficient (CDF)')
    axes[1, 1].set_xlabel('Input (Luminance)')
    axes[1, 1].set_ylabel('Output (Response)')
    axes[1, 1].set_title('Encoding Functions')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 정보량 비교
    def entropy_from_hist(data, bins=50):
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    H_linear = entropy_from_hist(linear_response)
    H_efficient = entropy_from_hist(efficient_response)
    
    print(f"Linear encoding entropy: {H_linear:.2f} bits")
    print(f"Efficient encoding entropy: {H_efficient:.2f} bits")
    print(f"Max possible (uniform): {np.log2(50):.2f} bits")

efficient_coding_demo()
```

---

## 4. 희소 코딩 (Sparse Coding)

### 4.1 개념

**희소 코딩**: 주어진 시간에 소수의 뉴런만 활성화되는 표현 방식

| 특성 | 밀집 코딩 | 희소 코딩 |
|------|----------|----------|
| 활성 뉴런 | 다수 | 소수 |
| 에너지 효율 | 낮음 | 높음 |
| 표현 용량 | 낮음 | 높음 |
| 예시 | 초기 감각 | 해마, IT 피질 |

### 4.2 표현 용량

```python
def sparse_coding_capacity():
    """희소 코딩의 표현 용량"""
    
    N = 1000  # 총 뉴런 수
    K_values = range(1, 101)  # 활성 뉴런 수
    
    # 조합의 수: C(N, K)
    from scipy.special import comb
    capacities = [comb(N, K, exact=False) for K in K_values]
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(K_values, capacities, 'b-', linewidth=2)
    plt.xlabel('Active Neurons (K)')
    plt.ylabel('Number of Patterns')
    plt.title(f'Sparse Coding Capacity (N={N})')
    plt.axvline(x=N//2, color='red', linestyle='--', label=f'K=N/2 (max)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sparsity = np.array(K_values) / N
    info_bits = np.log2(np.array(capacities) + 1)
    plt.plot(sparsity * 100, info_bits, 'g-', linewidth=2)
    plt.xlabel('Sparsity (%)')
    plt.ylabel('Information (bits)')
    plt.title('Information vs Sparsity')
    
    plt.tight_layout()
    plt.show()
    
    # 최적 희소성
    optimal_K = K_values[np.argmax(capacities)]
    print(f"최적 활성 뉴런 수: {optimal_K} / {N}")
    print(f"최적 희소성: {optimal_K/N*100:.1f}%")

sparse_coding_capacity()
```

---

## 📝 실습 문제

### 문제 1: 최적 튜닝 폭
주어진 자극 범위에서 Fisher Information을 최대화하는 튜닝 폭을 찾으세요.

### 문제 2: 효율적 코딩
자연 영상 데이터셋에서 최적의 인코딩 함수를 학습하세요.

### 문제 3: 희소성과 노이즈
희소 코딩이 노이즈에 대한 강건성에 미치는 영향을 분석하세요.

---

## 🔗 관련 개념

- [엔트로피](../../concepts/entropy)
- [상호정보량](../../concepts/mutual-information)
- [튜닝 커브](../../concepts/tuning-curve)
- [Fisher Information](../../concepts/fisher-information)

---

## 📚 참고 자료

- Simoncelli & Olshausen, "Natural Image Statistics and Neural Representation"
- Barlow, "Possible Principles Underlying the Transformation of Sensory Messages"
- Olshausen & Field, "Sparse Coding of Sensory Inputs"

---

## ⏭️ Next

```{button-ref} ../week5/day1-hodgkin-huxley
:color: primary

다음: W5D1 - Hodgkin-Huxley Model →
```
