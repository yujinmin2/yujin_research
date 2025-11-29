---
title: "W3D1 - Neural Decoding"
subtitle: "신경 디코딩: 뇌 신호에서 정보 추출하기"
---

# W3D1: Neural Decoding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W3D1_NeuralDecoding.ipynb)

---

## 📋 Overview

**핵심 질문**: 신경 활동을 관찰하여 자극이나 행동을 예측할 수 있는가?

**신경 디코딩**은 인코딩의 역과정으로, 신경 반응으로부터 자극 정보를 추출합니다.

```{mermaid}
flowchart LR
    subgraph 인코딩
        S1[자극 s] --> R1[반응 r]
    end
    
    subgraph 디코딩
        R2[반응 r] --> S2[추정 ŝ]
    end
    
    style S1 fill:#3498db
    style R1 fill:#e74c3c
    style R2 fill:#e74c3c
    style S2 fill:#2ecc71
```

---

## 🎯 Learning Objectives

1. **디코딩 문제**의 정의와 중요성 이해
2. **베이지안 디코딩** 프레임워크 이해
3. **Maximum Likelihood** 추정 구현
4. **Population Vector** 디코딩 구현
5. 다양한 **디코더 성능 평가** 방법 이해

---

## 📚 배경: 왜 디코딩인가?

### 과학적 목표
- 뇌가 어떤 정보를 표현하는지 검증
- 신경 코드의 특성 이해

### 공학적 목표
- **BCI**: 뇌 신호로 기기 제어
- **신경 보철**: 감각 정보 복원

---

## 1. 베이지안 디코딩 (Bayesian Decoding)

### 1.1 베이즈 정리

> 💡 **핵심 아이디어**: 관측된 신경 반응 $r$이 주어졌을 때, 자극 $s$의 확률 분포를 계산

$$P(s|r) = \frac{P(r|s) \cdot P(s)}{P(r)}$$

```{mermaid}
flowchart TB
    subgraph 입력
        R[신경 반응 r]
        PRIOR[사전확률 P/s/]
    end
    
    subgraph 모델
        L[우도 P/r|s/<br/>튜닝 커브에서 유도]
    end
    
    subgraph 출력
        POST[사후확률 P/s|r/]
        EST[추정값 ŝ]
    end
    
    R --> L
    L --> POST
    PRIOR --> POST
    POST --> EST
```

### 1.2 구성 요소

| 요소 | 수식 | 의미 | 출처 |
|------|------|------|------|
| **우도 (Likelihood)** | $P(r\|s)$ | 자극이 s일 때 반응 r의 확률 | 튜닝 커브 |
| **사전확률 (Prior)** | $P(s)$ | 자극의 기본 분포 | 경험/가정 |
| **사후확률 (Posterior)** | $P(s\|r)$ | 반응이 r일 때 자극의 확률 | 계산 결과 |

### 1.3 포아송 우도 함수

뉴런의 발화가 포아송 과정을 따른다면:

$$P(r|s) = \prod_{i=1}^{N} \frac{f_i(s)^{r_i}}{r_i!} e^{-f_i(s)}$$

여기서 $f_i(s)$는 뉴런 $i$의 튜닝 커브

```python
import numpy as np
from scipy.stats import poisson

def poisson_log_likelihood(response, tuning_curves, stimuli):
    """
    포아송 로그 우도 계산
    
    Parameters:
    -----------
    response : array (N,) - 각 뉴런의 스파이크 수
    tuning_curves : array (N, S) - 각 뉴런의 튜닝 커브
    stimuli : array (S,) - 가능한 자극 값들
    
    Returns:
    --------
    log_likelihood : array (S,) - 각 자극에 대한 로그 우도
    """
    N_neurons = len(response)
    N_stimuli = len(stimuli)
    
    log_likelihood = np.zeros(N_stimuli)
    
    for s_idx in range(N_stimuli):
        for n in range(N_neurons):
            # 예상 발화율
            expected_rate = tuning_curves[n, s_idx]
            # 실제 반응
            observed = response[n]
            # 포아송 로그 확률
            log_likelihood[s_idx] += poisson.logpmf(observed, expected_rate + 1e-10)
    
    return log_likelihood

# 예시
np.random.seed(42)
N_neurons = 8
N_stimuli = 180

# 튜닝 커브 생성 (방향 선택성)
stimuli = np.linspace(0, 180, N_stimuli)
preferred_dirs = np.linspace(0, 160, N_neurons)
tuning_curves = np.zeros((N_neurons, N_stimuli))

for n, pref in enumerate(preferred_dirs):
    tuning_curves[n] = 30 * np.exp(-0.5 * ((stimuli - pref) / 30)**2) + 5

# 자극 = 60도일 때의 반응 시뮬레이션
true_stimulus = 60
true_idx = np.argmin(np.abs(stimuli - true_stimulus))
response = np.random.poisson(tuning_curves[:, true_idx])

# 디코딩
log_like = poisson_log_likelihood(response, tuning_curves, stimuli)
decoded_idx = np.argmax(log_like)
decoded_stimulus = stimuli[decoded_idx]

print(f"실제 자극: {true_stimulus}°")
print(f"디코딩된 자극: {decoded_stimulus:.1f}°")
```

---

## 2. Maximum Likelihood Estimation (MLE)

### 2.1 개념

**MLE**는 관측된 데이터를 가장 잘 설명하는 파라미터를 찾습니다.

$$\hat{s}_{ML} = \arg\max_s P(r|s)$$

> 📌 사전확률이 균일(uniform)하면 MAP = MLE

### 2.2 시각화

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 튜닝 커브
for n in range(N_neurons):
    axes[0, 0].plot(stimuli, tuning_curves[n], label=f'N{n+1}: {preferred_dirs[n]:.0f}°')
axes[0, 0].axvline(x=true_stimulus, color='red', linestyle='--', linewidth=2, label='True stimulus')
axes[0, 0].set_xlabel('Stimulus Direction (°)')
axes[0, 0].set_ylabel('Firing Rate (Hz)')
axes[0, 0].set_title('Tuning Curves')
axes[0, 0].legend(fontsize=8)

# 2. 관측된 반응
axes[0, 1].bar(range(N_neurons), response, color='steelblue', edgecolor='black')
axes[0, 1].set_xlabel('Neuron')
axes[0, 1].set_ylabel('Spike Count')
axes[0, 1].set_title(f'Observed Response (True: {true_stimulus}°)')
axes[0, 1].set_xticks(range(N_neurons))
axes[0, 1].set_xticklabels([f'{p:.0f}°' for p in preferred_dirs], rotation=45)

# 3. 로그 우도
axes[1, 0].plot(stimuli, log_like, 'b-', linewidth=2)
axes[1, 0].axvline(x=true_stimulus, color='red', linestyle='--', label='True')
axes[1, 0].axvline(x=decoded_stimulus, color='green', linestyle='--', label='Decoded')
axes[1, 0].set_xlabel('Stimulus Direction (°)')
axes[1, 0].set_ylabel('Log Likelihood')
axes[1, 0].set_title('Maximum Likelihood Decoding')
axes[1, 0].legend()

# 4. 사후확률 (정규화)
posterior = np.exp(log_like - np.max(log_like))
posterior = posterior / np.sum(posterior)
axes[1, 1].fill_between(stimuli, posterior, alpha=0.5, color='purple')
axes[1, 1].plot(stimuli, posterior, 'purple', linewidth=2)
axes[1, 1].axvline(x=true_stimulus, color='red', linestyle='--', label='True')
axes[1, 1].axvline(x=decoded_stimulus, color='green', linestyle='--', label='MAP')
axes[1, 1].set_xlabel('Stimulus Direction (°)')
axes[1, 1].set_ylabel('P(s|r)')
axes[1, 1].set_title('Posterior Distribution')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

---

## 3. Population Vector Decoding

### 3.1 개념

**Population Vector**는 각 뉴런의 선호 방향을 발화율로 가중 평균하여 자극을 추정합니다.

$$\vec{P} = \sum_{i=1}^{N} r_i \cdot \vec{c}_i$$

여기서:
- $r_i$: 뉴런 $i$의 발화율
- $\vec{c}_i$: 뉴런 $i$의 선호 방향 단위 벡터

```{mermaid}
flowchart TB
    subgraph 뉴런활동
        N1[N1: 선호 0°<br/>r=10] 
        N2[N2: 선호 45°<br/>r=35]
        N3[N3: 선호 90°<br/>r=25]
        N4[N4: 선호 135°<br/>r=8]
    end
    
    PV[Population Vector<br/>벡터 합산]
    
    N1 --> PV
    N2 --> PV
    N3 --> PV
    N4 --> PV
    
    PV --> EST[추정 방향: ~55°]
```

### 3.2 구현

```python
def population_vector_decode(response, preferred_directions):
    """
    Population Vector 디코딩
    
    Parameters:
    -----------
    response : array (N,) - 각 뉴런의 발화율
    preferred_directions : array (N,) - 각 뉴런의 선호 방향 (도)
    
    Returns:
    --------
    decoded_direction : float - 디코딩된 방향 (도)
    """
    # 선호 방향을 라디안으로 변환
    pref_rad = np.deg2rad(preferred_directions)
    
    # 각 뉴런의 기여를 벡터로
    x = np.sum(response * np.cos(pref_rad))
    y = np.sum(response * np.sin(pref_rad))
    
    # 방향 계산
    decoded_rad = np.arctan2(y, x)
    decoded_deg = np.rad2deg(decoded_rad)
    
    # 0-180 범위로
    if decoded_deg < 0:
        decoded_deg += 180
    
    return decoded_deg

# 테스트
pv_decoded = population_vector_decode(response, preferred_dirs)
print(f"Population Vector 디코딩: {pv_decoded:.1f}°")
print(f"실제 자극: {true_stimulus}°")
print(f"오차: {abs(pv_decoded - true_stimulus):.1f}°")
```

### 3.3 시각화 (극좌표)

```python
fig = plt.figure(figsize=(10, 5))

# 극좌표 플롯
ax = fig.add_subplot(121, projection='polar')

# 각 뉴런의 기여
colors = plt.cm.viridis(np.linspace(0, 1, N_neurons))
for n, (pref, r) in enumerate(zip(preferred_dirs, response)):
    ax.arrow(np.deg2rad(pref), 0, 0, r/np.max(response),
             head_width=0.1, head_length=0.05,
             fc=colors[n], ec='black', linewidth=0.5, alpha=0.7)

# Population Vector
ax.arrow(0, 0, np.deg2rad(pv_decoded), 0.8,
         head_width=0.15, head_length=0.08,
         fc='red', ec='darkred', linewidth=2)

# 실제 방향
ax.plot([0, np.deg2rad(true_stimulus)], [0, 1], 'g--', linewidth=2)

ax.set_title('Population Vector Decoding')

# 오차 분포 (여러 trial 시뮬레이션)
ax2 = fig.add_subplot(122)

errors = []
for _ in range(500):
    # 반응 시뮬레이션
    sim_response = np.random.poisson(tuning_curves[:, true_idx])
    # 디코딩
    decoded = population_vector_decode(sim_response, preferred_dirs)
    errors.append(decoded - true_stimulus)

ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='red', linestyle='--')
ax2.set_xlabel('Decoding Error (°)')
ax2.set_ylabel('Count')
ax2.set_title(f'Error Distribution\nMean: {np.mean(errors):.1f}°, Std: {np.std(errors):.1f}°')

plt.tight_layout()
plt.show()
```

---

## 4. 디코더 성능 평가

### 4.1 평가 지표

| 지표 | 수식 | 의미 |
|------|------|------|
| **MSE** | $\frac{1}{N}\sum(\hat{s}-s)^2$ | 평균 제곱 오차 |
| **Bias** | $E[\hat{s}] - s$ | 편향 |
| **Variance** | $Var[\hat{s}]$ | 분산 |
| **정보량** | $I(S;\hat{S})$ | 상호정보량 |

### 4.2 디코더 비교

```python
def compare_decoders(n_trials=100):
    """MLE vs Population Vector 비교"""
    
    mle_errors = []
    pv_errors = []
    
    test_stimuli = np.linspace(20, 160, 8)
    
    for true_stim in test_stimuli:
        true_idx = np.argmin(np.abs(stimuli - true_stim))
        
        for _ in range(n_trials):
            # 반응 시뮬레이션
            resp = np.random.poisson(tuning_curves[:, true_idx])
            
            # MLE 디코딩
            log_like = poisson_log_likelihood(resp, tuning_curves, stimuli)
            mle_decoded = stimuli[np.argmax(log_like)]
            mle_errors.append(mle_decoded - true_stim)
            
            # PV 디코딩
            pv_decoded = population_vector_decode(resp, preferred_dirs)
            pv_errors.append(pv_decoded - true_stim)
    
    return np.array(mle_errors), np.array(pv_errors)

mle_err, pv_err = compare_decoders()

print("=== 디코더 성능 비교 ===")
print(f"MLE - Bias: {np.mean(mle_err):.2f}°, RMSE: {np.sqrt(np.mean(mle_err**2)):.2f}°")
print(f"PV  - Bias: {np.mean(pv_err):.2f}°, RMSE: {np.sqrt(np.mean(pv_err**2)):.2f}°")
```

---

## 📝 실습 문제

### 문제 1: 베이지안 디코딩
비균일 사전확률을 적용한 MAP 디코더를 구현하세요.

### 문제 2: 디코더 튜닝
뉴런 수(N=4, 8, 16, 32)에 따른 디코딩 정확도를 비교하세요.

### 문제 3: 실제 데이터
Allen Brain Observatory 데이터에 디코더를 적용해보세요.

---

## 🔗 관련 개념

- [베이지안 디코딩](../../concepts/bayesian-decoding)
- [튜닝 커브](../../concepts/tuning-curve)
- [Population Vector](../../concepts/population-vector)

---

## 📚 참고 자료

- Dayan & Abbott, Chapter 3: Neural Decoding
- Pouget et al. (2000): Information Processing with Population Codes
- Neuromatch Academy: Decoding Models

---

## ⏭️ Next

```{button-ref} day2-bci-applications
:color: primary

다음: W3D2 - BCI Applications →
```
