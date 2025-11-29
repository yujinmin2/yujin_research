---
title: "W3D2 - BCI Applications"
subtitle: "실제 BCI 디코딩 사례와 응용"
---

# W3D2: BCI Applications

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W3D2_BCIApplications.ipynb)

---

## 📋 Overview

신경 디코딩 이론이 실제 BCI 시스템에서 어떻게 적용되는지 살펴봅니다.

```{mermaid}
flowchart LR
    subgraph 뇌신호획득
        EEG[EEG<br/>비침습]
        ECOG[ECoG<br/>반침습]
        INTRA[Intracortical<br/>침습]
    end
    
    subgraph 디코딩
        PP[전처리] --> FE[특징추출] --> DEC[분류/회귀]
    end
    
    subgraph 응용
        MOTOR[운동 제어<br/>커서, 로봇팔]
        COMM[의사소통<br/>타이핑, 음성]
        SENSE[감각 복원<br/>시각, 촉각]
    end
    
    EEG --> PP
    ECOG --> PP
    INTRA --> PP
    DEC --> MOTOR
    DEC --> COMM
    DEC --> SENSE
```

---

## 🎯 Learning Objectives

1. **BCI 시스템의 구성요소** 이해
2. 다양한 **신호 획득 방법** 비교
3. **운동 BCI** 디코딩 파이프라인 구현
4. **P300 Speller** 원리와 구현
5. 최신 **BCI 연구 동향** 파악

---

## 1. BCI 시스템 구성요소

### 1.1 전체 파이프라인

```{mermaid}
flowchart TB
    subgraph 1. 신호획득
        BRAIN[뇌] --> SENSOR[센서<br/>EEG/ECoG/Array]
        SENSOR --> AMP[증폭기]
        AMP --> ADC[A/D 변환]
    end
    
    subgraph 2. 신호처리
        ADC --> FILT[필터링<br/>노이즈 제거]
        FILT --> FEAT[특징 추출<br/>PSD, ERP]
    end
    
    subgraph 3. 디코딩
        FEAT --> CLASS[분류기<br/>SVM, CNN]
        CLASS --> CMD[명령 생성]
    end
    
    subgraph 4. 출력
        CMD --> DEV[장치 제어<br/>커서, 휠체어]
        DEV --> FB[피드백]
    end
    
    FB -.-> BRAIN
```

### 1.2 신호 획득 방법 비교

| 방법 | 침습성 | 공간해상도 | 시간해상도 | 신호품질 | 장기안정성 |
|------|--------|-----------|-----------|---------|-----------|
| **EEG** | 비침습 | ~cm | ~ms | 낮음 | 높음 |
| **ECoG** | 반침습 | ~mm | ~ms | 높음 | 중간 |
| **Intracortical** | 침습 | ~μm | ~ms | 매우 높음 | 낮음 |
| **fMRI** | 비침습 | ~mm | ~s | 높음 | - |

---

## 2. 운동 BCI (Motor BCI)

### 2.1 운동 상상 (Motor Imagery)

운동 상상 시 **감각운동 피질**에서 특징적인 EEG 패턴이 발생합니다.

```{mermaid}
flowchart LR
    subgraph 왼손상상
        L_MU[Mu rhythm ↓<br/>C4 전극]
    end
    
    subgraph 오른손상상
        R_MU[Mu rhythm ↓<br/>C3 전극]
    end
    
    L_MU --> CLASS[분류기]
    R_MU --> CLASS
    CLASS --> CMD[왼쪽/오른쪽<br/>명령]
```

### 2.2 ERD/ERS (Event-Related Desynchronization/Synchronization)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch

def simulate_motor_imagery(condition='left', duration=4, fs=250):
    """
    운동 상상 EEG 시뮬레이션
    
    Parameters:
    -----------
    condition : str - 'left' or 'right' 손 상상
    duration : float - 시간 (초)
    fs : int - 샘플링 주파수
    """
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # 기저 EEG (알파 + 노이즈)
    baseline = 10 * np.sin(2 * np.pi * 10 * t) + 5 * np.random.randn(n_samples)
    
    # C3 (왼쪽 운동피질) - 오른손 상상 시 ERD
    # C4 (오른쪽 운동피질) - 왼손 상상 시 ERD
    
    # ERD: 1-2초 후 mu rhythm 감소
    erd_envelope = np.ones(n_samples)
    erd_start = int(1 * fs)
    erd_end = int(3 * fs)
    erd_envelope[erd_start:erd_end] = 0.3  # 70% 감소
    
    if condition == 'left':
        # 왼손 상상 → C4에서 ERD
        C3 = baseline + 5 * np.random.randn(n_samples)
        C4 = baseline * erd_envelope + 5 * np.random.randn(n_samples)
    else:
        # 오른손 상상 → C3에서 ERD
        C3 = baseline * erd_envelope + 5 * np.random.randn(n_samples)
        C4 = baseline + 5 * np.random.randn(n_samples)
    
    return t, C3, C4

# 시뮬레이션
t, C3_left, C4_left = simulate_motor_imagery('left')
_, C3_right, C4_right = simulate_motor_imagery('right')

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 왼손 상상
axes[0, 0].plot(t, C3_left, 'b-', alpha=0.7, label='C3')
axes[0, 0].plot(t, C4_left, 'r-', alpha=0.7, label='C4 (ERD)')
axes[0, 0].axvspan(1, 3, alpha=0.2, color='yellow', label='Motor Imagery')
axes[0, 0].set_title('Left Hand Imagery → C4 ERD')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Amplitude (μV)')
axes[0, 0].legend()

# 오른손 상상
axes[0, 1].plot(t, C3_right, 'b-', alpha=0.7, label='C3 (ERD)')
axes[0, 1].plot(t, C4_right, 'r-', alpha=0.7, label='C4')
axes[0, 1].axvspan(1, 3, alpha=0.2, color='yellow', label='Motor Imagery')
axes[0, 1].set_title('Right Hand Imagery → C3 ERD')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].legend()

# PSD 비교
for ax, (C3, C4, title) in zip(axes[1], 
                                [(C3_left, C4_left, 'Left Hand'), 
                                 (C3_right, C4_right, 'Right Hand')]):
    # 운동 상상 구간의 PSD
    mi_start = int(1 * 250)
    mi_end = int(3 * 250)
    
    f, psd_C3 = welch(C3[mi_start:mi_end], fs=250, nperseg=256)
    f, psd_C4 = welch(C4[mi_start:mi_end], fs=250, nperseg=256)
    
    ax.semilogy(f, psd_C3, 'b-', label='C3')
    ax.semilogy(f, psd_C4, 'r-', label='C4')
    ax.axvspan(8, 13, alpha=0.2, color='green', label='Mu band')
    ax.set_xlim(0, 40)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title(f'{title} - Power Spectrum')
    ax.legend()

plt.tight_layout()
plt.show()
```

### 2.3 특징 추출 & 분류

```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def extract_band_power(signal, fs=250, band=(8, 13)):
    """Mu 대역 파워 추출"""
    # 밴드패스 필터
    nyq = fs / 2
    b, a = butter(4, [band[0]/nyq, band[1]/nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    # 파워 계산
    return np.mean(filtered**2)

def create_dataset(n_trials=100):
    """학습 데이터셋 생성"""
    X = []
    y = []
    
    for _ in range(n_trials):
        # 왼손 상상
        _, C3, C4 = simulate_motor_imagery('left')
        mi_segment = slice(int(1*250), int(3*250))
        features = [
            extract_band_power(C3[mi_segment]),
            extract_band_power(C4[mi_segment]),
        ]
        X.append(features)
        y.append(0)  # 왼손 = 0
        
        # 오른손 상상
        _, C3, C4 = simulate_motor_imagery('right')
        features = [
            extract_band_power(C3[mi_segment]),
            extract_band_power(C4[mi_segment]),
        ]
        X.append(features)
        y.append(1)  # 오른손 = 1
    
    return np.array(X), np.array(y)

# 데이터셋 생성 및 분류
X, y = create_dataset(n_trials=50)

# SVM 분류기
clf = SVC(kernel='rbf')
scores = cross_val_score(clf, X, y, cv=5)
print(f"분류 정확도: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
```

---

## 3. P300 Speller

### 3.1 원리

**P300**은 드물게 나타나는 목표 자극에 대한 뇌의 반응으로, 자극 후 약 300ms에 양의 피크가 나타납니다.

```{mermaid}
flowchart TB
    subgraph 화면
        GRID[6x6 문자 매트릭스<br/>A B C D E F<br/>G H I J K L<br/>...]
    end
    
    subgraph 깜빡임
        ROW[행 깜빡임] --> TARGET{목표 포함?}
        COL[열 깜빡임] --> TARGET
    end
    
    TARGET -->|Yes| P300[P300 반응 ✓]
    TARGET -->|No| NOP[반응 없음]
    
    P300 --> DET[검출]
    DET --> CHAR[문자 선택]
```

### 3.2 구현

```python
def simulate_p300(is_target=True, fs=250, duration=0.8):
    """
    P300 EEG 시뮬레이션
    
    Parameters:
    -----------
    is_target : bool - 목표 자극 여부
    """
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # 기저 EEG
    eeg = 5 * np.random.randn(n_samples)
    
    if is_target:
        # P300 성분 추가 (300ms에 피크)
        p300_latency = 0.3  # 300ms
        p300_width = 0.05
        p300_amplitude = 8  # μV
        
        p300 = p300_amplitude * np.exp(-0.5 * ((t - p300_latency) / p300_width)**2)
        eeg += p300
    
    return t, eeg

# 시뮬레이션
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 단일 trial
t, target_eeg = simulate_p300(is_target=True)
_, nontarget_eeg = simulate_p300(is_target=False)

axes[0].plot(t * 1000, target_eeg, 'r-', label='Target', alpha=0.8)
axes[0].plot(t * 1000, nontarget_eeg, 'b-', label='Non-target', alpha=0.8)
axes[0].axvline(x=300, color='gray', linestyle='--', label='300ms')
axes[0].set_xlabel('Time (ms)')
axes[0].set_ylabel('Amplitude (μV)')
axes[0].set_title('Single Trial')
axes[0].legend()

# 평균 (여러 trial)
n_trials = 20
target_avg = np.zeros_like(t)
nontarget_avg = np.zeros_like(t)

for _ in range(n_trials):
    _, eeg = simulate_p300(is_target=True)
    target_avg += eeg
    _, eeg = simulate_p300(is_target=False)
    nontarget_avg += eeg

target_avg /= n_trials
nontarget_avg /= n_trials

axes[1].plot(t * 1000, target_avg, 'r-', linewidth=2, label='Target (avg)')
axes[1].plot(t * 1000, nontarget_avg, 'b-', linewidth=2, label='Non-target (avg)')
axes[1].axvline(x=300, color='gray', linestyle='--')
axes[1].fill_between(t * 1000, target_avg, nontarget_avg, 
                      where=target_avg > nontarget_avg, alpha=0.3, color='red')
axes[1].set_xlabel('Time (ms)')
axes[1].set_ylabel('Amplitude (μV)')
axes[1].set_title(f'Averaged ({n_trials} trials)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## 4. 최신 BCI 연구 동향

### 4.1 침습적 BCI 성과

| 연구 | 연도 | 성과 |
|------|------|------|
| **BrainGate** | 2006~ | 로봇팔 제어, 타이핑 |
| **Neuralink** | 2020~ | 고밀도 전극, 무선 전송 |
| **Stanford** | 2021 | 분당 90자 타이핑 |
| **Synchron** | 2022 | 혈관 내 스텐트형 전극 |

### 4.2 비침습적 BCI 발전

- **딥러닝**: CNN, Transformer 기반 디코딩
- **전이학습**: 피험자 간 모델 전이
- **실시간**: 저지연 처리 시스템

---

## 📝 실습 문제

### 문제 1: Motor Imagery 분류
CSP(Common Spatial Patterns) 특징을 사용하여 분류 정확도를 개선하세요.

### 문제 2: P300 Speller 구현
6x6 매트릭스에서 목표 문자를 검출하는 전체 파이프라인을 구현하세요.

### 문제 3: 실제 데이터
BCI Competition 데이터셋으로 디코더를 학습시키세요.

---

## 🔗 관련 개념

- [EEG](../../concepts/eeg)
- [BCI Decoder](../../concepts/bci-decoder)
- [베이지안 디코딩](../../concepts/bayesian-decoding)

---

## 📚 참고 자료

- Wolpaw & Wolpaw, "Brain-Computer Interfaces: Principles and Practice"
- BCI Competition datasets: http://www.bbci.de/competition/
- OpenBCI: https://openbci.com/

---

## ⏭️ Next

```{button-ref} ../week4/day1-information-theory
:color: primary

다음: W4D1 - Information Theory →
```
