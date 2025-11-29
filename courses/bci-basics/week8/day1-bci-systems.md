---
title: "W8D1 - BCI Systems"
subtitle: "실용적인 BCI 시스템 설계와 구현"
---

# W8D1: BCI Systems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W8D1_BCISystems.ipynb)

---

## 📋 Overview

지금까지 배운 모든 것을 통합하여 **실제 BCI 시스템**을 설계합니다.

```{mermaid}
flowchart TB
    subgraph 신호획득
        EEG[EEG]
        ECOG[ECoG]
        SPIKE[Intracortical]
    end
    
    subgraph 전처리
        FILT[필터링]
        ART[아티팩트 제거]
        REF[참조 선택]
    end
    
    subgraph 특징추출
        TIME[시간 특징]
        FREQ[주파수 특징]
        SPACE[공간 특징]
    end
    
    subgraph 분류/회귀
        SVM[SVM]
        LDA[LDA]
        DL[Deep Learning]
    end
    
    subgraph 출력
        CURSOR[커서 제어]
        SPELL[타이핑]
        ROBOT[로봇팔]
    end
    
    EEG --> FILT
    ECOG --> FILT
    SPIKE --> FILT
    FILT --> ART --> REF
    REF --> TIME
    REF --> FREQ
    REF --> SPACE
    TIME --> SVM
    FREQ --> LDA
    SPACE --> DL
    SVM --> CURSOR
    LDA --> SPELL
    DL --> ROBOT
```

---

## 🎯 Learning Objectives

1. **BCI 파이프라인** 전체 이해
2. **EEG 전처리** 구현
3. **특징 추출** 기법 적용
4. **실시간 분류** 시스템 구현

---

## 1. BCI 시스템 유형

### 1.1 분류

| 유형 | 신호 | 침습성 | 응용 |
|------|------|--------|------|
| **Motor Imagery** | EEG | 비침습 | 휠체어, 커서 |
| **P300 Speller** | EEG | 비침습 | 타이핑 |
| **SSVEP** | EEG | 비침습 | 선택 인터페이스 |
| **ECoG BCI** | ECoG | 반침습 | 고성능 제어 |
| **Intracortical** | Spikes | 침습 | 로봇팔, 음성 |

### 1.2 신호 특성 비교

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def compare_signals():
    """다양한 BCI 신호 특성 시뮬레이션"""
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fs = 1000  # 샘플링 주파수
    duration = 2
    t = np.arange(0, duration, 1/fs)
    
    # EEG 시뮬레이션
    np.random.seed(42)
    eeg = (10 * np.sin(2 * np.pi * 10 * t) +  # Alpha
           5 * np.sin(2 * np.pi * 22 * t) +   # Beta
           20 * np.random.randn(len(t)))       # Noise
    
    axes[0, 0].plot(t, eeg, 'b-', linewidth=0.5)
    axes[0, 0].set_title('EEG (Scalp)\nSNR: Low, Resolution: ~cm')
    axes[0, 0].set_ylabel('Amplitude (μV)')
    axes[0, 0].set_ylim(-100, 100)
    
    # EEG 스펙트럼
    f, psd = signal.welch(eeg, fs, nperseg=512)
    axes[0, 1].semilogy(f, psd, 'b-')
    axes[0, 1].set_xlim(0, 50)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].axvspan(8, 13, alpha=0.3, color='yellow', label='Alpha')
    axes[0, 1].axvspan(13, 30, alpha=0.3, color='green', label='Beta')
    axes[0, 1].legend()
    
    # ECoG 시뮬레이션
    ecog = (50 * np.sin(2 * np.pi * 10 * t) +
            30 * np.sin(2 * np.pi * 70 * t) +  # Gamma
            10 * np.random.randn(len(t)))
    
    axes[1, 0].plot(t, ecog, 'g-', linewidth=0.5)
    axes[1, 0].set_title('ECoG (Cortical Surface)\nSNR: Medium, Resolution: ~mm')
    axes[1, 0].set_ylabel('Amplitude (μV)')
    axes[1, 0].set_ylim(-200, 200)
    
    # ECoG 스펙트럼
    f, psd = signal.welch(ecog, fs, nperseg=512)
    axes[1, 1].semilogy(f, psd, 'g-')
    axes[1, 1].set_xlim(0, 150)
    axes[1, 1].axvspan(30, 100, alpha=0.3, color='red', label='High Gamma')
    axes[1, 1].legend()
    
    # Intracortical 시뮬레이션
    spike_times = np.sort(np.random.uniform(0, duration, 60))
    intracortical = np.zeros(len(t))
    for st in spike_times:
        idx = int(st * fs)
        if idx < len(t) - 20:
            # 스파이크 파형
            spike = 100 * np.exp(-np.arange(20) / 2) * np.sin(np.arange(20) * 0.5)
            intracortical[idx:idx+20] += spike
    intracortical += 5 * np.random.randn(len(t))
    
    axes[2, 0].plot(t, intracortical, 'r-', linewidth=0.5)
    axes[2, 0].set_title('Intracortical (Single Unit)\nSNR: High, Resolution: ~μm')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Amplitude (μV)')
    
    # 스파이크 래스터
    axes[2, 1].eventplot(spike_times, linewidths=1.5)
    axes[2, 1].set_xlim(0, duration)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_title(f'Spike Raster ({len(spike_times)/duration:.0f} Hz)')
    
    plt.tight_layout()
    plt.show()

compare_signals()
```

---

## 2. EEG 전처리 파이프라인

### 2.1 전체 흐름

```{mermaid}
flowchart LR
    RAW[Raw EEG] --> BP[밴드패스 필터<br/>0.5-40 Hz]
    BP --> NOTCH[노치 필터<br/>50/60 Hz]
    NOTCH --> REF[참조 재설정<br/>CAR/Laplacian]
    REF --> ART[아티팩트 제거<br/>ICA/임계값]
    ART --> EPOCH[에포킹]
    EPOCH --> FEAT[특징 추출]
```

### 2.2 구현

```python
from scipy.signal import butter, filtfilt, iirnotch

class EEGPreprocessor:
    def __init__(self, fs=250):
        self.fs = fs
    
    def bandpass_filter(self, data, low=0.5, high=40, order=4):
        """밴드패스 필터"""
        nyq = self.fs / 2
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data, axis=-1)
    
    def notch_filter(self, data, freq=60, Q=30):
        """노치 필터 (전원선 노이즈 제거)"""
        b, a = iirnotch(freq, Q, self.fs)
        return filtfilt(b, a, data, axis=-1)
    
    def common_average_reference(self, data):
        """공통 평균 참조 (CAR)"""
        return data - np.mean(data, axis=0)
    
    def artifact_rejection(self, data, threshold=100):
        """임계값 기반 아티팩트 제거"""
        mask = np.max(np.abs(data), axis=-1) < threshold
        return data[mask], mask
    
    def process(self, data):
        """전체 파이프라인"""
        data = self.bandpass_filter(data)
        data = self.notch_filter(data)
        data = self.common_average_reference(data)
        return data

# 예시
np.random.seed(42)
fs = 250
duration = 10
n_channels = 8
t = np.arange(0, duration, 1/fs)

# 시뮬레이션 EEG (신호 + 노이즈 + 아티팩트)
raw_eeg = np.zeros((n_channels, len(t)))
for ch in range(n_channels):
    # 신호
    raw_eeg[ch] = 10 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
    # 노이즈
    raw_eeg[ch] += 5 * np.random.randn(len(t))
    # 전원선 노이즈
    raw_eeg[ch] += 3 * np.sin(2 * np.pi * 60 * t)

# 전처리
preprocessor = EEGPreprocessor(fs=fs)
processed_eeg = preprocessor.process(raw_eeg)

# 비교
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Raw EEG
axes[0, 0].plot(t[:500], raw_eeg[0, :500], 'b-', linewidth=0.5)
axes[0, 0].set_title('Raw EEG (Channel 1)')
axes[0, 0].set_ylabel('Amplitude (μV)')

# Processed EEG
axes[0, 1].plot(t[:500], processed_eeg[0, :500], 'g-', linewidth=0.5)
axes[0, 1].set_title('Processed EEG')

# Raw 스펙트럼
f, psd = signal.welch(raw_eeg[0], fs, nperseg=256)
axes[1, 0].semilogy(f, psd, 'b-')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('PSD')
axes[1, 0].set_title('Raw Spectrum')
axes[1, 0].axvline(x=60, color='red', linestyle='--', label='60Hz noise')
axes[1, 0].legend()

# Processed 스펙트럼
f, psd = signal.welch(processed_eeg[0], fs, nperseg=256)
axes[1, 1].semilogy(f, psd, 'g-')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_title('Processed Spectrum')
axes[1, 1].axvline(x=60, color='red', linestyle='--')

plt.tight_layout()
plt.show()
```

---

## 3. 특징 추출

### 3.1 주요 특징

| 도메인 | 특징 | 용도 |
|--------|------|------|
| **시간** | 평균, 분산, Zero-crossing | 일반 |
| **주파수** | Band power, PSD | Motor Imagery |
| **시공간** | CSP, 코히어런스 | Motor Imagery |
| **ERP** | P300 진폭/지연 | P300 Speller |

### 3.2 Band Power 추출

```python
def extract_band_powers(data, fs=250):
    """주파수 대역별 파워 추출"""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }
    
    features = {}
    for band_name, (low, high) in bands.items():
        # 밴드패스 필터
        nyq = fs / 2
        b, a = butter(4, [low/nyq, high/nyq], btype='band')
        filtered = filtfilt(b, a, data, axis=-1)
        # 파워 계산
        power = np.mean(filtered**2, axis=-1)
        features[band_name] = power
    
    return features

# Motor Imagery 특징 추출 예시
def motor_imagery_features(eeg_left, eeg_right, fs=250):
    """좌우 운동 상상 분류를 위한 특징"""
    
    # 각 조건의 대역 파워
    feat_left = extract_band_powers(eeg_left, fs)
    feat_right = extract_band_powers(eeg_right, fs)
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    x = np.arange(len(feat_left))
    width = 0.35
    
    # C3 채널 (왼쪽 운동피질)
    axes[0].bar(x - width/2, [feat_left[b][0] for b in feat_left], width, 
                label='Left Hand MI', color='blue', alpha=0.7)
    axes[0].bar(x + width/2, [feat_right[b][0] for b in feat_right], width,
                label='Right Hand MI', color='red', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(list(feat_left.keys()))
    axes[0].set_ylabel('Power')
    axes[0].set_title('C3 (Left Motor Cortex)')
    axes[0].legend()
    
    # C4 채널 (오른쪽 운동피질)
    axes[1].bar(x - width/2, [feat_left[b][1] for b in feat_left], width,
                label='Left Hand MI', color='blue', alpha=0.7)
    axes[1].bar(x + width/2, [feat_right[b][1] for b in feat_right], width,
                label='Right Hand MI', color='red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(list(feat_left.keys()))
    axes[1].set_title('C4 (Right Motor Cortex)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# 시뮬레이션 데이터
np.random.seed(42)
fs = 250
n_samples = fs * 2

# 왼손 상상 → C4 mu 감소
eeg_left = np.zeros((2, n_samples))
eeg_left[0] = 10 * np.sin(2 * np.pi * 10 * np.arange(n_samples)/fs)  # C3 정상
eeg_left[1] = 5 * np.sin(2 * np.pi * 10 * np.arange(n_samples)/fs)   # C4 감소 (ERD)
eeg_left += 3 * np.random.randn(2, n_samples)

# 오른손 상상 → C3 mu 감소
eeg_right = np.zeros((2, n_samples))
eeg_right[0] = 5 * np.sin(2 * np.pi * 10 * np.arange(n_samples)/fs)  # C3 감소 (ERD)
eeg_right[1] = 10 * np.sin(2 * np.pi * 10 * np.arange(n_samples)/fs) # C4 정상
eeg_right += 3 * np.random.randn(2, n_samples)

motor_imagery_features(eeg_left, eeg_right)
```

---

## 4. 실시간 BCI 시스템

### 4.1 설계 고려사항

| 요소 | 고려사항 |
|------|----------|
| **지연** | 200ms 이하 권장 |
| **정확도** | 70% 이상 필요 |
| **적응** | 사용자/세션 변동 |
| **피드백** | 학습에 중요 |

### 4.2 간단한 실시간 시스템

```python
class SimpleBCI:
    def __init__(self, fs=250, n_channels=8, window_size=1.0):
        self.fs = fs
        self.n_channels = n_channels
        self.window_samples = int(window_size * fs)
        self.buffer = np.zeros((n_channels, self.window_samples))
        self.preprocessor = EEGPreprocessor(fs)
        self.classifier = None  # 학습된 분류기
    
    def update_buffer(self, new_data):
        """새 데이터로 버퍼 업데이트 (슬라이딩 윈도우)"""
        n_new = new_data.shape[1]
        self.buffer = np.roll(self.buffer, -n_new, axis=1)
        self.buffer[:, -n_new:] = new_data
    
    def extract_features(self):
        """현재 버퍼에서 특징 추출"""
        processed = self.preprocessor.process(self.buffer)
        features = extract_band_powers(processed, self.fs)
        # 벡터로 변환
        feat_vector = np.concatenate([features[b] for b in features])
        return feat_vector
    
    def predict(self):
        """실시간 예측"""
        if self.classifier is None:
            return None
        features = self.extract_features()
        return self.classifier.predict([features])[0]
    
    def simulate_realtime(self, data, labels, chunk_size=25):
        """실시간 시뮬레이션"""
        predictions = []
        true_labels = []
        
        for i in range(0, data.shape[1] - chunk_size, chunk_size):
            chunk = data[:, i:i+chunk_size]
            self.update_buffer(chunk)
            
            if i >= self.window_samples:
                pred = self.predict()
                if pred is not None:
                    predictions.append(pred)
                    # 해당 시점의 라벨 (간단화)
                    true_labels.append(labels[min(i // self.window_samples, len(labels)-1)])
        
        return predictions, true_labels

print("실시간 BCI 시스템 클래스 정의 완료")
print("실제 사용 시: classifier를 학습시킨 후 simulate_realtime() 호출")
```

---

## 5. 성능 평가

### 5.1 주요 지표

| 지표 | 수식 | 의미 |
|------|------|------|
| **정확도** | $\frac{TP+TN}{Total}$ | 전체 정확도 |
| **ITR** | $\frac{60}{T}[log_2N + P log_2P + (1-P)log_2\frac{1-P}{N-1}]$ | 정보 전송률 |
| **Cohen's κ** | $\frac{Acc - Chance}{1 - Chance}$ | 우연 보정 정확도 |

---

## 📝 실습 문제

### 문제 1: CSP 구현
Common Spatial Patterns 알고리즘을 구현하세요.

### 문제 2: 온라인 적응
세션 간 변동에 적응하는 분류기를 구현하세요.

### 문제 3: 피드백 시스템
시각적 피드백이 있는 Motor Imagery BCI를 구현하세요.

---

## 🔗 관련 개념

- [EEG](../../concepts/eeg)
- [ECoG](../../concepts/ecog)
- [BCI Decoder](../../concepts/bci-decoder)

---

## 📚 참고 자료

- Wolpaw & Wolpaw, "Brain-Computer Interfaces"
- BCI2000 documentation
- OpenBCI tutorials

---

## ⏭️ Next

```{button-ref} day2-future-directions
:color: primary

다음: W8D2 - Future Directions →
```
