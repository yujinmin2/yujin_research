---
title: "W2D2 - Spike Trains & Neural Code"
subtitle: "스파이크 트레인 분석과 신경 코드"
---

# W2D2: Spike Trains & Neural Code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W2D2_SpikeTrains.ipynb)

---

## 📋 Overview

**핵심 질문**: 스파이크 트레인에서 어떤 정보를 추출할 수 있는가?

스파이크 트레인(Spike Train)은 뉴런의 발화 시점들의 시퀀스입니다. 이를 분석하여 뉴런의 활동 패턴과 정보 처리 방식을 이해할 수 있습니다.

```
스파이크 트레인: |  | |   ||  |   |  ||  |
시간 →          0ms              500ms            1000ms
```

---

## 🎯 Learning Objectives

1. **스파이크 트레인**의 기본 통계량을 계산할 수 있다
2. **ISI (Inter-Spike Interval)** 분석을 수행할 수 있다
3. **PSTH (Peri-Stimulus Time Histogram)**를 구축할 수 있다
4. **Raster plot**을 해석할 수 있다
5. **포아송 과정**으로 스파이크를 모델링할 수 있다

---

## 1. 스파이크 트레인 기초

### 1.1 스파이크 트레인 표현

```python
import numpy as np
import matplotlib.pyplot as plt

def generate_poisson_spikes(rate, duration, seed=None):
    """
    포아송 과정으로 스파이크 생성
    
    Parameters:
    -----------
    rate : float - 평균 발화율 (Hz)
    duration : float - 시뮬레이션 기간 (초)
    seed : int - 랜덤 시드
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_spikes = np.random.poisson(rate * duration)
    spike_times = np.sort(np.random.uniform(0, duration, n_spikes))
    return spike_times

# 예시: 30Hz 뉴런, 2초 동안
spike_times = generate_poisson_spikes(rate=30, duration=2, seed=42)

print(f"스파이크 수: {len(spike_times)}")
print(f"평균 발화율: {len(spike_times) / 2:.1f} Hz")
print(f"처음 5개 스파이크 시간: {spike_times[:5]}")
```

### 1.2 스파이크 트레인 시각화

```python
def visualize_spike_train(spike_times, duration, title="Spike Train"):
    """스파이크 트레인 시각화"""
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    
    # 1. 래스터 플롯
    axes[0].eventplot(spike_times, lineoffsets=0, linelengths=0.8,
                      colors='black', linewidths=1.5)
    axes[0].set_ylabel('Raster')
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_yticks([])
    axes[0].set_title(title)
    
    # 2. 순간 발화율 (가우시안 커널)
    dt = 0.001  # 1ms
    t = np.arange(0, duration, dt)
    sigma = 0.05  # 50ms 커널
    
    rate = np.zeros(len(t))
    for spike in spike_times:
        kernel = np.exp(-0.5 * ((t - spike) / sigma)**2)
        rate += kernel
    rate = rate / (sigma * np.sqrt(2 * np.pi))
    
    axes[1].plot(t, rate, 'b-', linewidth=1)
    axes[1].fill_between(t, rate, alpha=0.3)
    axes[1].set_ylabel('Firing Rate (Hz)')
    axes[1].axhline(y=len(spike_times)/duration, color='red', 
                    linestyle='--', label=f'Mean: {len(spike_times)/duration:.1f} Hz')
    axes[1].legend()
    
    # 3. 누적 스파이크 수
    cumulative = np.arange(1, len(spike_times) + 1)
    axes[2].step(spike_times, cumulative, 'g-', where='post', linewidth=1.5)
    axes[2].set_ylabel('Cumulative Spikes')
    axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.show()

visualize_spike_train(spike_times, duration=2, title="Poisson Spike Train (30 Hz)")
```

---

## 2. Inter-Spike Interval (ISI) 분석

### 2.1 ISI란?

**ISI (Inter-Spike Interval)**는 연속된 두 스파이크 사이의 시간 간격입니다.

```
스파이크:    |       |     |        |
ISI:         <--ISI1--> <-ISI2-> <---ISI3--->
```

### 2.2 ISI 계산 및 분석

```python
def analyze_isi(spike_times):
    """ISI 분석"""
    
    # ISI 계산
    isi = np.diff(spike_times) * 1000  # ms로 변환
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. ISI 히스토그램
    axes[0, 0].hist(isi, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0, 0].axvline(x=np.mean(isi), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(isi):.1f} ms')
    axes[0, 0].axvline(x=np.median(isi), color='orange', linestyle='--',
                       label=f'Median: {np.median(isi):.1f} ms')
    axes[0, 0].set_xlabel('ISI (ms)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('ISI Histogram')
    axes[0, 0].legend()
    
    # 2. Log ISI 히스토그램
    axes[0, 1].hist(np.log10(isi), bins=30, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].set_xlabel('log10(ISI) (ms)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Log ISI Histogram')
    
    # 3. ISI 자기상관
    if len(isi) > 1:
        axes[1, 0].scatter(isi[:-1], isi[1:], alpha=0.5, s=20)
        axes[1, 0].set_xlabel('ISI_n (ms)')
        axes[1, 0].set_ylabel('ISI_n+1 (ms)')
        axes[1, 0].set_title('ISI Return Map')
        # 대각선 추가
        max_isi = max(isi)
        axes[1, 0].plot([0, max_isi], [0, max_isi], 'r--', alpha=0.5)
    
    # 4. ISI 통계
    stats_text = f"""ISI Statistics:
    
    Count: {len(isi)}
    Mean: {np.mean(isi):.2f} ms
    Std: {np.std(isi):.2f} ms
    CV: {np.std(isi)/np.mean(isi):.2f}
    Min: {np.min(isi):.2f} ms
    Max: {np.max(isi):.2f} ms
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
                    transform=axes[1, 1].transAxes, verticalalignment='center')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistics')
    
    plt.tight_layout()
    plt.show()
    
    return isi

isi = analyze_isi(spike_times)
```

### 2.3 CV (Coefficient of Variation)

**CV = σ_ISI / μ_ISI** 는 발화의 규칙성을 측정합니다.

| CV 값 | 의미 | 패턴 |
|-------|------|------|
| CV ≈ 0 | 매우 규칙적 | 메트로놈처럼 |
| CV ≈ 1 | 포아송 (랜덤) | 무작위 |
| CV > 1 | 불규칙/버스트 | 클러스터링 |

```python
def compare_cv():
    """다양한 CV를 가진 스파이크 트레인 비교"""
    np.random.seed(42)
    duration = 2
    n_spikes = 60
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 8))
    
    # 1. 규칙적 (CV ≈ 0)
    regular_times = np.linspace(0.01, duration - 0.01, n_spikes)
    regular_isi = np.diff(regular_times) * 1000
    cv_regular = np.std(regular_isi) / np.mean(regular_isi)
    
    axes[0, 0].eventplot(regular_times, linewidths=1.5)
    axes[0, 0].set_title(f'Regular (CV = {cv_regular:.2f})')
    axes[0, 0].set_xlim(0, duration)
    
    axes[0, 1].hist(regular_isi, bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('ISI (ms)')
    
    # 2. 포아송 (CV ≈ 1)
    poisson_times = generate_poisson_spikes(rate=30, duration=duration, seed=42)
    poisson_isi = np.diff(poisson_times) * 1000
    cv_poisson = np.std(poisson_isi) / np.mean(poisson_isi)
    
    axes[1, 0].eventplot(poisson_times, linewidths=1.5, colors='green')
    axes[1, 0].set_title(f'Poisson (CV = {cv_poisson:.2f})')
    axes[1, 0].set_xlim(0, duration)
    
    axes[1, 1].hist(poisson_isi, bins=20, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].set_xlabel('ISI (ms)')
    
    # 3. 버스트 (CV > 1)
    burst_times = []
    t = 0.1
    while t < duration - 0.1:
        # 버스트: 5개 스파이크가 빠르게
        for i in range(5):
            burst_times.append(t + i * 0.005)
        t += np.random.uniform(0.2, 0.4)  # 버스트 간 긴 간격
    burst_times = np.array(burst_times)
    burst_isi = np.diff(burst_times) * 1000
    cv_burst = np.std(burst_isi) / np.mean(burst_isi)
    
    axes[2, 0].eventplot(burst_times, linewidths=1.5, colors='red')
    axes[2, 0].set_title(f'Burst (CV = {cv_burst:.2f})')
    axes[2, 0].set_xlim(0, duration)
    axes[2, 0].set_xlabel('Time (s)')
    
    axes[2, 1].hist(burst_isi, bins=20, edgecolor='black', alpha=0.7, color='red')
    axes[2, 1].set_xlabel('ISI (ms)')
    
    for ax in axes[:, 0]:
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

compare_cv()
```

---

## 3. PSTH (Peri-Stimulus Time Histogram)

### 3.1 PSTH란?

**PSTH**는 자극 시점을 기준으로 정렬된 발화율의 시간적 패턴입니다. 여러 trial을 평균하여 신호 대 잡음비를 높입니다.

```
Trial 1:  |  |   ||  |     자극
Trial 2:    | |  | |  |      ↓
Trial 3:  |   | ||   |       
          ─────────────────────
PSTH:     ▁▂▃▆█▆▃▂▁   (평균 발화율)
```

### 3.2 PSTH 구현

```python
def create_psth(spike_trains, stim_times, window=(-0.2, 0.5), bin_size=0.01):
    """
    PSTH 생성
    
    Parameters:
    -----------
    spike_trains : list of arrays - 각 trial의 스파이크 시간
    stim_times : array - 자극 시점
    window : tuple - 자극 전후 분석 윈도우 (초)
    bin_size : float - 빈 크기 (초)
    """
    
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2
    
    all_aligned_spikes = []
    psth_matrix = []
    
    for trial_idx, (spikes, stim_time) in enumerate(zip(spike_trains, stim_times)):
        # 자극 시점 기준으로 정렬
        aligned = spikes - stim_time
        # 윈도우 내 스파이크만 선택
        aligned = aligned[(aligned >= window[0]) & (aligned <= window[1])]
        all_aligned_spikes.append(aligned)
        
        # 히스토그램
        counts, _ = np.histogram(aligned, bins=bins)
        psth_matrix.append(counts)
    
    psth_matrix = np.array(psth_matrix)
    psth_mean = np.mean(psth_matrix, axis=0) / bin_size  # Hz로 변환
    psth_sem = np.std(psth_matrix, axis=0) / np.sqrt(len(spike_trains)) / bin_size
    
    return bin_centers, psth_mean, psth_sem, all_aligned_spikes

def simulate_visual_response():
    """시각 자극에 대한 뉴런 반응 시뮬레이션"""
    np.random.seed(42)
    
    n_trials = 30
    baseline_rate = 10  # Hz
    evoked_rate = 80    # Hz
    duration = 2        # 초
    stim_onset = 1.0    # 자극 시작
    stim_duration = 0.3 # 자극 지속시간
    
    spike_trains = []
    stim_times = []
    
    for trial in range(n_trials):
        spikes = []
        t = 0
        
        while t < duration:
            # 현재 시점의 발화율 결정
            if stim_onset <= t < stim_onset + stim_duration:
                rate = evoked_rate
            else:
                rate = baseline_rate
            
            # 다음 스파이크까지의 시간 (지수 분포)
            isi = np.random.exponential(1 / rate)
            t += isi
            
            if t < duration:
                spikes.append(t)
        
        spike_trains.append(np.array(spikes))
        stim_times.append(stim_onset)
    
    return spike_trains, np.array(stim_times), stim_onset, stim_duration

# 데이터 생성
spike_trains, stim_times, stim_onset, stim_duration = simulate_visual_response()

# PSTH 계산
bin_centers, psth_mean, psth_sem, aligned_spikes = create_psth(
    spike_trains, stim_times, window=(-0.3, 0.6), bin_size=0.02
)

# 시각화
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                          gridspec_kw={'height_ratios': [2, 1]})

# 래스터 플롯
for i, spikes in enumerate(aligned_spikes):
    axes[0].scatter(spikes, np.full_like(spikes, i), 
                    marker='|', s=20, c='black', linewidths=0.5)

axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stimulus Onset')
axes[0].axvspan(0, stim_duration, alpha=0.2, color='red', label='Stimulus')
axes[0].set_ylabel('Trial')
axes[0].set_title('Raster Plot & PSTH: Visual Neuron Response')
axes[0].legend(loc='upper right')

# PSTH
axes[1].bar(bin_centers, psth_mean, width=0.02, alpha=0.7, color='steelblue',
            edgecolor='black', linewidth=0.5)
axes[1].fill_between(bin_centers, psth_mean - psth_sem, psth_mean + psth_sem,
                      alpha=0.3, color='steelblue')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].axvspan(0, stim_duration, alpha=0.2, color='red')
axes[1].set_xlabel('Time from Stimulus Onset (s)')
axes[1].set_ylabel('Firing Rate (Hz)')
axes[1].set_xlim(-0.3, 0.6)

plt.tight_layout()
plt.show()
```

---

## 4. 포아송 과정 (Poisson Process)

### 4.1 스파이크 트레인 모델링

포아송 과정은 스파이크 트레인의 가장 기본적인 확률 모델입니다.

**특성**:
- 스파이크 발생은 독립적
- 짧은 시간 dt 동안 스파이크 확률: λ·dt
- ISI 분포: 지수 분포

```python
def poisson_spike_model():
    """포아송 스파이크 모델 특성"""
    np.random.seed(42)
    
    rate = 50  # Hz
    duration = 10  # 초
    n_simulations = 1000
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 여러 시뮬레이션의 스파이크 수 분포
    spike_counts = []
    all_isis = []
    
    for _ in range(n_simulations):
        spikes = generate_poisson_spikes(rate, duration)
        spike_counts.append(len(spikes))
        if len(spikes) > 1:
            all_isis.extend(np.diff(spikes) * 1000)
    
    # 스파이크 수 히스토그램
    axes[0, 0].hist(spike_counts, bins=30, density=True, alpha=0.7, 
                    edgecolor='black', label='Simulated')
    
    # 이론적 포아송 분포
    from scipy.stats import poisson
    x = np.arange(min(spike_counts), max(spike_counts))
    axes[0, 0].plot(x, poisson.pmf(x, rate * duration), 'r-', 
                    linewidth=2, label='Poisson PMF')
    axes[0, 0].set_xlabel('Spike Count')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_title(f'Spike Count Distribution (λT = {rate * duration})')
    axes[0, 0].legend()
    
    # 2. ISI 분포
    axes[0, 1].hist(all_isis, bins=50, density=True, alpha=0.7,
                    edgecolor='black', label='Simulated')
    
    # 이론적 지수 분포
    x_exp = np.linspace(0, max(all_isis), 100)
    exp_pdf = rate/1000 * np.exp(-rate/1000 * x_exp)
    axes[0, 1].plot(x_exp, exp_pdf, 'r-', linewidth=2, label='Exponential PDF')
    axes[0, 1].set_xlabel('ISI (ms)')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].set_title('ISI Distribution')
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 100)
    
    # 3. Fano Factor over time
    window_sizes = np.linspace(0.01, 1, 50)
    fano_factors = []
    
    for window in window_sizes:
        counts = []
        for _ in range(500):
            spikes = generate_poisson_spikes(rate, window)
            counts.append(len(spikes))
        counts = np.array(counts)
        fano = np.var(counts) / np.mean(counts) if np.mean(counts) > 0 else 0
        fano_factors.append(fano)
    
    axes[1, 0].plot(window_sizes * 1000, fano_factors, 'b-', linewidth=2)
    axes[1, 0].axhline(y=1, color='red', linestyle='--', label='Poisson (F=1)')
    axes[1, 0].set_xlabel('Window Size (ms)')
    axes[1, 0].set_ylabel('Fano Factor')
    axes[1, 0].set_title('Fano Factor vs Window Size')
    axes[1, 0].legend()
    
    # 4. 포아송 특성 요약
    summary = f"""Poisson Process Properties:
    
    Rate (λ): {rate} Hz
    
    Spike Count:
      Mean: {np.mean(spike_counts):.1f}
      Var:  {np.var(spike_counts):.1f}
      (Expected: {rate * duration})
    
    ISI:
      Mean: {np.mean(all_isis):.1f} ms
      CV:   {np.std(all_isis)/np.mean(all_isis):.2f}
      (Expected CV: 1.0)
    
    Fano Factor: ~1.0 (variance = mean)
    """
    axes[1, 1].text(0.1, 0.5, summary, fontsize=11, family='monospace',
                    transform=axes[1, 1].transAxes, verticalalignment='center')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

poisson_spike_model()
```

---

## 📝 실습 문제

### 문제 1: ISI 분석
제공된 실제 뉴런 데이터의 ISI를 분석하고, 포아송 과정과 비교하세요.

### 문제 2: PSTH 구축
청각 자극에 대한 뉴런 반응 데이터로 PSTH를 구축하세요.

### 문제 3: 버스트 탐지
ISI 기반으로 버스트 스파이크를 탐지하는 알고리즘을 구현하세요.

---

## 🔗 참고 자료

- Dayan & Abbott, Chapter 1: Neural Encoding
- Rieke et al., "Spikes: Exploring the Neural Code"
- Neuromatch Academy: Spike Train Analysis

---

## ⏭️ Next

다음 시간에는 **신경 디코딩**을 배웁니다:
- 스파이크 → 자극 복원
- 베이지안 디코딩
- Population decoding

```{button-ref} ../week3/day1-neural-decoding
:color: primary

다음: W3D1 - Neural Decoding →
```
