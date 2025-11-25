---
title: "W2D2 - Spike Trains & Neural Code"
subtitle: "스파이크 트레인과 신경 코드"
---

# W2D2: Spike Trains & Neural Code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W2D2_SpikeTrains.ipynb)

---

## 🎯 Learning Objectives

1. 스파이크 트레인 분석 방법
2. Inter-Spike Interval (ISI) 분석
3. Peri-Stimulus Time Histogram (PSTH)

---

## 1. Inter-Spike Interval (ISI)

```python
def compute_isi(spike_times):
    """스파이크 간 간격 계산"""
    return np.diff(spike_times)

# ISI 히스토그램
spike_times = generate_poisson_spikes(rate=30, duration=10)
isi = compute_isi(spike_times)

plt.figure(figsize=(8, 4))
plt.hist(isi * 1000, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('ISI (ms)')
plt.ylabel('Count')
plt.title('Inter-Spike Interval Distribution')
plt.show()
```

---

## 2. PSTH (Peri-Stimulus Time Histogram)

자극 전후 뉴런 반응의 시간적 패턴을 분석합니다.

---

## ⏭️ Next

```{button-ref} ../week3/day1-neural-decoding
:color: primary

다음: W3D1 - Neural Decoding →
```
