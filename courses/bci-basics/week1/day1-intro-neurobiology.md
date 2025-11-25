---
title: "W1D1 - Introduction to Computational Neuroscience"
subtitle: "계산신경과학 입문"
subject: Computational Neuroscience
authors:
  - name: Yujin
---

# W1D1: Introduction to Computational Neuroscience

**계산신경과학 입문**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W1D1_Introduction.ipynb)
[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/yujin/yujin_research/blob/main/notebooks/W1D1_Introduction.ipynb)

---

## 📋 Overview

| 항목 | 내용 |
|-----|------|
| **소요 시간** | 약 45분 |
| **학습 목표** | 계산신경과학의 개념과 목표 이해 |
| **선수 지식** | 기초 수학, Python 기초 |

---

## 🎯 Learning Objectives

이 튜토리얼을 완료하면 다음을 할 수 있습니다:

1. 계산신경과학의 세 가지 모델 유형 구분 (Descriptive, Mechanistic, Interpretive)
2. 뇌의 기본 구조와 뉴런의 역할 이해
3. Python을 사용한 간단한 신경 데이터 시각화

---

## 📺 Video: What is Computational Neuroscience?

```{youtube} EMBED_VIDEO_ID_HERE
:width: 100%
```

*영상 링크가 준비되면 교체하세요*

---

## 1. What is Computational Neuroscience?

계산신경과학(Computational Neuroscience)은 **수학적 모델**과 **컴퓨터 시뮬레이션**을 사용하여 뇌가 어떻게 정보를 처리하는지 연구하는 학문입니다.

### 세 가지 핵심 질문

::::{grid} 1 3 3 3
:gutter: 3

:::{card} 📊 **Descriptive (What)**
뉴런이 외부 자극에 어떻게 반응하는가?

*Neural Encoding*
:::

:::{card} ⚙️ **Mechanistic (How)**
뉴런과 네트워크가 어떻게 동작하는가?

*Biophysical Models*
:::

:::{card} 🤔 **Interpretive (Why)**
왜 뇌는 이렇게 작동하는가?

*Computational Principles*
:::

::::

---

## 2. Setup: 환경 설정

```{code-block} python
:caption: 필요한 라이브러리 설치 및 임포트

# 필요한 패키지 설치 (Colab에서 실행)
# !pip install numpy matplotlib scipy -q

import numpy as np
import matplotlib.pyplot as plt

# 시각화 설정
plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

print("✅ Setup complete!")
```

---

## 3. 뉴런의 기본 구조

뉴런(Neuron)은 뇌의 기본 정보 처리 단위입니다.

### 뉴런의 구성 요소

```{figure} https://upload.wikimedia.org/wikipedia/commons/1/10/Blausen_0657_MultipolarNeuron.png
:width: 80%
:align: center
:alt: Neuron structure

뉴런의 기본 구조: 수상돌기(Dendrites), 세포체(Soma), 축삭(Axon)
```

| 구조 | 역할 |
|-----|------|
| **수상돌기 (Dendrites)** | 다른 뉴런으로부터 신호 수신 |
| **세포체 (Soma)** | 신호 통합 및 처리 |
| **축삭 (Axon)** | 다른 뉴런으로 신호 전달 |
| **시냅스 (Synapse)** | 뉴런 간 연결 지점 |

---

## 4. 활동 전위 (Action Potential)

뉴런은 **활동 전위(Action Potential)** 또는 **스파이크(Spike)**라 불리는 전기 신호를 통해 정보를 전달합니다.

### 🧑‍💻 Coding Exercise 1: 활동 전위 시뮬레이션

간단한 활동 전위의 형태를 시뮬레이션해봅시다.

```{code-block} python
:caption: 활동 전위 시뮬레이션

def action_potential(t, t_spike=5):
    """
    간단한 활동 전위 파형 생성
    
    Parameters:
    -----------
    t : array
        시간 배열 (ms)
    t_spike : float
        스파이크 발생 시점 (ms)
    
    Returns:
    --------
    V : array
        막전위 (mV)
    """
    V = np.zeros_like(t)
    
    # 휴지 전위
    V_rest = -70  # mV
    V[:] = V_rest
    
    # 스파이크 구간 (약 2ms)
    spike_idx = (t >= t_spike) & (t < t_spike + 2)
    
    # 탈분극 및 재분극
    t_local = t[spike_idx] - t_spike
    V[spike_idx] = V_rest + 100 * np.sin(np.pi * t_local / 2) * np.exp(-t_local / 1)
    
    # 과분극
    after_idx = (t >= t_spike + 2) & (t < t_spike + 10)
    t_after = t[after_idx] - (t_spike + 2)
    V[after_idx] = V_rest - 10 * np.exp(-t_after / 3)
    
    return V

# 시뮬레이션
t = np.linspace(0, 20, 1000)  # 0-20ms
V = action_potential(t, t_spike=5)

# 시각화
plt.figure(figsize=(10, 4))
plt.plot(t, V, 'b-', linewidth=2)
plt.axhline(y=-70, color='gray', linestyle='--', alpha=0.5, label='Resting potential')
plt.xlabel('Time (ms)', fontsize=12)
plt.ylabel('Membrane Potential (mV)', fontsize=12)
plt.title('Action Potential', fontsize=14, fontweight='bold')
plt.legend()
plt.ylim(-90, 40)
plt.tight_layout()
plt.show()
```

### ✍️ Think! 

위 코드를 실행하고 다음 질문에 답해보세요:

1. 휴지 전위(Resting potential)는 약 몇 mV인가요?
2. 스파이크의 최대 전위는 약 몇 mV인가요?
3. 과분극(Hyperpolarization) 구간에서 무슨 일이 일어나나요?

```{dropdown} 💡 정답 확인
1. 약 **-70 mV**
2. 약 **+30 mV** (실제 값은 +40mV 정도)
3. 과분극 구간에서 막전위가 휴지 전위보다 더 낮아집니다 (약 -80mV). 이는 불응기(Refractory period)와 관련이 있습니다.
```

---

## 5. 스파이크 트레인 (Spike Train)

뉴런은 일련의 스파이크, 즉 **스파이크 트레인(Spike Train)**을 통해 정보를 인코딩합니다.

### 🧑‍💻 Coding Exercise 2: 스파이크 트레인 생성

```{code-block} python
:caption: 포아송 스파이크 트레인 생성

def generate_poisson_spikes(rate, duration, dt=0.001):
    """
    포아송 과정으로 스파이크 트레인 생성
    
    Parameters:
    -----------
    rate : float
        평균 발화율 (Hz)
    duration : float
        시뮬레이션 시간 (초)
    dt : float
        시간 간격 (초)
    
    Returns:
    --------
    spike_times : array
        스파이크 발생 시점들
    """
    n_bins = int(duration / dt)
    spike_prob = rate * dt
    spikes = np.random.random(n_bins) < spike_prob
    spike_times = np.where(spikes)[0] * dt
    return spike_times

# 여러 뉴런의 스파이크 트레인 생성
n_neurons = 5
duration = 1.0  # 1초
rates = [10, 20, 30, 40, 50]  # Hz

plt.figure(figsize=(12, 5))

for i, rate in enumerate(rates):
    spike_times = generate_poisson_spikes(rate, duration)
    plt.eventplot(spike_times, lineoffsets=i+1, colors='black', linewidths=1.5)

plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Neuron #', fontsize=12)
plt.title('Spike Trains (Different Firing Rates)', fontsize=14, fontweight='bold')
plt.yticks(range(1, n_neurons+1), [f'{r} Hz' for r in rates])
plt.xlim(0, duration)
plt.tight_layout()
plt.show()
```

---

## 6. Summary

이번 튜토리얼에서 배운 내용:

::::{grid} 1 2 2 2
:gutter: 2

:::{card} ✅ **계산신경과학의 목표**
Descriptive, Mechanistic, Interpretive 모델
:::

:::{card} ✅ **뉴런 구조**
수상돌기, 세포체, 축삭, 시냅스
:::

:::{card} ✅ **활동 전위**
스파이크의 형태와 발생 메커니즘
:::

:::{card} ✅ **스파이크 트레인**
포아송 과정을 통한 신경 활동 모델링
:::

::::

---

## 📚 Further Reading

- [Neuronal Dynamics - Chapter 1](https://neuronaldynamics.epfl.ch/online/Ch1.html) (무료 온라인 교재)
- Dayan & Abbott, *Theoretical Neuroscience*, Chapter 1

---

## 💬 Feedback

이 튜토리얼에 대한 피드백을 남겨주세요!

```{button-link} https://github.com/yujin/yujin_research/issues/new?title=Feedback:W1D1
:color: secondary
:outline:

📝 피드백 남기기 (GitHub Issues)
```

---

## ⏭️ Next

```{button-ref} day2-neural-anatomy
:color: primary

다음: W1D2 - Neural Anatomy & Physiology →
```
