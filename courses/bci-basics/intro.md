---
title: BCI 기초 (Computational Neuroscience)
subtitle: 8주 과정 - Brain-Computer Interface & 계산신경과학 입문
---

# 🧠 BCI 기초 코스

계산신경과학의 기초부터 BCI 시스템 구축까지, 8주간의 체계적인 학습 여정입니다.

---

## 📋 코스 개요

| 항목 | 내용 |
|-----|------|
| **기간** | 8주 (주당 약 5-10시간) |
| **수준** | 중급 (선형대수, 미적분, Python 기초 필요) |
| **언어** | 한국어 (영어 자료 병행) |
| **실습** | Python, NumPy, Matplotlib |
| **참고** | Coursera Computational Neuroscience (U. Washington) |

---

## 🎯 학습 목표

이 코스를 완료하면 다음을 할 수 있습니다:

::::{grid} 1 2 2 2
:gutter: 3

:::{card} 1️⃣ **이해하기**
신경 시스템이 정보를 어떻게 인코딩하고 처리하는지 이해
:::

:::{card} 2️⃣ **모델링하기**
뉴런과 신경망의 수학적 모델 구축 및 시뮬레이션
:::

:::{card} 3️⃣ **디코딩하기**
신경 활동에서 의도와 정보 추출
:::

:::{card} 4️⃣ **적용하기**
BCI 시스템의 기본 원리와 구현 이해
:::

::::

---

## 📚 커리큘럼

### Week 1: 신경과학 기초 (Introduction & Basic Neurobiology)

신경과학의 기본 개념과 뉴런의 생물학적 구조를 학습합니다.

- **Day 1**: [Introduction to Computational Neuroscience](week1/day1-intro-neurobiology)
- **Day 2**: [Neural Anatomy & Physiology](week1/day2-neural-anatomy)

**핵심 개념**: 뉴런 구조, 활동전위, 시냅스, 신경전달물질

---

### Week 2: 신경 인코딩 (Neural Encoding)

뉴런이 정보를 어떻게 표현하는지 학습합니다.

- **Day 1**: [Neural Encoding Models](week2/day1-neural-encoding)
- **Day 2**: [Spike Trains & Neural Code](week2/day2-spike-trains)

**핵심 개념**: Spike trains, Tuning curves, Rate coding, Temporal coding

---

### Week 3: 신경 디코딩 (Neural Decoding)

신경 활동에서 정보를 추출하는 방법을 학습합니다.

- **Day 1**: [Neural Decoding Methods](week3/day1-neural-decoding)
- **Day 2**: [BCI Applications](week3/day2-bci-applications)

**핵심 개념**: Bayesian decoding, Population vectors, Neuroprosthetics

---

### Week 4: 정보 이론 (Information Theory)

정보 이론의 신경과학적 적용을 학습합니다.

- **Day 1**: [Information Theory Basics](week4/day1-information-theory)
- **Day 2**: [Neural Information Coding](week4/day2-neural-coding)

**핵심 개념**: Entropy, Mutual information, Channel capacity

---

### Week 5: 뉴런 모델링 (Computing in Carbon)

뉴런의 생물물리학적 모델을 학습합니다.

- **Day 1**: [Hodgkin-Huxley Model](week5/day1-hodgkin-huxley)
- **Day 2**: [Simplified Neuron Models](week5/day2-neuron-models)

**핵심 개념**: Hodgkin-Huxley, Integrate-and-Fire, Izhikevich model

---

### Week 6: 신경망 네트워크 (Computing with Networks)

뉴런들의 연결과 네트워크 동역학을 학습합니다.

- **Day 1**: [Synaptic Models](week6/day1-synaptic-models)
- **Day 2**: [Network Dynamics](week6/day2-network-models)

**핵심 개념**: Synaptic plasticity, Feedforward networks, Recurrent networks

---

### Week 7: 학습 알고리즘 (Learning)

신경과학적 관점의 학습 알고리즘을 학습합니다.

- **Day 1**: [Supervised Learning](week7/day1-supervised-learning)
- **Day 2**: [Reinforcement Learning](week7/day2-reinforcement-learning)

**핵심 개념**: Perceptron, Backpropagation, Reward prediction, Dopamine

---

### Week 8: BCI 시스템 (Brain-Computer Interface)

실제 BCI 시스템의 구성과 미래 방향을 학습합니다.

- **Day 1**: [BCI System Architecture](week8/day1-bci-systems)
- **Day 2**: [Future Directions](week8/day2-future-directions)

**핵심 개념**: EEG, ECoG, Intracortical, Signal processing, Applications

---

## 🛠️ 필요 도구

```{code-block} bash
# 기본 패키지 설치
pip install numpy scipy matplotlib pandas
pip install brian2  # 뉴런 시뮬레이션
pip install mne     # EEG/MEG 분석
pip install sklearn # 머신러닝
```

---

## 📖 참고 자료

### 교재
- **Theoretical Neuroscience** - Dayan & Abbott
- **Neuronal Dynamics** - Gerstner et al. (온라인 무료)
- **Principles of Neural Science** - Kandel et al.

### 온라인 코스
- [Coursera: Computational Neuroscience](https://www.coursera.org/learn/computational-neuroscience) - U. Washington
- [Neuromatch Academy](https://compneuro.neuromatch.io/)

---

## 🚀 시작하기

```{button-ref} week1/day1-intro-neurobiology
:color: primary
:expand:

Week 1, Day 1 시작하기 →
```
