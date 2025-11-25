---
title: "W3D1 - Neural Decoding Methods"
subtitle: "신경 디코딩 방법론"
---

# W3D1: Neural Decoding Methods

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W3D1_NeuralDecoding.ipynb)

---

## 📋 Overview

**핵심 질문**: 신경 활동으로부터 뇌가 보고/의도하는 것을 어떻게 추정할 수 있는가?

이것이 BCI의 핵심 기술입니다!

---

## 🎯 Learning Objectives

1. 베이지안 디코딩 원리 이해
2. Population Vector 디코딩
3. 칼만 필터 기반 디코딩

---

## 1. Bayesian Decoding

$$P(s|r) = \frac{P(r|s) P(s)}{P(r)}$$

- $P(s|r)$: 신경 반응이 주어졌을 때 자극의 확률 (posterior)
- $P(r|s)$: 자극이 주어졌을 때 신경 반응의 확률 (likelihood)
- $P(s)$: 자극의 사전 확률 (prior)

---

## 2. Population Vector Decoding

```python
def population_vector_decode(spike_counts, preferred_directions):
    """
    Population Vector를 사용한 방향 디코딩
    """
    pv_x = np.sum(spike_counts * np.cos(np.radians(preferred_directions)))
    pv_y = np.sum(spike_counts * np.sin(np.radians(preferred_directions)))
    
    decoded_direction = np.degrees(np.arctan2(pv_y, pv_x))
    return decoded_direction % 360
```

---

## ⏭️ Next

```{button-ref} day2-bci-applications
:color: primary

다음: W3D2 - BCI Applications →
```
