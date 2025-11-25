---
title: "W2D1 - Neural Encoding Models"
subtitle: "신경 인코딩 모델"
---

# W2D1: Neural Encoding Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W2D1_NeuralEncoding.ipynb)

---

## 📋 Overview

**핵심 질문**: 뉴런은 외부 세계의 정보를 어떻게 표현하는가?

---

## 🎯 Learning Objectives

1. Tuning curve의 개념 이해
2. Rate coding vs Temporal coding 구분
3. 선형 필터 모델 (Linear-Nonlinear model) 이해

---

## 1. Tuning Curves

뉴런의 **튜닝 커브(Tuning Curve)**는 특정 자극 특성에 대한 발화율의 관계를 나타냅니다.

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_tuning_curve(theta, preferred_theta, amplitude=50, width=30):
    """가우시안 튜닝 커브"""
    return amplitude * np.exp(-0.5 * ((theta - preferred_theta) / width)**2)

# 방향 선택적 뉴런
theta = np.linspace(0, 360, 100)
preferred_directions = [45, 135, 225, 315]

plt.figure(figsize=(10, 4))
for pref in preferred_directions:
    rate = gaussian_tuning_curve(theta, pref)
    plt.plot(theta, rate, label=f'Preferred: {pref}°')

plt.xlabel('Stimulus Direction (°)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Direction Tuning Curves')
plt.legend()
plt.show()
```

---

## 2. Rate Coding vs Temporal Coding

| Coding Type | 정보 표현 방식 |
|------------|--------------|
| **Rate Coding** | 발화율(Hz)로 정보 인코딩 |
| **Temporal Coding** | 스파이크 타이밍으로 정보 인코딩 |

---

## ⏭️ Next

```{button-ref} day2-spike-trains
:color: primary

다음: W2D2 - Spike Trains →
```
