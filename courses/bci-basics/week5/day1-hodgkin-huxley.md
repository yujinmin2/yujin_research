---
title: "W5D1 - Hodgkin-Huxley Model"
subtitle: "호지킨-헉슬리 모델"
---

# W5D1: Hodgkin-Huxley Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W5D1_HodgkinHuxley.ipynb)

---

## 📋 Overview

**1952년 노벨상 수상 연구**: Hodgkin과 Huxley는 오징어 거대 축삭에서 활동 전위 생성의 이온 메커니즘을 수학적으로 모델링했습니다.

---

## 🎯 Learning Objectives

1. Hodgkin-Huxley 방정식 이해
2. 이온 채널 게이팅 메커니즘
3. Python으로 HH 모델 시뮬레이션

---

## 1. The Hodgkin-Huxley Equations

$$C_m \frac{dV}{dt} = I_{ext} - g_{Na} m^3 h (V - E_{Na}) - g_K n^4 (V - E_K) - g_L (V - E_L)$$

| 파라미터 | 설명 |
|---------|------|
| $C_m$ | 막 정전용량 |
| $g_{Na}, g_K, g_L$ | 최대 컨덕턴스 |
| $m, h, n$ | 게이팅 변수 |
| $E_{Na}, E_K, E_L$ | 역전 전위 |

---

## 2. Gating Variables

```python
def alpha_n(V):
    return 0.01 * (10 - V) / (np.exp((10 - V) / 10) - 1)

def beta_n(V):
    return 0.125 * np.exp(-V / 80)

def alpha_m(V):
    return 0.1 * (25 - V) / (np.exp((25 - V) / 10) - 1)

def beta_m(V):
    return 4 * np.exp(-V / 18)

def alpha_h(V):
    return 0.07 * np.exp(-V / 20)

def beta_h(V):
    return 1 / (np.exp((30 - V) / 10) + 1)
```

---

## ⏭️ Next

```{button-ref} day2-neuron-models
:color: primary

다음: W5D2 - Simplified Neuron Models →
```
