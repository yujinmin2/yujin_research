---
title: "STDP (Spike-Timing-Dependent Plasticity)"
---

# ⏱️ STDP

> 스파이크 타이밍 의존 가소성

---

## 📖 정의

**STDP**는 시냅스 전후 뉴런의 스파이크 타이밍에 따라 시냅스 강도가 변하는 학습 규칙입니다.

---

## 📐 규칙

```{mermaid}
flowchart LR
    subgraph "Δt < 0: LTD"
        POST1[시냅스후 먼저] --> PRE1[시냅스전 나중]
    end
    
    subgraph "Δt > 0: LTP"
        PRE2[시냅스전 먼저] --> POST2[시냅스후 나중]
    end
```

| 타이밍 | 효과 | 의미 |
|--------|------|------|
| pre → post (Δt > 0) | **LTP** (강화) | 인과 관계 |
| post → pre (Δt < 0) | **LTD** (약화) | 비인과 관계 |

---

## 🧪 수식

$$\Delta w = \begin{cases} A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \\ -A_- e^{\Delta t / \tau_-} & \text{if } \Delta t < 0 \end{cases}$$

---

## 🔗 관련 개념

- [시냅스](synapse)
- [Hebbian Learning](hebbian-learning)
- [Spiking Neural Networks](spiking-nn)

---

## 📚 관련 수업

- [W6D1: Synaptic Models](../courses/bci-basics/week6/day1-synaptic-models)
- [W7D2: Reinforcement Learning](../courses/bci-basics/week7/day2-reinforcement-learning)
