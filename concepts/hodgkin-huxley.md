---
title: "Hodgkin-Huxley 모델"
---

# ⚡ Hodgkin-Huxley 모델

> 활동전위의 생물물리학적 수학 모델

---

## 📖 정의

**Hodgkin-Huxley 모델**은 이온 채널의 동역학을 기반으로 활동전위를 설명하는 미분방정식 모델입니다. 1952년 노벨상 수상 연구.

---

## 📐 핵심 방정식

$$C_m \frac{dV}{dt} = I - g_{Na}m^3h(V-E_{Na}) - g_K n^4(V-E_K) - g_L(V-E_L)$$

| 변수 | 의미 |
|------|------|
| $V$ | 막전위 |
| $C_m$ | 막 정전용량 |
| $g_{Na}, g_K, g_L$ | 이온 컨덕턴스 |
| $m, h, n$ | 게이팅 변수 |
| $E_{Na}, E_K, E_L$ | 역전위 |

---

## 🧬 등가 회로

```{mermaid}
flowchart TB
    subgraph 세포막
        C[막 정전용량<br/>Cm]
        Na[Na+ 채널<br/>gNa]
        K[K+ 채널<br/>gK]
        L[누출 채널<br/>gL]
    end
    
    I[전류 I] --> C
    I --> Na
    I --> K
    I --> L
```

---

## ⚙️ 게이팅 동역학

```python
# 게이팅 변수 업데이트
dm/dt = α_m(V)(1-m) - β_m(V)m
dh/dt = α_h(V)(1-h) - β_h(V)h
dn/dt = α_n(V)(1-n) - β_n(V)n
```

---

## 🔗 관련 개념

- [활동전위](action-potential)
- [LIF 모델](lif-model) - 단순화 버전
- [뉴런](neuron)

---

## 📚 관련 수업

- [W5D1: Hodgkin-Huxley Model](../courses/bci-basics/week5/day1-hodgkin-huxley)
- [W5D2: Neuron Models](../courses/bci-basics/week5/day2-neuron-models)
