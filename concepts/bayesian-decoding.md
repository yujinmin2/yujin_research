---
title: "베이지안 디코딩 (Bayesian Decoding)"
---

# 🎯 베이지안 디코딩 (Bayesian Decoding)

> 신경 반응으로부터 자극을 확률적으로 추정

---

## 📖 정의

**베이지안 디코딩**은 관측된 신경 반응 $r$로부터 자극 $s$의 확률 분포를 추정합니다.

---

## 📐 베이즈 정리

$$P(s|r) = \frac{P(r|s) \cdot P(s)}{P(r)}$$

| 요소 | 이름 | 의미 |
|------|------|------|
| $P(s\|r)$ | 사후확률 | 반응이 주어졌을 때 자극 확률 |
| $P(r\|s)$ | 우도 | 자극이 주어졌을 때 반응 확률 |
| $P(s)$ | 사전확률 | 자극의 기본 분포 |

---

## 🔬 디코딩 과정

```{mermaid}
flowchart LR
    R[신경 반응 r] --> L[우도 P흐r\s흐]
    P[사전확률 P흐s흐] --> B[베이즈 정리]
    L --> B
    B --> POST[사후확률 P흐s\r흐]
    POST --> EST[추정값 ŝ]
```

---

## 🧪 추정 방법

| 방법 | 수식 | 특징 |
|------|------|------|
| **MAP** | $\hat{s} = \arg\max P(s\|r)$ | 가장 확률 높은 값 |
| **Mean** | $\hat{s} = E[s\|r]$ | 평균값 |
| **Median** | $\hat{s} = \text{median}(P(s\|r))$ | 중앙값 |

---

## 🔗 관련 개념

- [튜닝 커브](tuning-curve) - 우도 함수 구성
- [Population Vector](population-vector) - 단순 디코딩
- [Maximum Likelihood](maximum-likelihood)
- [Fisher Information](fisher-information)

---

## 📚 관련 수업

- [W3D1: Neural Decoding](../courses/bci-basics/week3/day1-neural-decoding)
- [W3D2: BCI Applications](../courses/bci-basics/week3/day2-bci-applications)
