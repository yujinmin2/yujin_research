---
title: "상호정보량 (Mutual Information)"
---

# 🔀 상호정보량 (Mutual Information)

> 두 변수가 공유하는 정보량

---

## 📖 정의

**상호정보량 I(X;Y)**는 변수 X를 알 때 Y에 대한 불확실성이 얼마나 감소하는지를 측정합니다.

---

## 📐 수식

$$I(X;Y) = H(Y) - H(Y|X) = H(X) - H(X|Y)$$

또는:

$$I(X;Y) = \sum_{x,y} P(x,y) \log_2 \frac{P(x,y)}{P(x)P(y)}$$

---

## 🧠 신경과학 응용

```{mermaid}
flowchart LR
    S[자극 S] --> N[신경 반응 R]
    
    I["I(S;R) = 전달된 정보량"]
```

---

## 🔗 관련 개념

- [엔트로피](entropy)
- [튜닝 커브](tuning-curve)
- [Fisher Information](fisher-information)

---

## 📚 관련 수업

- [W4D1: Information Theory](../courses/bci-basics/week4/day1-information-theory)
- [W4D2: Neural Coding](../courses/bci-basics/week4/day2-neural-coding)
