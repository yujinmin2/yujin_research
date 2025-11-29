---
title: "엔트로피 (Entropy)"
---

# 📐 엔트로피 (Entropy)

> 정보의 불확실성 측정

---

## 📖 정의

**엔트로피**는 확률 분포의 불확실성 또는 정보량을 측정하는 양입니다.

---

## 📐 수식

**Shannon Entropy**:

$$H(X) = -\sum_{x} P(x) \log_2 P(x)$$

단위: **bits**

---

## 🧪 예시

| 분포 | 엔트로피 |
|------|----------|
| 확실한 결과 (P=1) | 0 bits |
| 동전 던지기 (P=0.5) | 1 bit |
| 주사위 (P=1/6) | 2.58 bits |

---

## 🔗 관련 개념

- [상호정보량](mutual-information)
- [Fisher Information](fisher-information)
- [Rate Coding](rate-coding)

---

## 📚 관련 수업

- [W4D1: Information Theory](../courses/bci-basics/week4/day1-information-theory)
