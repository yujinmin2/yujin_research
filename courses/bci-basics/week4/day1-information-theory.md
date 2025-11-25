---
title: "W4D1 - Information Theory Basics"
subtitle: "정보 이론 기초"
---

# W4D1: Information Theory Basics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W4D1_InformationTheory.ipynb)

---

## 🎯 Learning Objectives

1. 엔트로피(Entropy)의 개념
2. 상호 정보량(Mutual Information)
3. 신경 시스템의 채널 용량

---

## 1. Entropy

$$H(X) = -\sum_{i} p(x_i) \log_2 p(x_i)$$

```python
def entropy(probabilities):
    """엔트로피 계산 (bits)"""
    p = np.array(probabilities)
    p = p[p > 0]  # 0 제외
    return -np.sum(p * np.log2(p))

# 예시
p_uniform = [0.25, 0.25, 0.25, 0.25]
p_skewed = [0.7, 0.1, 0.1, 0.1]

print(f"Uniform: H = {entropy(p_uniform):.2f} bits")
print(f"Skewed:  H = {entropy(p_skewed):.2f} bits")
```

---

## 2. Mutual Information

$$I(X;Y) = H(X) - H(X|Y)$$

자극(X)과 신경 반응(Y) 사이의 상호 정보량은 신경 시스템이 전달하는 정보량을 측정합니다.

---

## ⏭️ Next

```{button-ref} day2-neural-coding
:color: primary

다음: W4D2 - Neural Information Coding →
```
