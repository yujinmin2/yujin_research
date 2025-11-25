---
title: "W7D1 - Supervised Learning"
subtitle: "지도 학습"
---

# W7D1: Supervised Learning in the Brain

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W7D1_SupervisedLearning.ipynb)

---

## 🎯 Learning Objectives

1. 퍼셉트론 (Perceptron)
2. 역전파 (Backpropagation)
3. 뇌에서의 지도 학습

---

## 1. Perceptron

$$y = \sigma\left(\sum_i w_i x_i + b\right)$$

```python
def perceptron(x, w, b):
    """단순 퍼셉트론"""
    return 1 if np.dot(w, x) + b > 0 else 0
```

---

## 2. Backpropagation

$$\frac{\partial L}{\partial w_{ij}} = \frac{\partial L}{\partial a_j} \cdot \frac{\partial a_j}{\partial w_{ij}}$$

### Credit Assignment Problem
- 뇌가 실제로 역전파를 사용하는가?
- 대안: Feedback alignment, Predictive coding

---

## ⏭️ Next

```{button-ref} day2-reinforcement-learning
:color: primary

다음: W7D2 - Reinforcement Learning →
```
