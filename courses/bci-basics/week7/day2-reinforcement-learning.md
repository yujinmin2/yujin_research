---
title: "W7D2 - Reinforcement Learning"
subtitle: "강화 학습"
---

# W7D2: Reinforcement Learning in the Brain

---

## 🎯 Learning Objectives

1. 보상 예측 오류 (Reward Prediction Error)
2. 도파민과 학습
3. TD Learning

---

## 1. Dopamine & Reward

Schultz et al. (1997)의 발견: 도파민 뉴런은 **보상 예측 오류**를 인코딩합니다.

$$\delta = r + \gamma V(s') - V(s)$$

---

## 2. Temporal Difference Learning

```python
def td_update(V, s, s_next, r, alpha=0.1, gamma=0.9):
    """TD(0) 업데이트"""
    delta = r + gamma * V[s_next] - V[s]  # RPE
    V[s] = V[s] + alpha * delta
    return V, delta
```

---

## 3. Actor-Critic Model

- **Actor**: 행동 선택 (Dorsal Striatum)
- **Critic**: 가치 평가 (Ventral Striatum)

---

## ⏭️ Next

```{button-ref} ../week8/day1-bci-systems
:color: primary

다음: W8D1 - BCI Systems →
```
