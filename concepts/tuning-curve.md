---
title: "튜닝 커브 (Tuning Curve)"
---

# 📈 튜닝 커브 (Tuning Curve)

> 자극 특성과 발화율의 관계 함수

---

## 📖 정의

**튜닝 커브**는 특정 자극 파라미터에 대한 뉴런의 발화율을 나타내는 함수입니다.

```{mermaid}
flowchart LR
    S[자극 특성<br/>방향, 주파수 등] --> N[뉴런] --> R[발화율<br/>Hz]
```

---

## 📊 유형

| 유형 | 형태 | 예시 |
|------|------|------|
| **Gaussian** | 종 모양 | V1 방향 선택성 |
| **Cosine** | 코사인 | 운동피질 |
| **Sigmoid** | S자형 | 강도 인코딩 |
| **Bandpass** | 대역통과 | 청각 주파수 |

---

## 🧪 예시: 방향 튜닝

```python
import numpy as np

def gaussian_tuning(theta, pref_theta, amplitude=50, width=30):
    """가우시안 튜닝 커브"""
    diff = np.abs(theta - pref_theta)
    return amplitude * np.exp(-0.5 * (diff / width)**2)

# 선호 방향: 45도
# theta=45에서 최대 발화율
```

---

## 🔗 관련 개념

- [Rate Coding](rate-coding)
- [Population Vector](population-vector)
- [스파이크 트레인](spike-train)

---

## 📚 관련 수업

- [W2D1: Neural Encoding Models](../courses/bci-basics/week2/day1-neural-encoding)
