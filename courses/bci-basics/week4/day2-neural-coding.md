---
title: "W4D2 - Neural Information Coding"
subtitle: "신경 정보 코딩"
---

# W4D2: Neural Information Coding

---

## 🎯 Learning Objectives

1. 스파이크 트레인의 정보량 계산
2. 효율적 코딩 가설 (Efficient Coding Hypothesis)
3. Sparse coding

---

## 1. Efficient Coding Hypothesis

Barlow (1961)의 가설: 감각 시스템은 자연 자극의 통계적 구조에 적응하여 정보 전달을 최대화한다.

---

## 2. Sparse Coding

뇌는 소수의 활성 뉴런으로 정보를 효율적으로 표현합니다.

```python
# Sparse representation example
activity = np.zeros(100)
active_neurons = np.random.choice(100, size=5, replace=False)
activity[active_neurons] = np.random.uniform(0.5, 1.0, size=5)

plt.figure(figsize=(12, 3))
plt.bar(range(100), activity)
plt.xlabel('Neuron Index')
plt.ylabel('Activity')
plt.title('Sparse Neural Representation')
plt.show()
```

---

## ⏭️ Next

```{button-ref} ../week5/day1-hodgkin-huxley
:color: primary

다음: W5D1 - Hodgkin-Huxley Model →
```
