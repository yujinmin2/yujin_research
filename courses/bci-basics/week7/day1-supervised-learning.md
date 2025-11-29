---
title: "W7D1 - Supervised Learning"
subtitle: "신경과학 관점의 지도 학습"
---

# W7D1: Supervised Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujinmin2/yujin_research/blob/main/notebooks/W7D1_SupervisedLearning.ipynb)

---

## 📋 Overview

**핵심 질문**: 뇌는 어떻게 입출력 매핑을 학습하는가?

**지도 학습**은 입력-출력 쌍으로부터 함수를 학습하는 방법입니다.

```{mermaid}
flowchart LR
    X[입력 x] --> NET[네트워크]
    NET --> Y[출력 ŷ]
    T[정답 y] --> ERR[오차 계산]
    Y --> ERR
    ERR --> |역전파| NET
```

---

## 🎯 Learning Objectives

1. **퍼셉트론** 학습 규칙 이해
2. **경사하강법** 구현
3. **역전파** 알고리즘 이해
4. **생물학적 타당성** 논의

---

## 1. 퍼셉트론 (Perceptron)

### 1.1 모델

$$y = \sigma\left(\sum_i w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

### 1.2 학습 규칙

$$\Delta w_i = \eta (y_{target} - y_{pred}) x_i$$

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = 0
        self.lr = lr
    
    def predict(self, x):
        return 1 if np.dot(self.w, x) + self.b > 0 else 0
    
    def train(self, X, y, epochs=100):
        history = []
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = yi - pred
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    errors += 1
            history.append(errors)
            if errors == 0:
                break
        return history

# AND 게이트 학습
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])
y_xor = np.array([0, 1, 1, 0])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (y, name) in zip(axes, [(y_and, 'AND'), (y_or, 'OR'), (y_xor, 'XOR')]):
    p = Perceptron(2)
    history = p.train(X, y, epochs=20)
    
    # 결정 경계
    xx = np.linspace(-0.5, 1.5, 100)
    if p.w[1] != 0:
        yy = -(p.w[0] * xx + p.b) / p.w[1]
        ax.plot(xx, yy, 'g-', linewidth=2, label='Decision boundary')
    
    # 데이터 포인트
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=100, label='Class 0')
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', s=100, label='Class 1')
    
    accuracy = sum([p.predict(xi) == yi for xi, yi in zip(X, y)]) / len(y)
    ax.set_title(f'{name} Gate\nAccuracy: {accuracy*100:.0f}%')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. 다층 퍼셉트론 (MLP)

### 2.1 구조

```{mermaid}
flowchart LR
    subgraph Input
        X1[x₁]
        X2[x₂]
    end
    
    subgraph Hidden
        H1[h₁]
        H2[h₂]
    end
    
    subgraph Output
        Y[y]
    end
    
    X1 --> H1
    X1 --> H2
    X2 --> H1
    X2 --> H2
    H1 --> Y
    H2 --> Y
```

### 2.2 순전파 (Forward Pass)

$$\mathbf{h} = \sigma(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)})$$
$$\mathbf{y} = \sigma(\mathbf{W}^{(2)} \mathbf{h} + \mathbf{b}^{(2)})$$

### 2.3 구현

```python
class MLP:
    def __init__(self, layer_sizes, lr=0.5):
        self.layers = []
        self.lr = lr
        
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.5
            b = np.zeros(layer_sizes[i+1])
            self.layers.append({'W': W, 'b': b})
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_deriv(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        activations = [x]
        for layer in self.layers:
            z = layer['W'] @ activations[-1] + layer['b']
            a = self.sigmoid(z)
            activations.append(a)
        return activations
    
    def backward(self, activations, y_true):
        deltas = [None] * len(self.layers)
        
        # 출력층 오차
        output_error = y_true - activations[-1]
        deltas[-1] = output_error * self.sigmoid_deriv(activations[-1])
        
        # 역전파
        for i in range(len(self.layers) - 2, -1, -1):
            error = self.layers[i+1]['W'].T @ deltas[i+1]
            deltas[i] = error * self.sigmoid_deriv(activations[i+1])
        
        # 가중치 업데이트
        for i in range(len(self.layers)):
            self.layers[i]['W'] += self.lr * np.outer(deltas[i], activations[i])
            self.layers[i]['b'] += self.lr * deltas[i]
    
    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for xi, yi in zip(X, y):
                activations = self.forward(xi)
                loss = 0.5 * np.sum((yi - activations[-1])**2)
                total_loss += loss
                self.backward(activations, yi)
            losses.append(total_loss / len(X))
        return losses
    
    def predict(self, x):
        return self.forward(x)[-1]

# XOR 문제 해결
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])

mlp = MLP([2, 4, 1], lr=1.0)
losses = mlp.train(X_xor, y_xor, epochs=5000)

# 결과
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 학습 곡선
axes[0].plot(losses, 'b-')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Learning Curve')
axes[0].set_yscale('log')

# 결정 경계
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 50), np.linspace(-0.5, 1.5, 50))
Z = np.array([mlp.predict(np.array([x, y]))[0] for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

axes[1].contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.7)
axes[1].scatter(X_xor[y_xor.flatten()==0, 0], X_xor[y_xor.flatten()==0, 1], 
                c='blue', s=100, edgecolors='black')
axes[1].scatter(X_xor[y_xor.flatten()==1, 0], X_xor[y_xor.flatten()==1, 1], 
                c='red', s=100, edgecolors='black')
axes[1].set_title('XOR Decision Boundary')
axes[1].set_xlabel('x₁')
axes[1].set_ylabel('x₂')

plt.tight_layout()
plt.show()

# 예측 결과
print("XOR 예측 결과:")
for xi, yi in zip(X_xor, y_xor):
    pred = mlp.predict(xi)[0]
    print(f"  {xi} → {pred:.3f} (정답: {yi[0]})")
```

---

## 3. 역전파의 생물학적 타당성

### 3.1 문제점

| 문제 | 설명 |
|------|------|
| **가중치 수송** | 역전파 시 동일한 W 사용 |
| **미분 필요** | 뉴런이 미분 계산? |
| **시간적 비국소성** | 순전파 완료 후 역전파 |
| **양방향 시냅스** | 한 시냅스로 양방향 전달 |

### 3.2 생물학적 대안

```{mermaid}
flowchart TB
    subgraph 역전파
        BP[표준 역전파<br/>Weight Transport]
    end
    
    subgraph 대안
        FA[Feedback Alignment<br/>랜덤 역전파]
        PREDICT[Predictive Coding<br/>예측 오차]
        EQUIV[Equilibrium Propagation<br/>평형 전파]
        LOCAL[Local Learning<br/>국소 학습]
    end
    
    BP --> FA
    BP --> PREDICT
    BP --> EQUIV
    BP --> LOCAL
```

---

## 4. Delta Rule과 LMS

### 4.1 Delta Rule

$$\Delta w_{ij} = \eta \cdot (y_j^{target} - y_j) \cdot x_i$$

### 4.2 생물학적 해석

| 요소 | 수학적 | 생물학적 |
|------|--------|----------|
| $\eta$ | 학습률 | 가소성 조절 |
| $y^{target} - y$ | 오차 | 도파민 신호? |
| $x_i$ | 입력 | 시냅스전 활동 |
| $y_j$ | 출력 | 시냅스후 활동 |

---

## 📝 실습 문제

### 문제 1: MNIST 분류
MLP로 MNIST 손글씨 숫자 분류기를 구현하세요.

### 문제 2: Feedback Alignment
랜덤 역방향 가중치로 학습이 가능한지 확인하세요.

### 문제 3: 온라인 학습
미니배치 대신 단일 샘플로 학습하는 온라인 버전을 구현하세요.

---

## 🔗 관련 개념

- [Supervised Learning](../../concepts/supervised-learning)
- [Hebbian Learning](../../concepts/hebbian-learning)
- [STDP](../../concepts/stdp)

---

## 📚 참고 자료

- Dayan & Abbott, Chapter 10
- Lillicrap et al., "Backpropagation and the brain"
- Rumelhart et al. (1986): 역전파 원논문

---

## ⏭️ Next

```{button-ref} day2-reinforcement-learning
:color: primary

다음: W7D2 - Reinforcement Learning →
```
