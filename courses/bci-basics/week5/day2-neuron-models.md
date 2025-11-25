---
title: "W5D2 - Simplified Neuron Models"
subtitle: "단순화된 뉴런 모델"
---

# W5D2: Simplified Neuron Models

---

## 🎯 Learning Objectives

1. Integrate-and-Fire (IF) 모델
2. Leaky Integrate-and-Fire (LIF) 모델
3. Izhikevich 모델

---

## 1. Leaky Integrate-and-Fire Model

$$\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R_m I_{ext}$$

if $V \geq V_{threshold}$: spike & reset to $V_{reset}$

```python
def simulate_lif(I_ext, duration, dt=0.1):
    """LIF 뉴런 시뮬레이션"""
    tau_m = 10  # ms
    V_rest = -70  # mV
    V_thresh = -55  # mV
    V_reset = -75  # mV
    R_m = 10  # MOhm
    
    steps = int(duration / dt)
    V = np.zeros(steps)
    V[0] = V_rest
    spikes = []
    
    for t in range(1, steps):
        dV = (-(V[t-1] - V_rest) + R_m * I_ext[t-1]) / tau_m
        V[t] = V[t-1] + dV * dt
        
        if V[t] >= V_thresh:
            spikes.append(t * dt)
            V[t] = V_reset
    
    return V, spikes
```

---

## 2. Izhikevich Model

더 생물학적으로 현실적이면서도 계산 효율적인 모델입니다.

---

## ⏭️ Next

```{button-ref} ../week6/day1-synaptic-models
:color: primary

다음: W6D1 - Synaptic Models →
```
