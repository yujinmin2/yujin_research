---
title: "W6D1 - Synaptic Models"
subtitle: "시냅스 모델"
---

# W6D1: Synaptic Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W6D1_SynapticModels.ipynb)

---

## 🎯 Learning Objectives

1. 시냅스 전류 모델링
2. AMPA, NMDA, GABA 수용체 모델
3. 시냅스 가소성 (Synaptic Plasticity)

---

## 1. Synaptic Current Models

$$I_{syn} = g_{syn}(t) \cdot (V - E_{syn})$$

```python
def alpha_synapse(t, t_spike, tau=2):
    """Alpha function synapse model"""
    dt = t - t_spike
    if dt < 0:
        return 0
    return (dt / tau) * np.exp(1 - dt / tau)
```

---

## 2. Synaptic Plasticity

### STDP (Spike-Timing-Dependent Plasticity)

$$\Delta w = \begin{cases} A_+ e^{-\Delta t / \tau_+} & \text{if } \Delta t > 0 \\ -A_- e^{\Delta t / \tau_-} & \text{if } \Delta t < 0 \end{cases}$$

---

## ⏭️ Next

```{button-ref} day2-network-models
:color: primary

다음: W6D2 - Network Dynamics →
```
