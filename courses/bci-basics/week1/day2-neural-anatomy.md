---
title: "W1D2 - Neural Anatomy & Physiology"
subtitle: "신경 해부학과 생리학"
---

# W1D2: Neural Anatomy & Physiology

**신경 해부학과 생리학**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W1D2_NeuralAnatomy.ipynb)

---

## 📋 Overview

| 항목 | 내용 |
|-----|------|
| **소요 시간** | 약 60분 |
| **학습 목표** | 뇌의 구조와 신경 신호 전달 메커니즘 이해 |

---

## 🎯 Learning Objectives

1. 뇌의 주요 영역과 기능 이해
2. 이온 채널과 막전위의 관계 이해
3. 시냅스 전달의 원리 이해

---

## 1. 뇌의 구조 (Brain Structure)

### 주요 뇌 영역

| 영역 | 기능 |
|-----|------|
| **대뇌 피질 (Cerebral Cortex)** | 고등 인지 기능, 감각 처리, 운동 제어 |
| **해마 (Hippocampus)** | 기억 형성 및 공간 탐색 |
| **기저핵 (Basal Ganglia)** | 운동 제어, 보상 학습 |
| **소뇌 (Cerebellum)** | 운동 조절, 균형 |
| **뇌간 (Brainstem)** | 생명 유지 기능 |

---

## 2. 이온 채널과 막전위

### Nernst 방정식

```{code-block} python
:caption: Nernst 방정식으로 평형 전위 계산

import numpy as np

def nernst_potential(z, C_out, C_in, T=310):
    """
    Nernst 방정식으로 이온의 평형 전위 계산
    
    E = (RT/zF) * ln(C_out/C_in)
    
    Parameters:
    -----------
    z : int
        이온의 원자가 (+1 for Na+, K+; -1 for Cl-; +2 for Ca2+)
    C_out : float
        세포 외부 이온 농도 (mM)
    C_in : float
        세포 내부 이온 농도 (mM)
    T : float
        온도 (K), 기본값 310K (37°C)
    
    Returns:
    --------
    E : float
        평형 전위 (mV)
    """
    R = 8.314  # J/(mol·K)
    F = 96485  # C/mol
    
    E = (R * T / (z * F)) * np.log(C_out / C_in) * 1000  # mV로 변환
    return E

# 주요 이온의 평형 전위 계산
ions = {
    'K+': {'z': 1, 'C_out': 5, 'C_in': 140},
    'Na+': {'z': 1, 'C_out': 145, 'C_in': 12},
    'Cl-': {'z': -1, 'C_out': 120, 'C_in': 4},
    'Ca2+': {'z': 2, 'C_out': 2, 'C_in': 0.0001},
}

print("이온별 평형 전위 (Nernst Potential)")
print("=" * 40)
for ion, params in ions.items():
    E = nernst_potential(**params)
    print(f"{ion:5s}: {E:+.1f} mV")
```

---

## 3. 시냅스 전달 (Synaptic Transmission)

### 시냅스의 종류

- **화학적 시냅스**: 신경전달물질을 통한 신호 전달
- **전기적 시냅스**: Gap junction을 통한 직접 전기 신호 전달

### 주요 신경전달물질

| 신경전달물질 | 유형 | 주요 기능 |
|------------|------|----------|
| **Glutamate** | 흥분성 | 학습, 기억 |
| **GABA** | 억제성 | 신경 활동 조절 |
| **Dopamine** | 조절성 | 보상, 동기 |
| **Acetylcholine** | 흥분성/조절성 | 근육 제어, 주의 |

---

## 4. Summary

- 뇌는 기능적으로 구분된 여러 영역으로 구성
- 막전위는 이온 농도 차이에 의해 결정
- 시냅스는 뉴런 간 정보 전달의 핵심

---

## ⏭️ Next

```{button-ref} ../week2/day1-neural-encoding
:color: primary

다음: W2D1 - Neural Encoding →
```
