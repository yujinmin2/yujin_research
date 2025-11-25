---
title: "W8D1 - BCI System Architecture"
subtitle: "BCI 시스템 구조"
---

# W8D1: BCI System Architecture

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W8D1_BCISystems.ipynb)

---

## 📋 Overview

BCI(Brain-Computer Interface)는 뇌 활동을 측정하여 외부 기기를 제어하는 시스템입니다.

---

## 🎯 Learning Objectives

1. BCI 시스템의 구성 요소
2. 신호 획득 방법 (EEG, ECoG, Intracortical)
3. 신호 처리 파이프라인

---

## 1. BCI System Components

```
[뇌] → [신호 획득] → [전처리] → [특징 추출] → [분류/디코딩] → [출력 장치]
```

---

## 2. Signal Acquisition Methods

| 방법 | 침습성 | 해상도 | 장점 |
|-----|-------|-------|------|
| **EEG** | 비침습 | 낮음 | 안전, 휴대 가능 |
| **ECoG** | 반침습 | 중간 | 높은 SNR |
| **Intracortical** | 침습 | 높음 | 단일 뉴런 기록 |

---

## 3. Signal Processing Pipeline

```python
# EEG 처리 예시
import mne

# 1. 필터링
raw_filtered = raw.filter(l_freq=1, h_freq=40)

# 2. 아티팩트 제거
ica = mne.preprocessing.ICA(n_components=20)
ica.fit(raw_filtered)

# 3. 특징 추출
# - Band power (alpha, beta, gamma)
# - Common Spatial Patterns (CSP)

# 4. 분류
# - LDA, SVM, Neural Networks
```

---

## ⏭️ Next

```{button-ref} day2-future-directions
:color: primary

다음: W8D2 - Future Directions →
```
