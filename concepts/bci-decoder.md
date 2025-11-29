---
title: "BCI Decoder"
---

# 🎮 BCI Decoder

> 신경 신호를 명령으로 변환하는 알고리즘

---

## 📖 정의

**BCI Decoder**는 측정된 신경 신호를 해석하여 사용자의 의도를 추출하고, 이를 외부 장치 명령으로 변환합니다.

---

## 🔄 디코딩 파이프라인

```{mermaid}
flowchart LR
    subgraph 입력
        N[신경 신호<br/>EEG, Spikes]
    end
    
    subgraph 처리
        P[전처리] --> F[특징 추출] --> D[디코더]
    end
    
    subgraph 출력
        C[명령<br/>움직임, 선택]
    end
    
    N --> P
    D --> C
```

---

## 🧪 디코더 유형

| 유형 | 알고리즘 | 용도 |
|------|----------|------|
| **선형** | Kalman Filter, Linear Regression | 연속 제어 (커서) |
| **분류** | SVM, LDA, CNN | 이산 선택 (P300) |
| **RNN** | LSTM, GRU | 시퀀스 (언어) |
| **베이지안** | Bayesian Filter | 확률적 추정 |

---

## 📊 성능 지표

| 지표 | 설명 |
|------|------|
| **정확도** | 올바른 분류 비율 |
| **ITR** | 정보 전송률 (bits/min) |
| **지연** | 명령 실행 시간 |
| **적응성** | 시간에 따른 성능 유지 |

---

## 🔗 관련 개념

- [베이지안 디코딩](bayesian-decoding)
- [Population Vector](population-vector)
- [EEG](eeg)
- [Intracortical](intracortical)

---

## 📚 관련 수업

- [W3D1: Neural Decoding](../courses/bci-basics/week3/day1-neural-decoding)
- [W8D1: BCI Systems](../courses/bci-basics/week8/day1-bci-systems)
