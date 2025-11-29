---
title: "🧠 BCI 지식 그래프"
subtitle: "개념들의 연결 관계"
---

# 🧠 BCI 지식 그래프

> 옵시디언 스타일로 BCI 핵심 개념들의 연결 관계를 시각화했습니다. 각 노드를 클릭하면 상세 페이지로 이동합니다.

---

## 🗺️ 전체 개념 맵

```{mermaid}
flowchart TB
    subgraph W1["🔬 Week 1: 신경과학 기초"]
        N[뉴런<br/>Neuron]
        AP[활동전위<br/>Action Potential]
        SYN[시냅스<br/>Synapse]
        NT[신경전달물질<br/>Neurotransmitter]
    end
    
    subgraph W2["📊 Week 2: 신경 인코딩"]
        TC[튜닝 커브<br/>Tuning Curve]
        RC[Rate Coding]
        TempC[Temporal Coding]
        ST[스파이크 트레인<br/>Spike Train]
        ISI[ISI 분석]
        PSTH[PSTH]
    end
    
    subgraph W3["🎯 Week 3: 신경 디코딩"]
        BD[베이지안 디코딩<br/>Bayesian Decoding]
        PV[Population Vector]
        ML[Maximum Likelihood]
    end
    
    subgraph W4["📐 Week 4: 정보 이론"]
        ENT[엔트로피<br/>Entropy]
        MI[상호정보량<br/>Mutual Information]
        FC[Fisher Information]
    end
    
    subgraph W5["⚡ Week 5: 뉴런 모델링"]
        HH[Hodgkin-Huxley]
        LIF[Leaky Integrate-and-Fire]
        IF[Integrate-and-Fire]
    end
    
    subgraph W6["🕸️ Week 6: 신경망"]
        RNN[Recurrent Networks]
        SNN[Spiking Neural Networks]
        STDP[STDP]
    end
    
    subgraph W7["🤖 Week 7: 학습"]
        SL[Supervised Learning]
        RL[Reinforcement Learning]
        HL[Hebbian Learning]
    end
    
    subgraph W8["🧩 Week 8: BCI 시스템"]
        EEG[EEG]
        ECOG[ECoG]
        INTRA[Intracortical]
        DEC[Decoder]
        APP[Applications]
    end
    
    %% 연결 관계
    N --> AP
    AP --> SYN
    SYN --> NT
    
    AP --> ST
    ST --> TC
    ST --> ISI
    ST --> PSTH
    TC --> RC
    TC --> TempC
    
    ST --> BD
    TC --> PV
    PV --> ML
    
    ST --> ENT
    TC --> MI
    MI --> FC
    
    AP --> HH
    HH --> LIF
    LIF --> IF
    
    SYN --> STDP
    LIF --> SNN
    SNN --> RNN
    
    STDP --> HL
    HL --> SL
    SL --> RL
    
    BD --> DEC
    PV --> DEC
    EEG --> DEC
    ECOG --> DEC
    INTRA --> DEC
    DEC --> APP
```

---

## 📚 개념별 바로가기

### 🔬 Week 1: 신경과학 기초

| 개념 | 설명 | 링크 |
|------|------|------|
| **뉴런 (Neuron)** | 신경계의 기본 단위 | [상세보기](neuron) |
| **활동전위 (Action Potential)** | 뉴런의 전기 신호 | [상세보기](action-potential) |
| **시냅스 (Synapse)** | 뉴런 간 연결부 | [상세보기](synapse) |
| **신경전달물질** | 시냅스 신호 전달 물질 | [상세보기](neurotransmitter) |

### 📊 Week 2: 신경 인코딩

| 개념 | 설명 | 링크 |
|------|------|------|
| **튜닝 커브 (Tuning Curve)** | 자극-반응 관계 함수 | [상세보기](tuning-curve) |
| **스파이크 트레인** | 뉴런 발화 시퀀스 | [상세보기](spike-train) |
| **Rate Coding** | 발화율 기반 인코딩 | [상세보기](rate-coding) |
| **PSTH** | 자극 정렬 히스토그램 | [상세보기](psth) |

### 🎯 Week 3: 신경 디코딩

| 개념 | 설명 | 링크 |
|------|------|------|
| **베이지안 디코딩** | 확률적 자극 추정 | [상세보기](bayesian-decoding) |
| **Population Vector** | 집단 활동 벡터 | [상세보기](population-vector) |
| **Maximum Likelihood** | 최대우도 추정 | [상세보기](maximum-likelihood) |

### 📐 Week 4: 정보 이론

| 개념 | 설명 | 링크 |
|------|------|------|
| **엔트로피 (Entropy)** | 정보의 불확실성 | [상세보기](entropy) |
| **상호정보량 (MI)** | 공유 정보량 | [상세보기](mutual-information) |
| **Fisher Information** | 추정 정밀도 | [상세보기](fisher-information) |

### ⚡ Week 5: 뉴런 모델링

| 개념 | 설명 | 링크 |
|------|------|------|
| **Hodgkin-Huxley** | 생물물리학적 뉴런 모델 | [상세보기](hodgkin-huxley) |
| **LIF 모델** | 간소화된 뉴런 모델 | [상세보기](lif-model) |
| **Integrate-and-Fire** | 적분-발화 모델 | [상세보기](integrate-fire) |

### 🕸️ Week 6: 신경망

| 개념 | 설명 | 링크 |
|------|------|------|
| **Spiking Neural Networks** | 스파이킹 신경망 | [상세보기](spiking-nn) |
| **STDP** | 스파이크 타이밍 가소성 | [상세보기](stdp) |
| **Recurrent Networks** | 순환 신경망 | [상세보기](recurrent-networks) |

### 🤖 Week 7: 학습

| 개념 | 설명 | 링크 |
|------|------|------|
| **Supervised Learning** | 지도 학습 | [상세보기](supervised-learning) |
| **Reinforcement Learning** | 강화 학습 | [상세보기](reinforcement-learning) |
| **Hebbian Learning** | 헵 학습 규칙 | [상세보기](hebbian-learning) |

### 🧩 Week 8: BCI 시스템

| 개념 | 설명 | 링크 |
|------|------|------|
| **EEG** | 뇌전도 | [상세보기](eeg) |
| **ECoG** | 피질뇌전도 | [상세보기](ecog) |
| **Intracortical** | 피질내 기록 | [상세보기](intracortical) |
| **BCI Decoder** | 신호 해석기 | [상세보기](bci-decoder) |

---

## 🔗 개념 연결 패턴

### 수직적 연결 (기초 → 응용)

```{mermaid}
flowchart LR
    A[뉴런] --> B[스파이크] --> C[인코딩] --> D[디코딩] --> E[BCI]
    
    style A fill:#e74c3c
    style B fill:#f39c12
    style C fill:#2ecc71
    style D fill:#3498db
    style E fill:#9b59b6
```

### 수평적 연결 (같은 레벨)

```{mermaid}
flowchart LR
    subgraph 신호처리
        EEG --- ECoG --- INTRA[Intracortical]
    end
    
    subgraph 모델링
        HH[Hodgkin-Huxley] --- LIF --- IF
    end
    
    subgraph 학습
        SL[Supervised] --- RL[Reinforcement] --- HL[Hebbian]
    end
```

---

## 🎓 학습 경로 추천

### 경로 1: BCI 엔지니어
```
신경과학 기초 → 신호처리(EEG) → 디코딩 → BCI 시스템
```

### 경로 2: 계산신경과학 연구자
```
뉴런 모델링 → 정보이론 → 신경망 → 학습 알고리즘
```

### 경로 3: 빠른 실습
```
스파이크 트레인 → PSTH → Population Vector → Decoder
```
