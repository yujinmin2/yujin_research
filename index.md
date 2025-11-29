---
title: Yujin Research Lab
subtitle: BCI & Computational Neuroscience 연구 플랫폼
---

# 🧠 Yujin Research Lab에 오신 것을 환영합니다

Brain-Computer Interface(BCI)와 계산신경과학을 공부하고 연구하는 공간입니다.

::::{grid} 1 2 2 4
:gutter: 3

:::{card} 📚 **BCI 기초**
:link: courses/bci-basics/intro
8주 과정의 계산신경과학 기초 코스
:::

:::{card} 🧠 **지식 그래프**
:link: concepts/index
개념 연결 맵 & 용어 사전
:::

:::{card} 📂 **자료실**
:link: resources/datasets
데이터셋, 논문, 도구
:::

:::{card} 💼 **프로젝트**
:link: projects/index
연구 프로젝트 쇼케이스
:::

::::

---

## 📖 콘텐츠 구성

```{list-table}
:header-rows: 1
:widths: 5 20 35 40

* - Week
  - 주제
  - 핵심 내용
  - 수업 바로가기
* - 1
  - 신경과학 기초
  - Introduction & Basic Neurobiology
  - [D1: 신경생물학](courses/bci-basics/week1/day1-intro-neurobiology) | [D2: 신경해부학](courses/bci-basics/week1/day2-neural-anatomy)
* - 2
  - 신경 인코딩
  - What do Neurons Encode?
  - [D1: 신경인코딩](courses/bci-basics/week2/day1-neural-encoding) | [D2: 스파이크트레인](courses/bci-basics/week2/day2-spike-trains)
* - 3
  - 신경 디코딩
  - Extracting Information from Neurons
  - [D1: 신경디코딩](courses/bci-basics/week3/day1-neural-decoding) | [D2: BCI응용](courses/bci-basics/week3/day2-bci-applications)
* - 4
  - 정보 이론
  - Information Theory & Neural Coding
  - [D1: 정보이론](courses/bci-basics/week4/day1-information-theory) | [D2: 신경코딩](courses/bci-basics/week4/day2-neural-coding)
* - 5
  - 뉴런 모델링
  - Computing in Carbon (Hodgkin-Huxley)
  - [D1: H-H모델](courses/bci-basics/week5/day1-hodgkin-huxley) | [D2: 뉴런모델](courses/bci-basics/week5/day2-neuron-models)
* - 6
  - 신경망 네트워크
  - Computing with Networks
  - [D1: 시냅스모델](courses/bci-basics/week6/day1-synaptic-models) | [D2: 네트워크](courses/bci-basics/week6/day2-network-models)
* - 7
  - 학습 알고리즘
  - Learning from Supervision and Rewards
  - [D1: 지도학습](courses/bci-basics/week7/day1-supervised-learning) | [D2: 강화학습](courses/bci-basics/week7/day2-reinforcement-learning)
* - 8
  - BCI 시스템
  - Brain-Computer Interface Applications
  - [D1: BCI시스템](courses/bci-basics/week8/day1-bci-systems) | [D2: 미래방향](courses/bci-basics/week8/day2-future-directions)
```

---

## 🗺️ 지식 그래프 미리보기

```{mermaid}
flowchart LR
    subgraph 기초
        N[뉴런] --> AP[활동전위]
        AP --> ST[스파이크]
    end
    
    subgraph 인코딩/디코딩
        ST --> TC[튜닝커브]
        TC --> DEC[디코딩]
    end
    
    subgraph 모델링
        AP --> HH[H-H모델]
        HH --> SNN[신경망]
    end
    
    subgraph BCI
        DEC --> BCI[BCI시스템]
        SNN --> BCI
    end
    
    click N "concepts/neuron"
    click AP "concepts/action-potential"
    click TC "concepts/tuning-curve"
    click HH "concepts/hodgkin-huxley"
    click BCI "concepts/bci-decoder"
```

👉 [전체 지식 그래프 보기](concepts/index)

---

## 🎯 학습 목표

이 플랫폼을 통해 다음을 배울 수 있습니다:

1. **신경과학 기초**: 뉴런의 구조와 기능, 신경 신호의 특성
2. **신경 인코딩/디코딩**: 뇌가 정보를 어떻게 표현하고 처리하는지
3. **정보 이론**: 신경 시스템의 정보 처리 원리
4. **뉴런 모델링**: Hodgkin-Huxley 모델과 다양한 뉴런 모델
5. **신경망 네트워크**: 시냅스 모델링과 네트워크 다이나믹스
6. **기계학습**: 지도학습과 강화학습의 신경과학적 기반
7. **BCI 시스템**: 실제 뇌-컴퓨터 인터페이스 구축

---

## 🚀 빠른 시작

```{button-ref} courses/bci-basics/intro
:color: primary
:expand:

BCI 기초 코스 시작하기 →
```

---

## 🛠️ 실습 환경

각 레슨에서 다음 환경으로 코드를 바로 실행할 수 있습니다:

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) **Google Colab**
- [![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/yujinmin2/yujin_research) **GitHub**

---

## 📜 라이선스

- 📄 **콘텐츠**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- 💻 **코드**: [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause)

---

*Last updated: 2025*
