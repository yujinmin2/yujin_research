---
title: "스파이크 트레인 (Spike Train)"
---

# 📊 스파이크 트레인 (Spike Train)

> 뉴런 발화의 시간적 시퀀스

---

## 📖 정의

**스파이크 트레인**은 뉴런이 발화한 시점들의 시퀀스입니다.

```
시간:  0ms    100ms   200ms   300ms   400ms   500ms
       |       |       |       |       |       |
스파이크: |  |    ||     |         ||  |     |
```

---

## 📐 수학적 표현

**포인트 프로세스**:

$$\rho(t) = \sum_{i} \delta(t - t_i)$$

여기서 $t_i$는 i번째 스파이크 시간

---

## 🔬 분석 방법

| 분석 | 설명 |
|------|------|
| **발화율** | spikes/second (Hz) |
| **ISI** | 스파이크 간 간격 |
| **CV** | 변동계수 (규칙성) |
| **PSTH** | 자극 정렬 히스토그램 |
| **래스터 플롯** | 여러 trial 시각화 |

---

## 🔗 관련 개념

- [활동전위](action-potential)
- [Rate Coding](rate-coding)
- [ISI 분석](psth)
- [PSTH](psth)
- [튜닝 커브](tuning-curve)

---

## 📚 관련 수업

- [W2D2: Spike Trains & Neural Code](../courses/bci-basics/week2/day2-spike-trains)
