# 🔗 Colab Link 자동 생성 규칙

## URL 패턴

### GitHub Notebook URL → Colab Link 변환 규칙

**입력 (Notebook URL):**
```
https://github.com/{owner}/{repo}/blob/{branch}/notebooks/{filename}.ipynb
```

**출력 (Colab Link):**
```
https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/notebooks/{filename}.ipynb
```

### 변환 규칙
1. `github.com` → `colab.research.google.com/github`
2. 나머지 경로는 그대로 유지
3. 파일 확장자 `.ipynb` 필수

---

## 📋 샘플 검증 (10건)

| # | Notebook URL | Colab Link | 상태 |
|---|-------------|------------|------|
| 1 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W1D1_Introduction.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W1D1_Introduction.ipynb` | ✅ |
| 2 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W1D2_NeuralAnatomy.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W1D2_NeuralAnatomy.ipynb` | ✅ |
| 3 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W2D1_Encoding.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W2D1_Encoding.ipynb` | ✅ |
| 4 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W2D2_SpikeTrains.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W2D2_SpikeTrains.ipynb` | ✅ |
| 5 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W3D1_Decoding.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W3D1_Decoding.ipynb` | ✅ |
| 6 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W3D2_MotorBCI.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W3D2_MotorBCI.ipynb` | ✅ |
| 7 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W4D1_InfoTheory.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W4D1_InfoTheory.ipynb` | ✅ |
| 8 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W4D2_NeuralCoding.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W4D2_NeuralCoding.ipynb` | ✅ |
| 9 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W5D1_EEG.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W5D1_EEG.ipynb` | ✅ |
| 10 | `https://github.com/yujin/yujin_research/blob/main/notebooks/W5D2_ERP.ipynb` | `https://colab.research.google.com/github/yujin/yujin_research/blob/main/notebooks/W5D2_ERP.ipynb` | ✅ |

**검증 결과: 10/10 OK ✅**

---

## 🎯 입력 효율화 팁

### 빠른 입력 방법

1. **Notebook URL만 입력하면 Colab Link 자동 추론 가능**
   - 동기화 스크립트에서 자동 생성
   - Notion에서는 둘 다 입력 권장

2. **URL 패턴 템플릿**
   ```
   Notebook URL: https://github.com/{REPO}/blob/main/notebooks/{WEEK}_{TOPIC}.ipynb
   Colab Link: (위 URL의 github.com → colab.research.google.com/github 치환)
   ```

---

## ⚠️ 주의사항

1. **Branch 일관성**: `main` 브랜치 사용 권장
2. **파일명 규칙**: `W{n}D{m}_{Topic}.ipynb`
3. **Private Repo**: Colab에서 인증 필요
4. **URL 검증**: `https://` 프로토콜 필수

---

# 📊 품질 가드 체크리스트

## ✅ Relation 검증

| 체크 항목 | 결과 |
|----------|------|
| Assets → Lessons 연결 | 🔄 진행 중 |
| Lessons → Courses 연결 | ✅ 완료 (기존 4건) |
| Relation 누락 0건 목표 | 🔄 진행 중 |

## ✅ Select 옵션 표준화

### Lessons DB
| 속성 | 옵션 | 표준화 |
|------|------|--------|
| Status | Draft, Writing, Review, Published | ✅ |
| Type | Lecture, Tutorial, Exercise, Project, Quiz | ✅ |
| Week | Week 0~8 | ✅ |
| Day | Day 1~5 | ✅ |

### Assets DB
| 속성 | 옵션 | 표준화 |
|------|------|--------|
| Type | 🎬 Video, 📓 Notebook, 📊 Dataset, 📄 PDF, 📊 Slides, 💻 Code, 🔗 Link | ✅ |
| License | CC BY 4.0, CC BY-SA 4.0, CC0, MIT, BSD-3, Apache 2.0, Proprietary | ✅ |

### Courses DB
| 속성 | 옵션 | 표준화 |
|------|------|--------|
| Status | Draft, In Progress, Published, Archived | ✅ |
| Level | 입문, 기초, 중급, 고급 | ✅ |
| Language | ko, en, ko/en | ✅ |

## ✅ URL 형식 검증

- [x] 모든 URL `https://` 프로토콜 사용
- [x] Colab Link 10건 샘플 검증 완료
- [x] Notebook URL 형식 일관성

## ✅ 템플릿 입력 시간

- 목표: 신규 레슨 입력 < 5분
- 필수 필드: Title, Slug, Week, Day, Order, Type, Status
- 선택 필드: Learning Objectives, Duration, Notebook URL, Colab Link

## ✅ Rollup 검증

| 코스 | 실제 레슨 수 | Lesson Count Rollup | 일치 |
|------|-------------|---------------------|------|
| BCI 기초 | 10개 | 자동 집계 | ✅ |

---

# 📝 입력 템플릿

## Lessons 템플릿

```
Lesson Title: W{n}D{m}: {Topic Name}
Slug: day{m}-{topic-slug}
Week: Week {n}
Day: Day {m}
Order: {n}.{m}
Type: Tutorial (기본값)
Status: Draft (기본값)
Duration: {분}분
Learning Objectives:
- 목표 1
- 목표 2
- 목표 3
Notebook URL: https://github.com/{repo}/blob/main/notebooks/W{n}D{m}_{Topic}.ipynb
Colab Link: https://colab.research.google.com/github/{repo}/blob/main/notebooks/W{n}D{m}_{Topic}.ipynb
```

## Assets 템플릿 (Notebook)

```
Asset Name: W{n}D{m} {Topic} Notebook
Type: 📓 Notebook
URL: https://github.com/{repo}/blob/main/notebooks/W{n}D{m}_{Topic}.ipynb
License: MIT
Description: {간단한 설명}
Related Lesson: {해당 레슨 선택}
```

## Assets 템플릿 (PDF/Paper)

```
Asset Name: {Paper Title}
Type: 📄 PDF
URL: https://arxiv.org/abs/{id} 또는 DOI URL
License: CC BY 4.0
Description: {저자 및 연도}
Related Lesson: {해당 레슨 선택}
Tags: Reference
```

---

# 🔄 동기화 PoC

## Notion → Markdown 변환 흐름

```
1. Notion API로 Lessons DB 쿼리 (Status = Published)
2. 각 페이지의 속성 추출:
   - Title, Slug, Week, Day, Order
   - Learning Objectives, Duration
   - Notebook URL, Colab Link
3. 페이지 본문 블록 변환:
   - heading → # ##
   - paragraph → text
   - code → ```python
   - callout → :::{admonition}
   - image → figure directive
4. MyST Markdown 파일 생성:
   - Frontmatter (YAML)
   - Colab/Kaggle 버튼 배지
   - 학습 목표 박스
   - 본문 콘텐츠
5. 파일 저장:
   - courses/{course-slug}/{week}/{slug}.md
```

## CSV Export 대안 흐름

```
1. Notion → CSV Export (Lessons DB)
2. Python 스크립트로 CSV 파싱
3. 각 행을 Markdown 템플릿에 매핑
4. Git commit & push
5. GitHub Actions 트리거 → Jupyter Book 빌드
```

---

*Last updated: 2024-11-25*
