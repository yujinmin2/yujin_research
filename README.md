# 🧠 BCI & Computational Neuroscience Research Platform

[![Build Status](https://github.com/yujin/yujin_research/actions/workflows/deploy.yml/badge.svg)](https://github.com/yujin/yujin_research/actions)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Notion Sync](https://img.shields.io/badge/Notion-Sync-black?logo=notion)](https://notion.so)

Brain-Computer Interface와 계산신경과학을 배우는 인터랙티브 연구 플랫폼입니다.

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      콘텐츠 작성 흐름                         │
└─────────────────────────────────────────────────────────────┘

  [Notion]                    [GitHub]                [배포]
     │                           │                       │
     │  ① 콘텐츠 작성             │                       │
     │  (Lessons DB)            │                       │
     ▼                           │                       │
┌─────────┐    자동 동기화     ┌─────────┐   빌드      ┌─────────┐
│ Notion  │ ───────────────► │ GitHub  │ ─────────► │ Vercel/ │
│  Pages  │  (sync_notion.py) │  Repo   │ (MyST CLI) │ Netlify │
└─────────┘                   └─────────┘            └─────────┘
                                  │                       │
                                  │  Jupyter Book         │
                                  │  빌드 프로세스          ▼
                                  │                 ┌──────────┐
                              ┌───┴───┐             │ 정적 웹  │
                              │ .ipynb │             │ 사이트   │
                              │ .md    │             └──────────┘
                              └────────┘
                                  │
                     ┌────────────┼────────────┐
                     ▼            ▼            ▼
               [Colab 버튼] [Kaggle 버튼] [Download]
```

## 📚 코스 구조

| Week | 주제 | 내용 |
|------|------|------|
| 1 | 신경과학 기초 | 뉴런, 활동전위, 시냅스 전달 |
| 2 | 신경 인코딩 | Tuning curves, Rate coding |
| 3 | 신경 디코딩 | Bayesian decoding, Population vectors |
| 4 | 정보 이론 | 엔트로피, 상호정보량 |
| 5 | 뉴런 모델링 | Hodgkin-Huxley, Integrate-and-Fire |
| 6 | 네트워크 모델 | STDP, E-I Balance |
| 7 | 학습 알고리즘 | Backpropagation, RL |
| 8 | BCI 시스템 | 신호 획득, 디코딩, 응용 |

## 🚀 빠른 시작

### 로컬 개발

```bash
# 1. 클론
git clone https://github.com/yujin/yujin_research.git
cd yujin_research

# 2. 의존성 설치
pip install -r requirements.txt
npm install -g mystmd

# 3. 개발 서버 실행
myst start

# 4. 브라우저에서 http://localhost:3000 접속
```

### 빌드

```bash
# HTML 빌드
myst build --html

# 빌드 결과는 _build/html/에 생성됨
```

## ⚙️ 설정

### 1. Notion 연동

1. [Notion Integrations](https://www.notion.so/my-integrations)에서 Integration 생성
2. Lessons DB에 Integration 연결
3. GitHub Secrets 설정:
   - `NOTION_API_KEY`: Integration API 키
   - `NOTION_DATABASE_ID`: `6bde9e09-8279-46ba-9a29-8e3984f973f9`

### 2. GitHub Pages 배포

1. Repository Settings → Pages
2. Source: **GitHub Actions** 선택
3. `main` 브랜치에 푸시하면 자동 배포

### 3. Vercel 배포 (대안)

```bash
# Vercel CLI 설치
npm i -g vercel

# 배포
vercel --prod
```

### 4. Netlify 배포 (대안)

1. [Netlify](https://netlify.com)에 GitHub 리포지토리 연결
2. `netlify.toml` 설정이 자동으로 적용됨

## 📁 프로젝트 구조

```
yujin_research/
├── myst.yml                 # Jupyter Book 설정
├── index.md                 # 홈페이지
├── about.md                 # 소개 페이지
├── courses/
│   └── bci-basics/         # BCI 기초 코스
│       ├── intro.md
│       ├── week1/
│       ├── week2/
│       └── ...
├── notebooks/              # Jupyter 노트북
├── resources/              # 데이터셋, 논문, 도구
├── projects/               # 프로젝트 갤러리
├── scripts/
│   └── sync_notion.py      # Notion 동기화 스크립트
├── .github/workflows/
│   └── deploy.yml          # CI/CD 파이프라인
├── vercel.json             # Vercel 설정
├── netlify.toml            # Netlify 설정
└── requirements.txt        # Python 의존성
```

## 🔄 Notion DB 구조

| DB | 용도 | ID |
|----|------|-----|
| 📚 Courses | 코스 관리 | `31c05592-b009-4418-968f-1d29ff067d7d` |
| 📖 Lessons | 레슨/튜토리얼 | `6bde9e09-8279-46ba-9a29-8e3984f973f9` |
| 📦 Assets | 자료 관리 | `5298c19b-b275-4cc8-a4c5-fd0bc20fdfac` |
| 💼 Projects | 프로젝트 | `ab243a9e-d28d-4ad0-b13c-ac513bddab6c` |

### Lessons DB 스키마

| 속성 | 타입 | 설명 |
|------|------|------|
| Lesson Title | title | 레슨 제목 |
| Slug | text | URL 슬러그 |
| Course | relation | 소속 코스 |
| Week | select | Week 0-8 |
| Day | select | Day 1-5 |
| Order | number | 정렬 순서 |
| Status | select | Draft/Writing/Review/Published |
| Type | select | Lecture/Tutorial/Exercise/Project/Quiz |
| Learning Objectives | text | 학습 목표 |
| Tags | multi_select | Python, Math, etc. |
| Notebook URL | url | GitHub 노트북 링크 |
| Colab Link | url | Colab 링크 |

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

- **콘텐츠**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **코드**: [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause)

## 🙏 감사의 말

- [Coursera Computational Neuroscience](https://www.coursera.org/learn/computational-neuroscience) - University of Washington
- [Neuromatch Academy](https://neuromatch.io/)
- [MyST Markdown](https://mystmd.org/)

---

Made with ❤️ by Yujin
