#!/usr/bin/env python3
"""
Notion → Markdown 동기화 스크립트 (Enhanced)

Lessons DB의 콘텐츠를 Jupyter Book 구조에 맞게 변환합니다.

사용법:
    python scripts/sync_notion.py

환경 변수:
    NOTION_API_KEY: Notion Integration API 키
    NOTION_DATABASE_ID: Lessons DB ID (기본값: 6bde9e09-8279-46ba-9a29-8e3984f973f9)

출력 구조:
    courses/bci-basics/
    ├── week1/
    │   ├── day1-intro-neurobiology.md
    │   └── day2-neural-anatomy.md
    ├── week2/
    │   └── ...
    └── _toc.yml (자동 생성)
"""

import os
import re
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from notion_client import Client
except ImportError:
    print("⚠️ notion-client not installed. Run: pip install notion-client")
    exit(1)

# ============================================================
# 설정
# ============================================================
NOTION_API_KEY = os.environ.get("NOTION_API_KEY")

# Database IDs (업데이트된 스키마 반영)
LESSONS_DB_ID = os.environ.get("NOTION_DATABASE_ID", "6bde9e09-8279-46ba-9a29-8e3984f973f9")
COURSES_DB_ID = os.environ.get("NOTION_COURSES_DB_ID", "31c05592-b009-4418-968f-1d29ff067d7d")
ASSETS_DB_ID = os.environ.get("NOTION_ASSETS_DB_ID", "5298c19b-b275-4cc8-a4c5-fd0bc20fdfac")

# 출력 디렉토리
BASE_OUTPUT_DIR = Path("courses")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "yujin/yujin_research")

# Published 상태만 동기화
SYNC_STATUS = ["Published"]

# 코스 슬러그 매핑 (Notion Course ID → 폴더명)
# 동적으로 Courses DB에서 가져옴
COURSE_SLUGS: Dict[str, str] = {}


# ============================================================
# Notion 클라이언트
# ============================================================
def get_notion_client() -> Client:
    """Notion 클라이언트 초기화"""
    if not NOTION_API_KEY:
        raise ValueError("❌ NOTION_API_KEY environment variable is not set")
    return Client(auth=NOTION_API_KEY)


# ============================================================
# 속성 추출 헬퍼
# ============================================================
def get_title(properties: Dict, key: str = "Lesson Title") -> str:
    """title 속성 추출"""
    prop = properties.get(key, {})
    title_list = prop.get("title", [])
    return "".join([t.get("plain_text", "") for t in title_list])


def get_rich_text(properties: Dict, key: str) -> str:
    """rich_text 속성 추출"""
    prop = properties.get(key, {})
    text_list = prop.get("rich_text", [])
    return "".join([t.get("plain_text", "") for t in text_list])


def get_select(properties: Dict, key: str) -> Optional[str]:
    """select 속성 추출"""
    prop = properties.get(key, {})
    select = prop.get("select")
    return select.get("name") if select else None


def get_multi_select(properties: Dict, key: str) -> List[str]:
    """multi_select 속성 추출"""
    prop = properties.get(key, {})
    return [item.get("name", "") for item in prop.get("multi_select", [])]


def get_number(properties: Dict, key: str) -> Optional[float]:
    """number 속성 추출"""
    prop = properties.get(key, {})
    return prop.get("number")


def get_url(properties: Dict, key: str) -> Optional[str]:
    """url 속성 추출"""
    prop = properties.get(key, {})
    return prop.get("url")


def get_relation(properties: Dict, key: str) -> List[str]:
    """relation 속성에서 페이지 ID 목록 추출"""
    prop = properties.get(key, {})
    return [item.get("id", "") for item in prop.get("relation", [])]


# ============================================================
# 코스 정보 로드
# ============================================================
def load_course_slugs(client: Client) -> Dict[str, Dict]:
    """Courses DB에서 코스 정보 로드 (ID → slug, name 매핑)"""
    global COURSE_SLUGS
    
    if not COURSES_DB_ID:
        print("⚠️ COURSES_DB_ID not set, using default slug")
        return {}
    
    print(f"📚 Loading courses from: {COURSES_DB_ID}")
    
    try:
        results = client.databases.query(database_id=COURSES_DB_ID)
    except AttributeError:
        # Fallback for older API
        print("⚠️ Using fallback query method")
        results = {"results": []}
    except Exception as e:
        print(f"⚠️ Could not load courses: {e}")
        results = {"results": []}
    
    for page in results.get("results", []):
        page_id = page["id"]
        props = page.get("properties", {})
        
        course_name = get_title(props, "Course Name")
        slug = get_rich_text(props, "Slug") or slugify(course_name)
        
        COURSE_SLUGS[page_id] = {
            "name": course_name,
            "slug": slug
        }
        print(f"  📖 {course_name} → {slug}")
    
    return COURSE_SLUGS


def get_course_slug(client: Client, course_page_ids: List[str]) -> str:
    """Course relation에서 슬러그 추출"""
    if not course_page_ids:
        return "uncategorized"
    
    course_id = course_page_ids[0]  # 첫 번째 코스 사용
    
    # 캐시된 정보 확인
    if course_id in COURSE_SLUGS:
        return COURSE_SLUGS[course_id]["slug"]
    
    # 캐시에 없으면 API 호출
    try:
        page = client.pages.retrieve(page_id=course_id)
        props = page.get("properties", {})
        slug = get_rich_text(props, "Slug")
        
        if slug:
            COURSE_SLUGS[course_id] = {"slug": slug}
            return slug
    except Exception as e:
        print(f"  ⚠️ Could not fetch course info: {e}")
    
    return "bci-basics"  # 기본값


# ============================================================
# Rich Text → Markdown 변환
# ============================================================
def rich_text_to_markdown(rich_text_list: List[Dict]) -> str:
    """Notion rich_text를 Markdown으로 변환 (서식 포함)"""
    result = []
    
    for rt in rich_text_list:
        text = rt.get("plain_text", "")
        annotations = rt.get("annotations", {})
        href = rt.get("href")
        
        # 서식 적용
        if annotations.get("code"):
            text = f"`{text}`"
        if annotations.get("bold"):
            text = f"**{text}**"
        if annotations.get("italic"):
            text = f"*{text}*"
        if annotations.get("strikethrough"):
            text = f"~~{text}~~"
        if annotations.get("underline"):
            text = f"<u>{text}</u>"
        
        # 링크
        if href:
            text = f"[{text}]({href})"
        
        result.append(text)
    
    return "".join(result)


# ============================================================
# 블록 → Markdown 변환
# ============================================================
def block_to_markdown(block: Dict, indent: int = 0) -> str:
    """Notion 블록을 Markdown으로 변환"""
    block_type = block.get("type")
    indent_str = "  " * indent
    
    if block_type == "paragraph":
        text = rich_text_to_markdown(block["paragraph"]["rich_text"])
        return f"{indent_str}{text}\n\n" if text else "\n"
    
    elif block_type == "heading_1":
        text = rich_text_to_markdown(block["heading_1"]["rich_text"])
        return f"## {text}\n\n"  # H1은 페이지 제목용이므로 H2로 변환
    
    elif block_type == "heading_2":
        text = rich_text_to_markdown(block["heading_2"]["rich_text"])
        return f"### {text}\n\n"
    
    elif block_type == "heading_3":
        text = rich_text_to_markdown(block["heading_3"]["rich_text"])
        return f"#### {text}\n\n"
    
    elif block_type == "bulleted_list_item":
        text = rich_text_to_markdown(block["bulleted_list_item"]["rich_text"])
        return f"{indent_str}- {text}\n"
    
    elif block_type == "numbered_list_item":
        text = rich_text_to_markdown(block["numbered_list_item"]["rich_text"])
        return f"{indent_str}1. {text}\n"
    
    elif block_type == "to_do":
        text = rich_text_to_markdown(block["to_do"]["rich_text"])
        checked = "x" if block["to_do"].get("checked") else " "
        return f"{indent_str}- [{checked}] {text}\n"
    
    elif block_type == "code":
        text = rich_text_to_markdown(block["code"]["rich_text"])
        language = block["code"].get("language", "python")
        
        # 실행 가능한 코드 셀로 변환
        if language == "python":
            return f"```{{code-cell}} python\n{text}\n```\n\n"
        else:
            return f"```{language}\n{text}\n```\n\n"
    
    elif block_type == "quote":
        text = rich_text_to_markdown(block["quote"]["rich_text"])
        lines = text.split("\n")
        quoted = "\n".join([f"> {line}" for line in lines])
        return f"{quoted}\n\n"
    
    elif block_type == "divider":
        return "---\n\n"
    
    elif block_type == "callout":
        text = rich_text_to_markdown(block["callout"]["rich_text"])
        emoji = block["callout"].get("icon", {}).get("emoji", "💡")
        
        # MyST admonition으로 변환
        admonition_type = {
            "💡": "tip",
            "📝": "note", 
            "⚠️": "warning",
            "❗": "important",
            "🔥": "danger",
            "❓": "question",
            "✅": "success",
        }.get(emoji, "note")
        
        return f"```{{admonition}} {emoji}\n:class: {admonition_type}\n{text}\n```\n\n"
    
    elif block_type == "image":
        image_data = block["image"]
        url = ""
        if image_data.get("type") == "file":
            url = image_data.get("file", {}).get("url", "")
        elif image_data.get("type") == "external":
            url = image_data.get("external", {}).get("url", "")
        
        caption = rich_text_to_markdown(image_data.get("caption", []))
        alt = caption or "image"
        
        return f"```{{figure}} {url}\n:alt: {alt}\n:align: center\n\n{caption}\n```\n\n"
    
    elif block_type == "video":
        video_data = block["video"]
        url = ""
        if video_data.get("type") == "external":
            url = video_data.get("external", {}).get("url", "")
        
        if "youtube.com" in url or "youtu.be" in url:
            video_id = extract_youtube_id(url)
            return f"```{{youtube}} {video_id}\n:width: 100%\n:align: center\n```\n\n"
        
        return f"[🎬 Video]({url})\n\n"
    
    elif block_type == "toggle":
        text = rich_text_to_markdown(block["toggle"]["rich_text"])
        return f"```{{dropdown}} {text}\n:animate: fade-in-slide-down\n\n*내용을 펼쳐보세요*\n```\n\n"
    
    elif block_type == "equation":
        expression = block["equation"].get("expression", "")
        return f"$$\n{expression}\n$$\n\n"
    
    elif block_type == "bookmark":
        url = block["bookmark"].get("url", "")
        caption = rich_text_to_markdown(block["bookmark"].get("caption", []))
        return f"[{caption or url}]({url})\n\n"
    
    elif block_type == "embed":
        url = block["embed"].get("url", "")
        return f"<iframe src=\"{url}\" width=\"100%\" height=\"400\"></iframe>\n\n"
    
    else:
        return ""


def extract_youtube_id(url: str) -> str:
    """YouTube URL에서 video ID 추출"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return url


# ============================================================
# 페이지 동기화
# ============================================================
def get_page_blocks(client: Client, page_id: str) -> List[Dict]:
    """페이지의 모든 블록 재귀적으로 가져오기"""
    blocks = []
    
    response = client.blocks.children.list(block_id=page_id)
    for block in response.get("results", []):
        blocks.append(block)
        
        # 자식 블록이 있으면 재귀적으로 가져오기
        if block.get("has_children"):
            children = get_page_blocks(client, block["id"])
            block["children"] = children
    
    return blocks


def generate_frontmatter(page: Dict, properties: Dict) -> str:
    """MyST frontmatter 생성"""
    title = get_title(properties)
    week = get_select(properties, "Week")
    day = get_select(properties, "Day")
    tags = get_multi_select(properties, "Tags")
    lesson_type = get_select(properties, "Type")
    
    frontmatter = {
        "title": title,
        "subtitle": f"{week} - {day}" if week and day else None,
        "subject": "BCI & Computational Neuroscience",
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    
    # kernelspec 추가 (Python 코드가 있는 경우)
    if "Python" in tags or lesson_type == "Tutorial":
        frontmatter["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3"
        }
    
    # None 값 제거
    frontmatter = {k: v for k, v in frontmatter.items() if v is not None}
    
    return yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False)


def generate_header_badges(properties: Dict) -> str:
    """Colab/Kaggle 버튼 배지 생성"""
    colab_link = get_url(properties, "Colab Link")
    notebook_url = get_url(properties, "Notebook URL")
    
    badges = []
    
    if colab_link:
        badges.append(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link})")
    elif notebook_url:
        # Colab 링크 자동 생성
        if "github.com" in notebook_url:
            colab_url = notebook_url.replace("github.com", "colab.research.google.com/github")
            badges.append(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})")
    
    if notebook_url and "github.com" in notebook_url:
        # Kaggle 링크 생성
        kaggle_url = notebook_url.replace("github.com", "kaggle.com/kernels/welcome?src=")
        badges.append(f"[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]({kaggle_url})")
    
    if badges:
        return " ".join(badges) + "\n\n"
    return ""


def generate_learning_box(properties: Dict) -> str:
    """학습 목표 박스 생성"""
    objectives = get_rich_text(properties, "Learning Objectives")
    duration = get_rich_text(properties, "Duration")
    prerequisites = get_rich_text(properties, "Prerequisites")
    
    if not objectives:
        return ""
    
    content = "```{admonition} 🎯 학습 목표\n:class: tip\n\n"
    
    if duration:
        content += f"**⏱️ 예상 시간:** {duration}\n\n"
    
    if prerequisites:
        content += f"**📚 선수 지식:** {prerequisites}\n\n"
    
    content += "**학습 후 할 수 있는 것:**\n"
    
    # 학습 목표를 리스트로 변환
    for line in objectives.split("\n"):
        line = line.strip()
        if line:
            if not line.startswith("-") and not line.startswith("•"):
                line = f"- {line}"
            content += f"{line}\n"
    
    content += "```\n\n"
    return content


def slugify(text: str) -> str:
    """텍스트를 URL-safe 슬러그로 변환"""
    slug = re.sub(r'[^\w\s가-힣-]', '', text)
    slug = re.sub(r'\s+', '-', slug)
    return slug.lower()


def sync_lesson(client: Client, page: Dict) -> Optional[Path]:
    """개별 레슨 페이지 동기화"""
    page_id = page["id"]
    properties = page.get("properties", {})
    
    # 속성 추출
    title = get_title(properties)
    slug = get_rich_text(properties, "Slug") or slugify(title)
    week = get_select(properties, "Week")
    status = get_select(properties, "Status")
    
    # Course relation에서 코스 슬러그 가져오기
    course_ids = get_relation(properties, "Course")
    course_slug = get_course_slug(client, course_ids)
    
    # 상태 확인
    if status not in SYNC_STATUS:
        print(f"  ⏭️ Skipping (status: {status}): {title}")
        return None
    
    print(f"  📝 Syncing: {title} → {course_slug}")
    
    # 출력 경로 결정 (courses/{course_slug}/{week}/{slug}.md)
    output_dir = BASE_OUTPUT_DIR / course_slug
    
    if week:
        week_num = week.lower().replace(" ", "")  # "Week 1" → "week1"
        output_dir = output_dir / week_num
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slug}.md"
    
    # 블록 콘텐츠 가져오기
    blocks = get_page_blocks(client, page_id)
    
    # Markdown 생성
    md_content = "---\n"
    md_content += generate_frontmatter(page, properties)
    md_content += "---\n\n"
    
    # 제목
    md_content += f"# {title}\n\n"
    
    # 배지
    md_content += generate_header_badges(properties)
    
    # 학습 목표 박스
    md_content += generate_learning_box(properties)
    
    # 본문 콘텐츠
    for block in blocks:
        md_content += block_to_markdown(block)
        
        # 자식 블록 처리
        if "children" in block:
            for child in block["children"]:
                md_content += block_to_markdown(child, indent=1)
    
    # 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return output_path


# ============================================================
# TOC 자동 생성
# ============================================================
def generate_toc(lessons: List[Dict]) -> None:
    """myst.yml의 TOC 업데이트를 위한 정보 출력"""
    grouped = {}
    
    for lesson in lessons:
        props = lesson.get("properties", {})
        week = get_select(props, "Week") or "Week 0"
        slug = get_rich_text(props, "Slug") or slugify(get_title(props))
        title = get_title(props)
        order = get_number(props, "Order") or 0
        
        if week not in grouped:
            grouped[week] = []
        
        grouped[week].append({
            "slug": slug,
            "title": title,
            "order": order
        })
    
    # TOC 출력
    print("\n📑 TOC Structure:")
    print("-" * 40)
    
    for week in sorted(grouped.keys()):
        week_slug = week.lower().replace(" ", "")
        print(f"  - title: {week}")
        print(f"    children:")
        
        lessons_in_week = sorted(grouped[week], key=lambda x: x["order"])
        for lesson in lessons_in_week:
            print(f"      - file: courses/bci-basics/{week_slug}/{lesson['slug']}")


# ============================================================
# 메인 동기화
# ============================================================
def sync_database(client: Client, database_id: str) -> None:
    """데이터베이스의 모든 Published 페이지 동기화"""
    
    # 먼저 코스 정보 로드
    load_course_slugs(client)
    
    print(f"\n📥 Querying lessons database: {database_id}")
    
    # Published 상태만 필터링
    filter_params = {
        "or": [{"property": "Status", "select": {"equals": status}} for status in SYNC_STATUS]
    }
    
    # Order 기준 정렬
    sorts = [
        {"property": "Order", "direction": "ascending"}
    ]
    
    try:
        results = client.databases.query(
            database_id=database_id,
            filter=filter_params,
            sorts=sorts
        )
    except Exception as e:
        print(f"❌ Database query failed: {e}")
        results = {"results": []}
    
    pages = results.get("results", [])
    print(f"📄 Found {len(pages)} published lessons")
    
    synced = []
    for page in pages:
        try:
            path = sync_lesson(client, page)
            if path:
                synced.append(page)
        except Exception as e:
            title = get_title(page.get("properties", {}))
            print(f"  ❌ Error syncing '{title}': {e}")
    
    print(f"\n✅ Successfully synced {len(synced)} lessons")
    
    # TOC 정보 출력
    if synced:
        generate_toc(synced)


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("🚀 Notion → Jupyter Book Sync")
    print(f"📅 {datetime.now().isoformat()}")
    print("=" * 60)
    
    if not LESSONS_DB_ID:
        print("⚠️ NOTION_DATABASE_ID not set. Skipping sync.")
        return
    
    try:
        client = get_notion_client()
        sync_database(client, LESSONS_DB_ID)
        print("\n🎉 Sync completed successfully!")
    except Exception as e:
        print(f"\n❌ Sync failed: {e}")
        raise


if __name__ == "__main__":
    main()
