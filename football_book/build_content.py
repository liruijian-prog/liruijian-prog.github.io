#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from markdown import Markdown


ROOT = Path(__file__).resolve().parents[1]
READER_ROOT = ROOT / "reader"
DEFAULT_OUTPUT = READER_ROOT / "content"

BOOK_TITLE = "《人才漏斗：中国足球的结构基因、制度断层与治理操作系统》"
BOOK_SHORT_TITLE = "人才漏斗"
BOOK_SUBTITLE = "中国青少年足球的结构诊断、国际镜像与治理重建"
BOOK_DESCRIPTION = (
    "一套面向手机阅读的书稿阅读器原型。内容来自当前书稿副本，可在后续修订后重建并自动同步到阅读端。"
)

ANNOTATABLE_TAGS = ("p", "li", "blockquote", "figure", "div", "pre", "h2", "h3", "h4")
REFERENCE_HEADING_PATTERNS = (
    "参考文献",
    "资料来源",
    "数据分级附录",
    "数据来源与参考文献",
    "附录 · 数据分级说明与主要参考文献",
)


@dataclass(frozen=True)
class ChapterSpec:
    id: str
    section_id: str
    section_title: str
    relative_path: str


BOOK_STRUCTURE: list[ChapterSpec] = [
    ChapterSpec("intro-01", "opening", "引言与图示", "00_引言与图示/引言_第1节_本书回答什么问题.md"),
    ChapterSpec("intro-02", "opening", "引言与图示", "00_引言与图示/引言_第2节_全书路线图.md"),
    ChapterSpec("intro-03", "opening", "引言与图示", "00_引言与图示/引言_第3节_本书的理论引擎.md"),
    ChapterSpec("concept-card", "opening", "引言与图示", "00_引言与图示/概念论证文_教练身份谬误.md"),
    ChapterSpec("part1-ch01", "part-1", "第一编 · 病灶诊断", "01_第一编_病灶诊断_第1-6章/第1章_沉默的基座.md"),
    ChapterSpec("part1-ch02", "part-1", "第一编 · 病灶诊断", "01_第一编_病灶诊断_第1-6章/第2章_柠檬市场.md"),
    ChapterSpec("part1-ch03", "part-1", "第一编 · 病灶诊断", "01_第一编_病灶诊断_第1-6章/第3章_看不见的淘汰.md"),
    ChapterSpec("part1-ch04", "part-1", "第一编 · 病灶诊断", "01_第一编_病灶诊断_第1-6章/第4章_教练困局.md"),
    ChapterSpec("part1-ch05", "part-1", "第一编 · 病灶诊断", "01_第一编_病灶诊断_第1-6章/第5章_比赛的碎片.md"),
    ChapterSpec("part1-ch06", "part-1", "第一编 · 病灶诊断", "01_第一编_病灶诊断_第1-6章/第6章_病灶链条.md"),
    ChapterSpec("part2-ch07", "part-2", "第二编 · 国际镜像", "02_第二编_国际镜像_新版_第7-13章/第7章_赛制的力量.md"),
    ChapterSpec("part2-ch08", "part-2", "第二编 · 国际镜像", "02_第二编_国际镜像_新版_第7-13章/第8章_教练是怎么炼成的.md"),
    ChapterSpec("part2-ch09", "part-2", "第二编 · 国际镜像", "02_第二编_国际镜像_新版_第7-13章/第9章_青训是什么.md"),
    ChapterSpec("part2-ch10", "part-2", "第二编 · 国际镜像", "02_第二编_国际镜像_新版_第7-13章/第10章_七条青训DNA.md"),
    ChapterSpec("part2-ch11", "part-2", "第二编 · 国际镜像", "02_第二编_国际镜像_新版_第7-13章/第11章_科技赋能.md"),
    ChapterSpec("part2-ch12", "part-2", "第二编 · 国际镜像", "02_第二编_国际镜像_新版_第7-13章/第12章_文化的根.md"),
    ChapterSpec("part2-ch13", "part-2", "第二编 · 国际镜像", "02_第二编_国际镜像_新版_第7-13章/第13章_镜子反射.md"),
    ChapterSpec("part3-ch14", "part-3", "第三编 · 治理重建", "03_第三编_治理重建_第14-20章/第14章_沉淀产品与可持续性.md"),
    ChapterSpec("part3-ch15", "part-3", "第三编 · 治理重建", "03_第三编_治理重建_第14-20章/第15章_复兴杯：分布式联赛操作系统.md"),
    ChapterSpec("part3-ch16", "part-3", "第三编 · 治理重建", "03_第三编_治理重建_第14-20章/第16章_引入式青训.md"),
    ChapterSpec("part3-ch17", "part-3", "第三编 · 治理重建", "03_第三编_治理重建_第14-20章/第17章_中国答卷.md"),
    ChapterSpec("part3-ch18", "part-3", "第三编 · 治理重建", "03_第三编_治理重建_第14-20章/第18章_文化沉淀.md"),
    ChapterSpec("part3-ch19", "part-3", "第三编 · 治理重建", "03_第三编_治理重建_第14-20章/第19章_六个打通.md"),
    ChapterSpec("part3-ch20", "part-3", "第三编 · 治理重建", "03_第三编_治理重建_第14-20章/第20章_治理范式.md"),
    ChapterSpec("afterword", "closing", "后记", "00_引言与图示/后记_一份来自北体大的答卷.md"),
    ChapterSpec("appendix-a", "appendices", "附录", "04_附录/附录A_完整术语表.md"),
    ChapterSpec("appendix-b", "appendices", "附录", "04_附录/附录B_数据分级与方法论.md"),
    ChapterSpec("appendix-d", "appendices", "附录", "04_附录/附录D_可扩展研究议题清单.md"),
    ChapterSpec("appendix-e", "appendices", "附录", "04_附录/附录E_软信息论文完整推导.md"),
    ChapterSpec("appendix-f", "appendices", "附录", "04_附录/附录F_案例索引.md"),
]


IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
HEADING_1_PATTERN = re.compile(r"^#\s+(.+)$")
HEADING_2_PATTERN = re.compile(r"^##\s+(.+)$")
BLOCK_MATH_PATTERN = re.compile(r"(?P<fence>^\$\$(?:\r?\n)?)(?P<content>.*?)(?:\r?\n)?\$\$\s*$", re.MULTILINE | re.DOTALL)
INLINE_MATH_PATTERN = re.compile(r"(?<!\\)\$(?P<content>[^$\n]+?)(?<!\\)\$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured JSON content for the mobile reader.")
    parser.add_argument("--source", type=Path, default=None, help="Source manuscript directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output content directory.")
    return parser.parse_args()


def detect_default_source() -> Path:
    outputs_dir = ROOT / "outputs"
    candidates = sorted(
        outputs_dir.glob("book_终稿前收口版_*/manuscript"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    fallback = outputs_dir / "book"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("未找到可用书稿目录，请显式传入 --source。")


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "chapters").mkdir(parents=True, exist_ok=True)
    (output_dir / "assets").mkdir(parents=True, exist_ok=True)


def make_markdown_converter() -> Markdown:
    return Markdown(
        extensions=["extra", "tables", "fenced_code", "sane_lists", "toc"],
        output_format="html5",
    )


def chapter_sections() -> list[dict[str, str]]:
    seen: set[str] = set()
    sections: list[dict[str, str]] = []
    for spec in BOOK_STRUCTURE:
        if spec.section_id in seen:
            continue
        sections.append({"id": spec.section_id, "title": spec.section_title})
        seen.add(spec.section_id)
    return sections


def copy_cover_asset(source_root: Path, output_dir: Path) -> str:
    cover_source = source_root / "00_引言与图示" / "中国足球诊断金字塔.png"
    cover_target = output_dir / "assets" / "cover" / cover_source.name
    cover_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cover_source, cover_target)
    return "content/assets/cover/中国足球诊断金字塔.png"


def local_asset_path(src_path: Path, source_root: Path, output_dir: Path) -> str:
    relative = src_path.relative_to(source_root)
    target = output_dir / "assets" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(src_path, target)
    return f"content/assets/{relative.as_posix()}"


def preprocess_markdown(markdown_text: str, source_file: Path, source_root: Path, output_dir: Path) -> str:
    def replace_image(match: re.Match[str]) -> str:
        alt_text, raw_src = match.group(1), match.group(2).strip()
        if raw_src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        src_path = (source_file.parent / raw_src).resolve()
        app_src = local_asset_path(src_path, source_root, output_dir)
        return f"![{alt_text}]({app_src})"

    text = IMAGE_PATTERN.sub(replace_image, markdown_text)
    text = protect_math(text)
    return text


def protect_math(markdown_text: str) -> str:
    def replace_block(match: re.Match[str]) -> str:
        content = match.group("content").strip()
        return f'\n<div class="math-block" data-math="{html_escape(content, quote=True)}"></div>\n'

    def replace_inline(match: re.Match[str]) -> str:
        content = match.group("content").strip()
        return f'<span class="math-inline" data-math="{html_escape(content, quote=True)}"></span>'

    protected = BLOCK_MATH_PATTERN.sub(replace_block, markdown_text)
    protected = INLINE_MATH_PATTERN.sub(replace_inline, protected)
    return protected


def extract_titles(markdown_text: str, fallback: str) -> tuple[str, str]:
    title = fallback
    subtitle = ""
    for line in markdown_text.splitlines():
        match_h1 = HEADING_1_PATTERN.match(line)
        if match_h1:
            title = match_h1.group(1).strip()
            continue
        match_h2 = HEADING_2_PATTERN.match(line)
        if match_h2:
            subtitle = match_h2.group(1).strip()
            break
    return title, subtitle


def chinese_char_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def reading_minutes(text: str) -> int:
    count = max(chinese_char_count(text), 1)
    return max(round(count / 450), 1)


def extract_summary(soup: BeautifulSoup) -> str:
    quote = soup.find("blockquote")
    if quote:
        return quote.get_text(" ", strip=True)
    paragraph = soup.find("p")
    if paragraph:
        return paragraph.get_text(" ", strip=True)
    return ""


def classify_caption_paragraphs(soup: BeautifulSoup) -> None:
    for paragraph in soup.find_all("p"):
        if paragraph.find("em") is None:
            continue
        paragraph_text = paragraph.get_text(" ", strip=True)
        emphasized = paragraph.find("em").get_text(" ", strip=True)
        if paragraph_text != emphasized:
            continue
        previous = paragraph.find_previous_sibling()
        if previous and previous.name == "figure":
            paragraph["class"] = [*paragraph.get("class", []), "figure-caption"]


def clean_structural_markup(soup: BeautifulSoup) -> None:
    first_h1 = soup.find("h1")
    if first_h1:
        first_h1.decompose()
    first_h2 = soup.find("h2")
    if first_h2:
        first_h2.decompose()

    for image in soup.find_all("img"):
        image["loading"] = "lazy"
        image["decoding"] = "async"
        image["class"] = [*image.get("class", []), "reader-image"]
        figure = soup.new_tag("figure", attrs={"class": "figure-block"})
        image.wrap(figure)
        parent = figure.parent
        if parent and parent.name == "p" and len(parent.contents) == 1 and not parent.get_text(" ", strip=True):
            parent.replace_with(figure)

    for table in soup.find_all("table"):
        table["class"] = [*table.get("class", []), "data-table"]
        wrapper = soup.new_tag("div", attrs={"class": "table-wrap"})
        table.wrap(wrapper)

    for anchor in soup.find_all("a"):
        href = anchor.get("href", "")
        if href.startswith(("http://", "https://")):
            anchor["target"] = "_blank"
            anchor["rel"] = "noreferrer"

    classify_caption_paragraphs(soup)


def is_reference_heading(tag: Tag) -> bool:
    if tag.name not in {"h2", "h3", "h4"}:
        return False
    text = tag.get_text(" ", strip=True)
    return any(pattern in text for pattern in REFERENCE_HEADING_PATTERNS)


def extract_reference_cards(soup: BeautifulSoup) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    for heading in list(soup.find_all(is_reference_heading)):
        title = heading.get_text(" ", strip=True)
        level = int(heading.name[1])
        siblings: list[Tag] = [heading]
        current = heading.find_next_sibling()
        while current is not None:
            if current.name in {"h2", "h3", "h4"} and int(current.name[1]) <= level:
                break
            next_node = current.find_next_sibling()
            siblings.append(current)
            current = next_node

        fragment = BeautifulSoup("", "html.parser")
        for node in siblings:
            fragment.append(node.extract())

        cards.append(
            {
                "id": f"card-{len(cards) + 1:02d}",
                "title": title,
                "html": str(fragment),
            }
        )
    return cards


def assign_block_ids(soup: BeautifulSoup) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    counters: dict[str, int] = {}
    heading_stack: list[tuple[int, str]] = []

    def has_annotatable_ancestor(tag: Tag) -> bool:
        parent = tag.parent
        while isinstance(parent, Tag):
            if parent.name in ANNOTATABLE_TAGS and not (
                parent.name == "div" and "table-wrap" not in parent.get("class", [])
            ):
                return True
            parent = parent.parent
        return False

    def current_heading_context() -> str:
        return " / ".join(title for _, title in heading_stack[-2:]) or "root"

    for tag in soup.find_all(ANNOTATABLE_TAGS):
        if tag.name == "div" and "table-wrap" not in tag.get("class", []):
            continue
        if has_annotatable_ancestor(tag):
            continue

        if tag.name in {"h2", "h3", "h4"}:
            level = int(tag.name[1])
            title = tag.get_text(" ", strip=True)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

        block_text = tag.get_text(" ", strip=True)
        if not block_text:
            continue

        normalized = re.sub(r"\s+", " ", block_text)
        signature = "|".join(
            [
                tag.name,
                current_heading_context(),
                normalized[:280],
            ]
        )
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
        counters[digest] = counters.get(digest, 0) + 1
        block_id = f"b-{digest}-{counters[digest]:02d}"

        tag["data-block-id"] = block_id
        tag["data-block-type"] = tag.name
        tag["class"] = [*tag.get("class", []), "annotatable-block"]
        blocks.append(
            {
                "id": block_id,
                "type": tag.name,
                "preview": normalized[:120],
            }
        )
    return blocks


def normalize_html(html: str, *, extract_cards: bool) -> tuple[str, str, int, list[dict[str, str]], list[dict[str, str]], str]:
    soup = BeautifulSoup(html, "html.parser")
    clean_structural_markup(soup)
    reference_cards = extract_reference_cards(soup) if extract_cards else []
    block_index = assign_block_ids(soup)
    summary = extract_summary(soup)
    plain_text = soup.get_text(" ", strip=True)
    image_count = len(soup.find_all("img"))
    return str(soup), summary, image_count, reference_cards, block_index, plain_text


def slugify(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", value.lower()).strip("-")
    return normalized or fallback


def extract_glossary(source_root: Path, output_dir: Path) -> dict[str, object]:
    source_file = source_root / "04_附录" / "附录A_完整术语表.md"
    markdown_text = source_file.read_text(encoding="utf-8")
    converter = make_markdown_converter()
    html = converter.convert(preprocess_markdown(markdown_text, source_file, source_root, output_dir))
    soup = BeautifulSoup(html, "html.parser")

    entries: list[dict[str, object]] = []
    seen_titles: set[str] = set()

    for table in soup.find_all("table"):
        headers = [cell.get_text(" ", strip=True) for cell in table.select("thead th")]
        if headers[:5] != ["概念", "缩写", "英文", "首次出现", "一句话定义"]:
            continue
        for row in table.select("tbody tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            if len(cells) < 5:
                continue
            title, short, english, chapter, definition = cells[:5]
            if title in seen_titles:
                continue
            aliases = [title]
            if short and short != "—":
                aliases.append(short)
            entry_id = slugify(short if short and short != "—" else title, f"term-{len(entries)+1:03d}")
            entries.append(
                {
                    "id": entry_id,
                    "title": title,
                    "short": "" if short == "—" else short,
                    "english": english,
                    "chapter": chapter,
                    "definition": definition,
                    "aliases": aliases,
                }
            )
            seen_titles.add(title)

    payload = {
        "version": datetime.fromtimestamp(source_file.stat().st_mtime).strftime("%Y.%m.%d.%H%M"),
        "updatedAt": datetime.fromtimestamp(source_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "entries": entries,
    }
    (output_dir / "glossary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_content(source_root: Path, output_dir: Path) -> None:
    reset_output_dir(output_dir)
    cover_image_path = copy_cover_asset(source_root, output_dir)
    glossary_payload = extract_glossary(source_root, output_dir)

    sections = chapter_sections()
    chapters_meta: list[dict[str, object]] = []
    search_index: list[dict[str, object]] = []

    latest_timestamp = 0.0
    total_cn_chars = 0

    for order, spec in enumerate(BOOK_STRUCTURE, start=1):
        source_file = source_root / spec.relative_path
        markdown_text = source_file.read_text(encoding="utf-8")
        latest_timestamp = max(latest_timestamp, source_file.stat().st_mtime)

        title, subtitle = extract_titles(markdown_text, source_file.stem)
        processed_markdown = preprocess_markdown(markdown_text, source_file, source_root, output_dir)
        converter = make_markdown_converter()
        html = converter.convert(processed_markdown)
        normalized_html, summary, image_count, reference_cards, block_index, plain_text = normalize_html(
            html,
            extract_cards=spec.section_id != "appendices",
        )

        cn_count = chinese_char_count(plain_text)
        total_cn_chars += cn_count

        chapter_payload = {
            "id": spec.id,
            "order": order,
            "title": title,
            "subtitle": subtitle,
            "sectionId": spec.section_id,
            "sectionTitle": spec.section_title,
            "html": normalized_html,
            "wordCount": cn_count,
            "readingMinutes": reading_minutes(plain_text),
            "summary": summary,
            "imageCount": image_count,
            "referenceCards": reference_cards,
            "blockIndex": block_index,
            "sourcePath": spec.relative_path,
            "updatedAt": datetime.fromtimestamp(source_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        }

        chapter_json_path = output_dir / "chapters" / f"{spec.id}.json"
        chapter_json_path.write_text(json.dumps(chapter_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        chapter_meta = {
            "id": spec.id,
            "order": order,
            "title": title,
            "subtitle": subtitle,
            "sectionId": spec.section_id,
            "sectionTitle": spec.section_title,
            "contentPath": f"content/chapters/{spec.id}.json",
            "summary": summary,
            "wordCount": cn_count,
            "readingMinutes": reading_minutes(plain_text),
            "imageCount": image_count,
            "referenceCardCount": len(reference_cards),
            "updatedAt": chapter_payload["updatedAt"],
        }
        chapters_meta.append(chapter_meta)

        search_index.append(
            {
                "id": spec.id,
                "title": title,
                "subtitle": subtitle,
                "sectionTitle": spec.section_title,
                "plainText": plain_text,
            }
        )

    version = datetime.fromtimestamp(latest_timestamp).strftime("%Y.%m.%d.%H%M")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    manifest = {
        "book": {
            "title": BOOK_TITLE,
            "shortTitle": BOOK_SHORT_TITLE,
            "subtitle": BOOK_SUBTITLE,
            "description": BOOK_DESCRIPTION,
            "version": version,
            "generatedAt": generated_at,
            "sourceRoot": str(source_root.relative_to(ROOT)),
            "coverImagePath": cover_image_path,
            "chapterCount": len(chapters_meta),
            "totalChineseChars": total_cn_chars,
            "glossaryPath": "content/glossary.json",
            "glossaryCount": len(glossary_payload["entries"]),
        },
        "sections": sections,
        "chapters": chapters_meta,
    }

    (output_dir / "book.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "search-index.json").write_text(
        json.dumps({"version": version, "items": search_index}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_root = args.source.resolve() if args.source else detect_default_source().resolve()
    output_dir = args.output.resolve()
    build_content(source_root, output_dir)
    print(f"built reader content from {source_root}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
