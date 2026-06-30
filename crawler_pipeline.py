import csv
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "web"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "web"


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _slugify_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.netloc}{parsed.path}".strip("/")
    slug = re.sub(r"[^0-9a-zA-Z가-힣_-]+", "_", base).strip("_").lower()
    return slug[:90] or "web_page"


def normalize_url(url: str) -> str:
    cleaned = url.strip()
    if not cleaned:
        raise RuntimeError("크롤링할 URL을 입력하세요.")

    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"https://{cleaned}"

    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("URL은 http:// 또는 https:// 형식이어야 합니다.")

    return cleaned


class PageContentParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta_description = ""
        self.text_blocks = []
        self.current_tag = None
        self.current_text = []
        self.skip_depth = 0
        self.in_title = False
        self.title_parts = []
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_cell = []
        self.current_row = []
        self.current_table = []
        self.tables = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
            return

        if self.skip_depth:
            return

        if tag == "title":
            self.in_title = True
        elif tag == "meta" and attrs_dict.get("name", "").lower() == "description":
            self.meta_description = _clean_space(attrs_dict.get("content", ""))
        elif tag in {"h1", "h2", "h3", "p", "li"}:
            self.current_tag = tag
            self.current_text = []
        elif tag == "table":
            self.in_table = True
            self.current_table = []
        elif self.in_table and tag == "tr":
            self.in_row = True
            self.current_row = []
        elif self.in_row and tag in {"th", "td"}:
            self.in_cell = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
            return

        if self.skip_depth:
            return

        if tag == "title":
            self.in_title = False
            self.title = _clean_space(" ".join(self.title_parts))
        elif tag in {"h1", "h2", "h3", "p", "li"} and self.current_tag == tag:
            text = _clean_space(" ".join(self.current_text))
            if len(text) >= 2:
                self.text_blocks.append(text)
            self.current_tag = None
            self.current_text = []
        elif self.in_cell and tag in {"th", "td"}:
            self.current_row.append(_clean_space(" ".join(self.current_cell)))
            self.in_cell = False
            self.current_cell = []
        elif self.in_row and tag == "tr":
            if any(cell for cell in self.current_row):
                self.current_table.append(self.current_row)
            self.in_row = False
            self.current_row = []
        elif self.in_table and tag == "table":
            if self.current_table:
                self.tables.append(self.current_table)
            self.in_table = False
            self.current_table = []

    def handle_data(self, data):
        if self.skip_depth:
            return

        if self.in_title:
            self.title_parts.append(data)
        if self.current_tag:
            self.current_text.append(data)
        if self.in_cell:
            self.current_cell.append(data)


def _table_to_csv(rows):
    width = max((len(row) for row in rows), default=0)
    normalized_rows = [
        [_normalize_table_cell(cell) for cell in row + [""] * (width - len(row))]
        for row in rows
        if any(cell.strip() for cell in row)
    ]

    if not normalized_rows:
        return ""

    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerows(normalized_rows)
    return buffer.getvalue().strip()


def _normalize_table_cell(cell: str) -> str:
    cleaned = _clean_space(cell)
    cleaned = re.sub(r"\[\s*[^\]]+\s*\]", "", cleaned).strip()

    number_candidate = cleaned.replace(",", "").replace("%", "").replace("−", "-")
    if re.fullmatch(r"-?\d+(\.\d+)?", number_candidate):
        return number_candidate

    return cleaned


def _build_text_document(url, parser):
    blocks = []
    if parser.title:
        blocks.append(f"제목: {parser.title}")
    if parser.meta_description:
        blocks.append(f"요약: {parser.meta_description}")
    blocks.append(f"출처 URL: {url}")

    unique_blocks = []
    seen = set()
    for block in parser.text_blocks:
        key = block.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_blocks.append(block)

    blocks.append("본문:")
    blocks.extend(unique_blocks[:250])
    return "\n\n".join(blocks).strip()


def crawl_web_page(url: str) -> dict:
    normalized_url = normalize_url(url)
    slug = _slugify_url(normalized_url)
    raw_dir = RAW_ROOT / slug
    processed_dir = PROCESSED_ROOT / slug
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        normalized_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; PublicDataRAG/1.0; "
                "+https://example.local/public-data-rag)"
            )
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            raw_bytes = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"웹 페이지 요청 실패 ({error.code})") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"웹 페이지 연결 실패: {error.reason}") from error

    encoding_match = re.search(r"charset=([\w-]+)", content_type, re.IGNORECASE)
    encodings = [encoding_match.group(1)] if encoding_match else []
    encodings.extend(["utf-8", "cp949", "euc-kr"])

    html = None
    for encoding in encodings:
        try:
            html = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if html is None:
        html = raw_bytes.decode("utf-8", errors="replace")

    raw_html_path = raw_dir / "page.html"
    raw_html_path.write_text(html, encoding="utf-8")

    parser = PageContentParser()
    parser.feed(html)

    table_csv = ""
    for table in parser.tables:
        if len(table) >= 2:
            table_csv = _table_to_csv(table)
            if table_csv:
                break

    page_title = parser.title or urllib.parse.urlparse(normalized_url).netloc
    safe_title = _clean_space(page_title)[:80] or slug

    if table_csv:
        processed_path = processed_dir / f"{slug}_table.csv"
        processed_path.write_text(table_csv, encoding="utf-8")
        document_title = f"{safe_title}.csv"
        document_content = table_csv
        document_type = "csv_table"
    else:
        document_content = _build_text_document(normalized_url, parser)
        if len(document_content) < 30:
            raise RuntimeError("크롤링한 페이지에서 저장할 본문이나 표를 찾지 못했습니다.")
        processed_path = processed_dir / f"{slug}.txt"
        processed_path.write_text(document_content, encoding="utf-8")
        document_title = f"{safe_title}.txt"
        document_type = "text"

    metadata = {
        "source": "web",
        "url": normalized_url,
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "raw_file": os.path.relpath(raw_html_path, PROJECT_ROOT),
        "processed_file": os.path.relpath(processed_path, PROJECT_ROOT),
        "document_type": document_type,
        "title": page_title,
        "table_count": len(parser.tables),
        "text_block_count": len(parser.text_blocks),
    }
    metadata_path = processed_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "url": normalized_url,
        "document_title": document_title,
        "document_content": document_content,
        "document_type": document_type,
        "processed_path": str(processed_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }
