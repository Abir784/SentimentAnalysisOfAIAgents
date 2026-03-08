from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class MoltBookScraperConfig:
    timeout_seconds: int = 30
    user_agent: str = "Mozilla/5.0 (compatible; SentimentAnalysisBot/1.0)"
    use_reader_fallback: bool = True


class MoltBookScraper:
    def __init__(self, config: MoltBookScraperConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml",
            }
        )

    def scrape_post(self, url: str) -> Dict[str, Any]:
        response = self.session.get(url, timeout=self.config.timeout_seconds)
        response.raise_for_status()
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        now_iso = datetime.now(timezone.utc).isoformat()
        post_id = _extract_post_id(url)
        topic = _extract_topic(url, soup)
        author = _extract_author(soup)
        upvotes = _extract_upvotes(soup)
        comments_count = _extract_comments_count(soup)
        published = _extract_published_at(soup)

        title = _extract_title(soup)
        body = _extract_article_body(soup)
        text = _compose_text(title, body)

        if self.config.use_reader_fallback and _needs_rendered_fallback(text):
            rendered_text = self._fetch_rendered_text(url)
            if rendered_text:
                topic = topic or _extract_topic_from_rendered_text(rendered_text)
                title = _extract_title_from_rendered_text(rendered_text) or title
                body = _extract_body_from_rendered_text(rendered_text) or body
                text = _compose_text(title, body)
                author = author or _extract_author_from_rendered_text(rendered_text)
                upvotes = upvotes or _extract_upvotes_from_rendered_text(rendered_text)
                comments_count = comments_count or _extract_comments_count_from_rendered_text(
                    rendered_text
                )
                comments = _extract_comments_from_rendered_text(rendered_text, post_id)
            else:
                comments = []
        else:
            comments = []

        # Keep key extraction artifacts for auditability and parser debugging.
        payload = {
            "id": post_id,
            "post_id": post_id,
            "thread_id": post_id,
            "parent_id": None,
            "author": author,
            "agent_id": author,
            "topic": topic,
            "text": text,
            "created_at": published,
            "upvotes": upvotes,
            "reply_count": comments_count,
            "comments": comments,
            "url": url,
            "title": title,
            "raw_html_excerpt": html[:4000],
        }

        return {
            "platform": "moltbook",
            "fetched_at": now_iso,
            "source_payload": payload,
            "source_page": 1,
        }

    def _fetch_rendered_text(self, url: str) -> Optional[str]:
        reader_url = _build_reader_url(url)
        response = self.session.get(reader_url, timeout=self.config.timeout_seconds)
        if response.status_code != 200:
            return None
        text = response.text.strip()
        if not text:
            return None
        return text


def _extract_post_id(url: str) -> Optional[str]:
    m = re.search(r"/post/([0-9a-fA-F-]{8,})", url)
    return m.group(1) if m else None


def _extract_topic(url: str, soup: BeautifulSoup) -> Optional[str]:
    m = re.search(r"/m/([A-Za-z0-9_-]+)", url)
    if m:
        return m.group(1)

    text = soup.get_text("\n", strip=True)
    m2 = re.search(r"m/([A-Za-z0-9_-]+)", text)
    if m2:
        return m2.group(1)
    return None


def _extract_author(soup: BeautifulSoup) -> Optional[str]:
    text = soup.get_text("\n", strip=True)
    by_match = re.search(r"Posted by\s+([A-Za-z0-9_-]+)", text)
    if by_match:
        return by_match.group(1)

    u_match = re.search(r"/u/([A-Za-z0-9_-]+)", text)
    if u_match:
        return u_match.group(1)
    return None


def _extract_upvotes(soup: BeautifulSoup) -> Optional[int]:
    text = soup.get_text("\n", strip=True)
    m = re.search(r"▲\s*([0-9][0-9,]*)", text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_comments_count(soup: BeautifulSoup) -> Optional[int]:
    text = soup.get_text("\n", strip=True)
    m = re.search(r"Comments\s*\(([0-9][0-9,]*)\)", text, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None

    m2 = re.search(r"💬\s*([0-9][0-9,]*)", text)
    if m2:
        try:
            return int(m2.group(1).replace(",", ""))
        except ValueError:
            return None

    return None


def _extract_published_at(soup: BeautifulSoup) -> Optional[str]:
    ld = _extract_json_ld_article(soup)
    if ld and isinstance(ld.get("datePublished"), str):
        return _normalize_iso(ld["datePublished"])

    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        return _normalize_iso(str(time_tag.get("datetime")))
    return None


def _extract_title(soup: BeautifulSoup) -> str:
    ld = _extract_json_ld_article(soup)
    if ld and isinstance(ld.get("headline"), str) and ld["headline"].strip():
        return ld["headline"].strip()

    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(" ", strip=True)
        if h1_text:
            return h1_text

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return str(og_title.get("content")).strip()

    return ""


def _extract_article_body(soup: BeautifulSoup) -> str:
    ld = _extract_json_ld_article(soup)
    if ld and isinstance(ld.get("articleBody"), str) and ld["articleBody"].strip():
        return ld["articleBody"].strip()

    article = soup.find("article")
    if article:
        text = article.get_text("\n", strip=True)
        if text:
            return _trim_at_comments_section(text)

    full_text = soup.get_text("\n", strip=True)
    return _extract_body_from_full_text(full_text)


def _extract_json_ld_article(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for script in scripts:
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue

        candidates: List[Dict[str, Any]] = []
        if isinstance(parsed, dict):
            if isinstance(parsed.get("@graph"), list):
                for item in parsed["@graph"]:
                    if isinstance(item, dict):
                        candidates.append(item)
            candidates.append(parsed)
        elif isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    candidates.append(item)

        for c in candidates:
            t = c.get("@type")
            if isinstance(t, list):
                types = {str(x) for x in t}
            else:
                types = {str(t)}
            if {"Article", "NewsArticle", "BlogPosting"}.intersection(types):
                return c
    return None


def _extract_body_from_full_text(full_text: str) -> str:
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("# ") or line.startswith("I measured my cold-start"):
            start_idx = i
            break

    stop_markers = ("## Comments", "## CONTINUE READING", "Additional Links")
    out: List[str] = []
    for line in lines[start_idx:]:
        if any(marker in line for marker in stop_markers):
            break
        out.append(line)
    return "\n".join(out)


def _trim_at_comments_section(text: str) -> str:
    markers = ["## Comments", "Comments (", "## CONTINUE READING"]
    min_idx: Optional[int] = None
    for marker in markers:
        i = text.find(marker)
        if i >= 0 and (min_idx is None or i < min_idx):
            min_idx = i
    if min_idx is not None:
        return text[:min_idx].strip()
    return text.strip()


def _compose_text(title: str, body: str) -> str:
    title = title.strip()
    body = body.strip()
    if title and body:
        return f"{title}\n\n{body}"
    if title:
        return title
    return body


def _needs_rendered_fallback(text: str) -> bool:
    if len(text) < 450:
        return True
    low = text.lower()
    generic_markers = ["loading...", "privacy policy", "owner login", "built for agents"]
    marker_hits = sum(1 for marker in generic_markers if marker in low)
    return marker_hits >= 2


def _build_reader_url(url: str) -> str:
    cleaned = url.replace("https://", "").replace("http://", "")
    return f"https://r.jina.ai/http://{cleaned}"


def _extract_title_from_rendered_text(rendered_text: str) -> Optional[str]:
    m = re.search(r"^Title:\s*(.+)$", rendered_text, flags=re.MULTILINE)
    if m:
        return m.group(1).replace("| moltbook", "").strip()

    md = _markdown_content_section(rendered_text)
    if md:
        first = md.splitlines()[0].strip()
        if first:
            return first.replace("| moltbook", "").strip()
    return None


def _extract_body_from_rendered_text(rendered_text: str) -> Optional[str]:
    md = _markdown_content_section(rendered_text)
    if not md:
        return None

    lines = [ln.rstrip() for ln in md.splitlines()]
    if not lines:
        return None

    # Find the post title row under the "Posted by" section and start after its underline.
    posted_idx = 0
    for i, line in enumerate(lines):
        if "Posted by" in line:
            posted_idx = i
            break

    start_idx: Optional[int] = None
    for i in range(posted_idx, len(lines) - 1):
        underline = lines[i + 1].strip()
        if _is_markdown_underline(underline):
            start_idx = i + 2
            break

    if start_idx is None:
        start_idx = posted_idx + 1

    out: List[str] = []
    stop_markers = (
        "Comments (",
        "## Comments",
        "## CONTINUE READING",
        "## Additional Links",
    )
    for line in lines[start_idx:]:
        stripped = line.strip()
        if any(marker in stripped for marker in stop_markers):
            break
        if not stripped:
            out.append("")
            continue
        if stripped.startswith("[") and "](http" in stripped:
            # Skip navigation-only markdown links that are not post body text.
            continue
        out.append(line)

    body = "\n".join(out).strip()
    return body or None


def _extract_author_from_rendered_text(rendered_text: str) -> Optional[str]:
    m = re.search(
        r"Posted by \[([A-Za-z0-9_-]+)\]\(https?://www\.moltbook\.com/u/",
        rendered_text,
    )
    if m:
        return m.group(1)
    return None


def _extract_upvotes_from_rendered_text(rendered_text: str) -> Optional[int]:
    m = re.search(r"▲\s*([0-9][0-9,]*)", rendered_text)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_comments_count_from_rendered_text(rendered_text: str) -> Optional[int]:
    m = re.search(r"Comments\s*\(([0-9][0-9,]*)\)", rendered_text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"💬\s*([0-9][0-9,]*)\s*comments", rendered_text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_topic_from_rendered_text(rendered_text: str) -> Optional[str]:
    m = re.search(r"\[m/([A-Za-z0-9_-]+)\]\(http", rendered_text)
    if m:
        return m.group(1)
    return None


def _markdown_content_section(rendered_text: str) -> Optional[str]:
    marker = "Markdown Content:"
    i = rendered_text.find(marker)
    if i < 0:
        return None
    return rendered_text[i + len(marker) :].strip()


def _is_markdown_underline(line: str) -> bool:
    if len(line) < 3:
        return False
    chars = set(line)
    return chars == {"="} or chars == {"-"}


def _extract_comments_from_rendered_text(
    rendered_text: str,
    post_id: Optional[str],
) -> List[Dict[str, Any]]:
    md = _markdown_content_section(rendered_text)
    if not md:
        return []

    lines = [ln.rstrip("\n") for ln in md.splitlines()]
    start_idx = None
    for i, line in enumerate(lines):
        if "Comments (" in line:
            start_idx = i
            break
    if start_idx is None:
        return []

    author_re = re.compile(
        r"^\[(?P<author>[^\]]+)\]\(https?://www\.moltbook\.com/u/[^\)]+\)•(?P<time>.+)$"
    )
    score_re = re.compile(r"^▲\s*(?P<score>[0-9][0-9,]*)\s*▼$")

    comments: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    body_lines: List[str] = []
    pending_score: Optional[int] = None
    idx = 0

    def flush_current() -> None:
        nonlocal current, body_lines, idx
        if not current:
            return
        text = "\n".join(body_lines).strip()
        if text:
            idx += 1
            comment_id = f"{post_id or 'post'}-c{idx:05d}"
            comments.append(
                {
                    "comment_id": comment_id,
                    "post_id": post_id,
                    "parent_id": post_id,
                    "level": 0,
                    "author_id": current["author_id"],
                    "relative_time": current["relative_time"],
                    "is_verified": current["is_verified"],
                    "upvotes": current["upvotes"],
                    "text": text,
                }
            )
        current = None
        body_lines = []

    for raw_line in lines[start_idx + 1 :]:
        line = raw_line.strip()

        if (
            line.startswith("## CONTINUE READING")
            or line.startswith("## Additional Links")
            or line.startswith("### ↯Up next")
        ):
            break

        if not line:
            if current is not None and body_lines:
                body_lines.append("")
            continue

        if line in {"⭐ Best🆕 New📜 Old"}:
            continue

        if _is_markdown_underline(line):
            continue

        score_m = score_re.match(line)
        if score_m:
            try:
                pending_score = int(score_m.group("score").replace(",", ""))
            except ValueError:
                pending_score = None
            continue

        author_m = author_re.match(line)
        if author_m:
            flush_current()
            time_raw = author_m.group("time").strip()
            is_verified = "✅ Verified" in time_raw
            time_clean = time_raw.replace("✅ Verified", "").strip()
            current = {
                "author_id": author_m.group("author").strip(),
                "relative_time": time_clean,
                "is_verified": is_verified,
                "upvotes": pending_score,
            }
            pending_score = None
            continue

        if current is not None:
            body_lines.append(raw_line)

    flush_current()
    return comments


def _normalize_iso(value: str) -> str:
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    dt = datetime.fromisoformat(v)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
