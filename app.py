from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import fitz
import numpy as np
from ctransformers import AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


PDF_PATH = Path(os.getenv("PDF_PATH", "shaastra_contextdoc.pdf"))
LLM_MODEL_PATH = Path(
    os.getenv("LLM_MODEL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_CONTEXT_CHARS = 6000
LONG_CONTEXT_NOTICE = (
    "The retrieved context is very long, so only a shortened portion is shown. "
    "If the user asks for a list, answer with only a few relevant examples and "
    "mention that the list is not exhaustive."
)

CATEGORIES = [
    "Events",
    "Workshops",
    "Competitions",
    "Lectures",
    "Venues",
    "Hackathons",
    "Social Initiatives",
]

DETAIL_KEY_PATTERN = r"(Date|Dates|Time|Venue|Description)"
DETAIL_ALIASES = {"Dates": "Date"}
SCHEDULE_QUERY_WORDS = {
    "when",
    "where",
    "date",
    "dates",
    "time",
    "timing",
    "venue",
    "schedule",
    "happening",
}
OUT_OF_SCOPE_QUERY_WORDS = {
    "weather",
    "temperature",
    "rain",
    "raining",
    "forecast",
    "humidity",
    "news",
    "stock",
    "price",
    "traffic",
    "flight",
    "train",
    "cricket",
    "score",
    "today",
    "tomorrow",
}
SHAASTRA_QUERY_WORDS = {
    "shaastra",
    "event",
    "events",
    "workshop",
    "workshops",
    "competition",
    "competitions",
    "lecture",
    "lectures",
    "venue",
    "venues",
    "hackathon",
    "hackathons",
    "robowars",
    "robo",
    "iitm",
    "iit",
}
EVENT_LIST_CONTEXT_PREFIX = "__EVENT_NAME_LIST__"
MONTH_ALIASES = {
    "jan": "january",
    "january": "january",
}


@dataclass
class TreeNode:
    name: str
    level: int
    node_type: str
    value: str = ""
    parent: "TreeNode | None" = None
    children: list["TreeNode"] = field(default_factory=list)

    def add_child(self, child: "TreeNode") -> "TreeNode":
        child.parent = self
        self.children.append(child)
        return child

    @property
    def path(self) -> str:
        parts = []
        node: TreeNode | None = self
        while node:
            parts.append(node.name)
            node = node.parent
        return " > ".join(reversed(parts))

    def text(self, include_children: bool = False) -> str:
        own = f"{self.path}: {self.value}".strip()
        if not include_children:
            return own
        child_text = "\n".join(child.text(include_children=True) for child in self.children)
        return f"{own}\n{child_text}".strip()


def clean_line(line: str) -> str:
    line = (
        line.replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\ufffd", "-")
    )
    line = re.sub(r"\s+", " ", line).strip()
    return line.strip("-:| ")


def extract_pdf_text(pdf_path: Path) -> str:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text"))
    return "\n".join(pages)


def is_category(line: str) -> str | None:
    normalized = clean_line(line).lower()
    for category in CATEGORIES:
        if normalized == category.lower():
            return category
        if re.fullmatch(rf"(?:\d+\.\s*)?{re.escape(category.lower())}", normalized):
            return category
    return None


def split_inline_details(line: str) -> dict[str, str]:
    details: dict[str, str] = {}
    pattern = rf"{DETAIL_KEY_PATTERN}\s*:\s*"
    matches = list(re.finditer(pattern, line, flags=re.IGNORECASE))
    for index, match in enumerate(matches):
        key = normalize_detail_key(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
        value = clean_line(line[start:end])
        if value:
            details[key] = value
    return details


def normalize_detail_key(key: str) -> str:
    key = key.strip().title()
    return DETAIL_ALIASES.get(key, key)


def looks_like_item_title(line: str) -> bool:
    if not looks_like_item(line):
        return False
    words = line.split()
    if len(words) > 9:
        return False
    if line.endswith((".", ",")):
        return False
    if re.match(r"^(a|an|the|this|where|participants|race|build|watch)\b", line, re.I):
        return False
    uppercase_letters = sum(1 for char in line if char.isalpha() and char.isupper())
    letters = sum(1 for char in line if char.isalpha())
    mostly_upper = bool(letters and uppercase_letters / letters > 0.65)
    titleish = all(
        word[:1].isupper() or word.lower() in {"of", "and", "in", "with", "the", "to", "from"}
        for word in words
        if re.search(r"[A-Za-z]", word)
    )
    return mostly_upper or titleish


def looks_like_item(line: str) -> bool:
    if not line or len(line) > 120:
        return False
    if split_inline_details(line):
        return False
    lower = line.lower()
    if lower.startswith(("date:", "dates:", "time:", "venue:", "description:")):
        return False
    if re.match(r"^(page|table of contents)\b", lower):
        return False
    return bool(re.search(r"[A-Za-z]", line))


def add_or_update_detail(item: TreeNode, key: str, value: str) -> None:
    for child in item.children:
        if child.node_type == "detail" and child.name.lower() == key.lower():
            child.value = f"{child.value} {value}".strip()
            return
    item.add_child(TreeNode(name=key, level=3, node_type="detail", value=value))


def get_or_create_item(category: TreeNode, name: str) -> TreeNode:
    for child in category.children:
        if child.node_type == "item" and child.name.lower() == name.lower():
            return child
    return category.add_child(TreeNode(name=name, level=2, node_type="item"))


def guess_category(title: str, description: str, category_nodes: dict[str, TreeNode]) -> TreeNode:
    text = f"{title} {description}".lower()
    if "workshop" in text or title.lower() in {"robotics 101"}:
        return category_nodes["Workshops"]
    if any(word in text for word in ["hackathon", "hack"]):
        return category_nodes["Hackathons"]
    if any(word in text for word in ["challenge", "competition", "quiz", "court", "robowars"]):
        return category_nodes["Competitions"]
    if any(word in text for word in ["lecture", "summit", "conference"]):
        return category_nodes["Lectures"]
    return category_nodes["Events"]


def normalize_lookup_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def known_items(category_nodes: dict[str, TreeNode]) -> list[TreeNode]:
    items = []
    for category in category_nodes.values():
        items.extend(child for child in category.children if child.node_type == "item")
    return items


def find_known_item_in_line(line: str, items: list[TreeNode]) -> TreeNode | None:
    normalized_line = normalize_lookup_text(line)
    if not normalized_line:
        return None

    matches = []
    for item in items:
        normalized_name = normalize_lookup_text(item.name)
        if not normalized_name or len(normalized_name) < 4:
            continue
        if re.search(rf"\b{re.escape(normalized_name)}\b", normalized_line):
            matches.append((len(normalized_name), item))

    if not matches:
        return None
    return max(matches, key=lambda match: match[0])[1]


def extract_time_value(line: str) -> str | None:
    match = re.search(
        r"\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\s*(?:-|to)\s*"
        r"\d{1,2}(?::\d{2})?\s*(?:AM|PM|am|pm)?",
        line,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_line(match.group(0))
    return None


def extract_schedule_venue(line: str, item_name: str, time_value: str | None) -> str:
    venue = re.sub(re.escape(item_name), "", line, flags=re.IGNORECASE)
    if time_value:
        venue = venue.replace(time_value, "")
    venue = re.sub(r"\s{2,}", " ", venue)
    return clean_line(venue)


def extract_explicit_event_date(text: str) -> str | None:
    match = re.search(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+jan(?:uary)?(?:,\s*2026)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_line(match.group(0)).upper()

    match = re.search(r"\bJanuary\s+\d{1,2},\s*2026\b", text, flags=re.IGNORECASE)
    if match:
        return clean_line(match.group(0))

    return None


def enrich_tree_from_detail_blocks(
    text: str, category_nodes: dict[str, TreeNode]
) -> None:
    lines = [clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    for index, line in enumerate(lines):
        date_match = re.match(r"^(Date|Dates)\s*:\s*(.+)$", line, re.IGNORECASE)
        if not date_match:
            continue

        title_index: int | None = None
        for candidate_index in range(index - 1, max(-1, index - 10), -1):
            candidate = lines[candidate_index]
            if candidate.startswith("http"):
                break
            if re.match(rf"^{DETAIL_KEY_PATTERN}\s*:", candidate, re.IGNORECASE):
                break
            if is_category(candidate):
                break
            if looks_like_item_title(candidate):
                title_index = candidate_index
                break

        if title_index is None:
            continue

        title = lines[title_index]
        description_lines = [
            desc_line
            for desc_line in lines[title_index + 1 : index]
            if not re.match(rf"^{DETAIL_KEY_PATTERN}\s*:", desc_line, re.IGNORECASE)
        ]
        description = " ".join(description_lines)
        details = {"Date": clean_line(date_match.group(2))}

        cursor = index + 1
        while cursor < len(lines):
            detail_match = re.match(
                rf"^{DETAIL_KEY_PATTERN}\s*:\s*(.+)$", lines[cursor], re.IGNORECASE
            )
            if not detail_match:
                break
            details[normalize_detail_key(detail_match.group(1))] = clean_line(
                detail_match.group(2)
            )
            cursor += 1

        if "Time" not in details and "Venue" not in details:
            continue

        category = guess_category(title, description, category_nodes)
        item = get_or_create_item(category, title)
        if description:
            add_or_update_detail(item, "Description", description)
        for key, value in details.items():
            add_or_update_detail(item, key, value)


def enrich_tree_from_schedule_rows(
    text: str, category_nodes: dict[str, TreeNode]
) -> None:
    lines = [clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    current_date: str | None = None
    in_schedule_section = False

    for index, line in enumerate(lines):
        date_heading = re.search(r"\b(January\s+\d{1,2},\s*2026)\b", line, re.IGNORECASE)
        if date_heading:
            current_date = date_heading.group(1)
            nearby_heading_text = " ".join(lines[max(0, index - 2) : min(len(lines), index + 8)])
            in_schedule_section = bool(
                re.search(r"\b(day\s*\d+|imc\s+day\s*\d+)\b", nearby_heading_text, re.I)
                and re.search(r"\b(events?|venue|time)\b", nearby_heading_text, re.I)
            )

        if line.lower() in {
            "workshops",
            "elecfest events",
            "envisage",
            "spotlight lectures",
            "coding and logic",
            "competitions",
            "shows",
        }:
            in_schedule_section = False

        if not current_date or not in_schedule_section:
            continue

        item = find_known_item_in_line(line, known_items(category_nodes))
        if not item:
            continue

        search_window = " ".join(lines[index : min(index + 4, len(lines))])
        time_value = extract_time_value(search_window)
        if not time_value:
            continue

        event_date = extract_explicit_event_date(search_window) or current_date

        artist_match = re.search(r"\bby\s+(.+)$", line, flags=re.IGNORECASE)
        if artist_match:
            add_or_update_detail(item, "Artist", clean_line(artist_match.group(1)))

        venue = extract_schedule_venue(line, item.name, time_value)

        if (not venue or venue.lower().startswith("by ")) and index + 1 < len(lines):
            venue = clean_line(lines[index + 1])

        note_parts = [f"Additional schedule listing date: {event_date}"]
        if time_value:
            note_parts.append(f"time {time_value}")
        if venue and not re.fullmatch(r"events|venue|time", venue, flags=re.IGNORECASE):
            note_parts.append(f"venue {venue}")

        add_or_update_detail(item, "Schedule Note", ", ".join(note_parts) + ".")


def enrich_tree_from_spotlight_schedule(
    text: str, category_nodes: dict[str, TreeNode]
) -> None:
    lines = [clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]

    schedule_start = None
    for index, line in enumerate(lines):
        if line.lower() == "spotlight lecture schedule":
            schedule_start = index + 1
            break

    if schedule_start is None:
        return

    speakers = []
    index = schedule_start
    while index < len(lines):
        line = lines[index]
        if line.upper().startswith("HEAR4YOU") or is_category(line):
            break

        if (
            index + 3 < len(lines)
            and re.match(r"^(Dr\.|Ms\.|Mr\.|Prof\.)\s+", line, re.IGNORECASE)
            and lines[index + 1].lower().startswith("venue:")
            and lines[index + 2].lower().startswith("date:")
            and lines[index + 3].lower().startswith("time:")
        ):
            speakers.append(
                {
                    "name": line,
                    "venue": clean_line(lines[index + 1].split(":", 1)[1]),
                    "date": clean_line(lines[index + 2].split(":", 1)[1]),
                    "time": clean_line(lines[index + 3].split(":", 1)[1]),
                }
            )
            index += 4
            continue

        index += 1

    if not speakers:
        return

    item = get_or_create_item(category_nodes["Lectures"], "Spotlight Lecture Schedule")
    add_or_update_detail(
        item,
        "Speakers",
        "\n".join(
            f"{speaker['name']} - {speaker['date']}, {speaker['time']}, {speaker['venue']}"
            for speaker in speakers
        ),
    )


def parse_tree(text: str) -> TreeNode:
    root = TreeNode(name="Shaastra 2026", level=0, node_type="root")
    category_nodes = {
        category: root.add_child(TreeNode(name=category, level=1, node_type="category"))
        for category in CATEGORIES
    }

    current_category: TreeNode | None = None
    current_item: TreeNode | None = None
    pending_detail_key: str | None = None
    overview_lines: list[str] = []

    for raw_line in text.splitlines():
        line = clean_line(raw_line)
        if not line:
            continue

        category = is_category(line)
        if category:
            current_category = category_nodes[category]
            current_item = None
            pending_detail_key = None
            continue

        if current_category is None:
            overview_lines.append(line)
            continue

        inline_details = split_inline_details(line)
        if inline_details and current_item:
            for key, value in inline_details.items():
                add_or_update_detail(current_item, key, value)
            pending_detail_key = None
            continue

        detail_match = re.match(rf"^{DETAIL_KEY_PATTERN}\s*:\s*(.*)$", line, re.IGNORECASE)
        if detail_match and current_item:
            key = normalize_detail_key(detail_match.group(1))
            value = clean_line(detail_match.group(2))
            if value:
                add_or_update_detail(current_item, key, value)
            pending_detail_key = key
            continue

        if pending_detail_key and current_item and not looks_like_item_title(line):
            add_or_update_detail(current_item, pending_detail_key, line)
            continue

        if current_category and looks_like_item_title(line):
            current_item = current_category.add_child(
                TreeNode(name=line, level=2, node_type="item")
            )
            pending_detail_key = None
            continue

        if current_item and looks_like_item(line):
            add_or_update_detail(current_item, "Description", line)
            pending_detail_key = "Description"

    enrich_tree_from_detail_blocks(text, category_nodes)
    enrich_tree_from_schedule_rows(text, category_nodes)
    enrich_tree_from_spotlight_schedule(text, category_nodes)
    root.value = " ".join(overview_lines)
    return root


def iter_nodes(root: TreeNode) -> Iterable[TreeNode]:
    yield root
    for child in root.children:
        yield from iter_nodes(child)


def build_index(root: TreeNode, embedder: SentenceTransformer) -> tuple[list[TreeNode], np.ndarray]:
    nodes = list(iter_nodes(root))
    corpus = [node.text(include_children=node.node_type != "root") for node in nodes]
    embeddings = embedder.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    return nodes, embeddings


def retrieve(
    query: str,
    embedder: SentenceTransformer,
    nodes: list[TreeNode],
    embeddings: np.ndarray,
    top_k: int = 3,
) -> list[TreeNode]:
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    semantic_scores = cosine_similarity(query_embedding, embeddings)[0]
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    lexical_scores = []
    for node in nodes:
        node_text = node.text(include_children=True).lower()
        node_terms = set(re.findall(r"[a-z0-9]+", node_text))
        exact_name_bonus = 0.25 if node.name.lower() in query.lower() else 0.0
        overlap = len(query_terms & node_terms) / max(len(query_terms), 1)
        lexical_scores.append(overlap + exact_name_bonus)
    scores = semantic_scores + (0.20 * np.array(lexical_scores))
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [retrieval_context_node(nodes[index]) for index in top_indices]


def retrieval_context_node(node: TreeNode) -> TreeNode:
    if node.node_type == "detail" and node.parent:
        return node.parent
    return node


def format_retrieved_context(nodes: list[TreeNode]) -> str:
    blocks = []
    seen = set()
    for node in nodes:
        text = node.text(include_children=node.node_type != "root")
        if text not in seen:
            blocks.append(text)
            seen.add(text)
    return "\n\n".join(blocks)


def is_general_shaastra_query(query: str) -> bool:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    general_words = {"what", "about", "define", "explain", "describe", "overview"}
    return "shaastra" in query_terms and bool(query_terms & general_words)


def is_spotlight_speaker_query(query: str) -> bool:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    return "spotlight" in query_terms and bool(
        query_terms & {"speaker", "speakers", "who", "lecture", "lectures"}
    )


def is_comedy_night_query(query: str) -> bool:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    return "comedy" in query_terms and "night" in query_terms


def is_event_name_list_query(query: str) -> bool:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    has_event_word = bool(query_terms & {"event", "events"})
    has_schedule_word = bool(query_terms & SCHEDULE_QUERY_WORDS)
    return has_event_word and (has_schedule_word or extract_query_month_day(query) is not None)


def extract_query_month_day(query: str) -> tuple[str, int] | None:
    month_pattern = r"(jan(?:uary)?)"
    day_pattern = r"(\d{1,2})(?:st|nd|rd|th)?"

    match = re.search(rf"\b{month_pattern}\s+{day_pattern}\b", query, re.IGNORECASE)
    if match:
        return MONTH_ALIASES[match.group(1).lower()], int(match.group(2))

    match = re.search(rf"\b{day_pattern}\s+{month_pattern}\b", query, re.IGNORECASE)
    if match:
        return MONTH_ALIASES[match.group(2).lower()], int(match.group(1))

    return None


def text_mentions_month_day(text: str, month: str, day: int) -> bool:
    month_short = month[:3]
    patterns = [
        rf"\b{re.escape(month)}\s+{day}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
        rf"\b{day}(?:st|nd|rd|th)?\s+{re.escape(month)}\b",
        rf"\b{day}(?:st|nd|rd|th)?\s+{re.escape(month_short)}\b",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def is_schedule_artifact_item(node: TreeNode) -> bool:
    name = node.name.strip()
    lower_name = name.lower()
    if lower_name in {"date", "time", "venue", "finals", "challenge", "streets"}:
        return True
    if re.fullmatch(r"[\d:apm\s-]+", lower_name):
        return True
    if re.search(
        r"\b(rmn|rjn|kcb|bsb|hsb|msb|icsr|sac|oat|clt|hall|lab|library|grounds)\b",
        lower_name,
    ):
        return True
    return False


def matching_items_for_query_date(root: TreeNode, query: str, limit: int = 6) -> list[TreeNode]:
    query_date = extract_query_month_day(query)
    if not query_date:
        return []

    month, day = query_date
    matches = []
    seen = set()
    for node in iter_nodes(root):
        if node.node_type != "item" or not node.parent:
            continue
        if node.parent.name.lower() == "venues":
            continue
        if is_schedule_artifact_item(node):
            continue

        detail_text = " ".join(
            child.value
            for child in node.children
            if child.node_type == "detail" and child.name.lower() in {"date", "schedule note"}
        )
        if not text_mentions_month_day(detail_text, month, day):
            continue

        normalized_name = node.name.lower()
        if normalized_name in seen:
            continue
        seen.add(normalized_name)
        matches.append(node)
        if len(matches) == limit:
            break

    return matches


def format_event_name_list_context(items: list[TreeNode]) -> str:
    lines = [EVENT_LIST_CONTEXT_PREFIX]
    lines.extend(f"- {item.name}" for item in items)
    return "\n".join(lines)


def build_query_context(query: str, root: TreeNode, retrieved_nodes: list[TreeNode]) -> str:
    if is_general_shaastra_query(query) and not is_schedule_query(query):
        return root.text(include_children=False)
    if is_event_name_list_query(query):
        matching_items = matching_items_for_query_date(root, query)
        if matching_items:
            return format_event_name_list_context(matching_items)
    if is_spotlight_speaker_query(query):
        for node in iter_nodes(root):
            if (
                node.name.lower() == "spotlight lecture schedule"
                and any(child.name.lower() == "speakers" for child in node.children)
            ):
                return node.text(include_children=True)
    if is_comedy_night_query(query):
        for node in iter_nodes(root):
            if node.name.lower() == "comedy night":
                return node.text(include_children=True)
    return format_retrieved_context(retrieved_nodes)


def limit_context_for_prompt(context: str) -> str:
    if len(context) <= MAX_CONTEXT_CHARS:
        return context

    shortened = context[:MAX_CONTEXT_CHARS].rsplit("\n", 1)[0].strip()
    return f"{LONG_CONTEXT_NOTICE}\n\n{shortened}"


def concise_long_context_answer(context: str) -> str | None:
    if len(context) <= MAX_CONTEXT_CHARS:
        return None

    items = []
    seen = set()
    for line in context.splitlines():
        match = re.match(r"Shaastra 2026\s*>\s*[^>]+\s*>\s*([^:>]+)\s*:", line)
        if not match:
            continue
        item = clean_line(match.group(1))
        if not item or item.lower() in seen:
            continue
        seen.add(item.lower())
        items.append(item)
        if len(items) == 6:
            break

    if not items:
        return (
            "The relevant context is quite lengthy. Here are a few details from the retrieved "
            "context, but this is not exhaustive:\n"
            f"{limit_context_for_prompt(context)}"
        )

    return (
        "The relevant list is quite long, so here are a few matching items "
        "(not exhaustive):\n"
        + "\n".join(f"- {item}" for item in items)
    )


def event_name_list_answer(query: str, context: str) -> str | None:
    if not is_event_name_list_query(query):
        return None
    if not context.startswith(EVENT_LIST_CONTEXT_PREFIX):
        return None

    names = [
        clean_line(line.removeprefix("-"))
        for line in context.splitlines()[1:]
        if clean_line(line.removeprefix("-"))
    ]
    if not names:
        return None

    query_date = extract_query_month_day(query)
    if query_date:
        month, day = query_date
        heading = f"A few events on {month.title()} {day}:"
    else:
        heading = "A few matching events:"
    return heading + "\n" + "\n".join(f"- {name}" for name in names)


def is_schedule_query(query: str) -> bool:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    return bool(query_terms & SCHEDULE_QUERY_WORDS)


def is_out_of_scope_query(query: str) -> bool:
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    return bool(query_terms & OUT_OF_SCOPE_QUERY_WORDS) and not bool(
        query_terms & SHAASTRA_QUERY_WORDS
    )


def load_llm() -> AutoModelForCausalLM | None:
    if not LLM_MODEL_PATH.exists():
        print(f"Local LLM not found at {LLM_MODEL_PATH}.")
        print("Answers will use retrieved text until you place the GGUF model there.")
        return None

    return AutoModelForCausalLM.from_pretrained(
        str(LLM_MODEL_PATH),
        model_type="mistral",
        context_length=4096,
        max_new_tokens=180,
        temperature=0.2,
    )


def fallback_answer(query: str, context: str) -> str:
    if is_out_of_scope_query(query):
        return (
            "This information is not available in the Shaastra 2026 context document, "
            "so I cannot retrieve it from the provided data."
        )

    if not is_schedule_query(query):
        first_line = context.splitlines()[0] if context else "No relevant context found."
        return re.sub(r"^Shaastra 2026:\s*", "", first_line).strip()

    fields = {"Date": "Not specified", "Time": "Not specified", "Venue": "Not specified"}
    for key in fields:
        match = re.search(rf"\b{key}\s*:\s*([^\n]+)", context, flags=re.IGNORECASE)
        if match:
            fields[key] = match.group(1).strip()
    return "\n".join(f"{key}: {value}" for key, value in fields.items())


def spotlight_speaker_answer(context: str) -> str | None:
    match = re.search(r"\bSpeakers\s*:\s*(.+)", context, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None

    speakers = [
        clean_line(line)
        for line in match.group(1).splitlines()
        if re.match(r"^(Dr\.|Ms\.|Mr\.|Prof\.)\s+", clean_line(line), re.IGNORECASE)
    ]
    if not speakers:
        return None

    return "Spotlight lecture speakers:\n" + "\n".join(f"- {speaker}" for speaker in speakers)


def comedy_night_answer(context: str) -> str | None:
    if "COMEDY NIGHT" not in context.upper():
        return None

    artist_match = re.search(r"\bArtist\s*:\s*([^\n]+)", context, flags=re.IGNORECASE)
    schedule_match = re.search(
        r"Additional schedule listing date:\s*(.*?),\s*time\s*([^,]+),\s*venue\s*([^.]+)",
        context,
        flags=re.IGNORECASE,
    )
    description_match = re.search(r"\bDescription\s*:\s*([^\n]+)", context, flags=re.IGNORECASE)

    if not schedule_match:
        return None

    artist = artist_match.group(1).strip() if artist_match else ""
    date = schedule_match.group(1).strip()
    time = schedule_match.group(2).strip()
    venue = schedule_match.group(3).strip()

    title = f"Comedy Night by {artist}" if artist else "Comedy Night"
    answer = f"{title}\nDate: {date}\nTime: {time}\nVenue: {venue}"
    if description_match:
        answer += f"\n\n{description_match.group(1).strip()}"
    return answer


def generate_answer(
    query: str,
    context: str,
    llm: AutoModelForCausalLM | None,
) -> str:
    if is_out_of_scope_query(query):
        return (
            "This information is not available in the Shaastra 2026 context document, "
            "so I cannot retrieve it from the provided data."
        )

    if is_spotlight_speaker_query(query):
        answer = spotlight_speaker_answer(context)
        if answer:
            return answer

    if is_comedy_night_query(query):
        answer = comedy_night_answer(context)
        if answer:
            return answer

    event_list_answer = event_name_list_answer(query, context)
    if event_list_answer:
        return event_list_answer

    long_answer = concise_long_context_answer(context)
    if long_answer:
        return long_answer

    if llm is None:
        return fallback_answer(query, context)

    if is_schedule_query(query):
        instruction = """Use this exact format when the information is available:
Date: <value>
Time: <value>
Venue: <value>

If a field is missing, write "Not specified".
If the question cannot be answered from the retrieved context, say that the information is not available in the Shaastra 2026 context document.
If the retrieved context says it was shortened because it is very long, provide only a few relevant items and mention that the list is not exhaustive.
Keep any extra explanation short and only add it after those fields if needed."""
    else:
        instruction = """Answer in 1-3 concise sentences.
If the question cannot be answered from the retrieved context, say that the information is not available in the Shaastra 2026 context document.
If the retrieved context says it was shortened because it is very long, provide only a few relevant items and mention that the list is not exhaustive.
Do not force Date, Time, or Venue fields unless the user specifically asks for schedule information.
If multiple matching items are retrieved, describe each item separately with its name on a new line.
If you mention an additional schedule listing, include its schedule listing date from the context.
Do not mention an additional schedule listing if its date is not available in the retrieved context."""

    context = limit_context_for_prompt(context)

    prompt = f"""[INST]
You are a concise RAG assistant for Shaastra 2026.
Answer only from the retrieved context.
{instruction}

Question:
{query}

Retrieved context:
{context}
[/INST]
"""
    return llm(prompt).strip()


def main() -> None:
    print("Loading PDF and building Shaastra 2026 RAG index...")
    text = extract_pdf_text(PDF_PATH)
    root = parse_tree(text)

    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    nodes, embeddings = build_index(root, embedder)

    print("Loading local Mistral model...")
    llm = load_llm()

    print("\nShaastra RAG chatbot is ready. Type 'exit' or 'quit' to stop.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Bot: Goodbye.")
            break
        if not query:
            continue

        retrieved_nodes = retrieve(query, embedder, nodes, embeddings, top_k=3)
        context = build_query_context(query, root, retrieved_nodes)
        answer = generate_answer(query, context, llm)
        print(f"Bot:\n{answer}\n")


if __name__ == "__main__":
    main()
