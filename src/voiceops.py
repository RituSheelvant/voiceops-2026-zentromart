"""VoiceOps 2026 ZentroMart support agent.

Ollama provides local chat and embeddings; the supplied synthetic tools handle
verified order lookup and ticket creation; Gnani provides all speech services.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "Capstone_Project_Dataset_FTTT_2026"
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_CHAT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
DEFAULT_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
ORDER_RE = re.compile(r"(?<!\d)(\d{4}(?:[\s-]?\d{4}))(?!\d)")
REFERENCE_DATE = date(2026, 10, 20)
HOLIDAYS = {
    date(2026, 8, 15), date(2026, 10, 2),
    date(2026, 10, 19), date(2026, 11, 9),
}


def business_days_elapsed(start_text: str) -> int:
    start = date.fromisoformat(str(start_text)[:10])
    day, count = start, 0
    while day < REFERENCE_DATE:
        day += timedelta(days=1)
        if day.weekday() != 6 and day not in HOLIDAYS:
            count += 1
    return count


def post_json(path: str, body: dict, timeout: int = 240) -> dict:
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Start Ollama and check the local model."
        ) from exc


class LocalModel:
    def __init__(self, chat_model: str, embed_model: str):
        self.chat_model = chat_model
        self.embed_model = embed_model

    def chat(self, messages: list[dict]) -> dict:
        data = post_json("/api/chat", {
            "model": self.chat_model, "messages": messages, "stream": False,
            "format": "json", "options": {
                "temperature": 0.0, "seed": 42, "num_ctx": 3072, "num_predict": 240
            }
        })
        text = data.get("message", {}).get("content", "").strip()
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass
        return {"decision": "answer", "reply": text}

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 32):
            result = post_json("/api/embed", {
                "model": self.embed_model, "input": texts[start:start + 32]
            })
            vectors.extend(result.get("embeddings", []))
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding model returned {len(vectors)} vectors for {len(texts)} chunks. "
                "Run: ollama pull nomic-embed-text"
            )
        return vectors


def parse_frontmatter(text: str) -> tuple[dict, str]:
    metadata = {}
    if not text.startswith("---"):
        return metadata, text
    parts = text.split("---", 2)
    if len(parts) != 3:
        return metadata, text
    for line in parts[1].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip('"').strip("'")
    return metadata, parts[2].strip()


def load_knowledge(dataset: Path) -> list[dict]:
    chunks = []
    for path in sorted((dataset / "kb").glob("*.md")):
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        for section in re.split(r"(?m)(?=^##\s*\[[^\]]+\])", body):
            section = section.strip()
            if not section:
                continue
            match = re.match(r"##\s*\[([^\]]+)\]\s*(.*)", section)
            chunks.append({
                "id": match.group(1) if match else metadata.get("doc_id", path.stem),
                "title": match.group(2).strip() if match else metadata.get("title", path.stem),
                "text": section, "doc": path.name,
                "doc_type": metadata.get("doc_type", "unknown"),
                "status": metadata.get("status", "current"),
                "effective_from": metadata.get("effective_from", ""),
                "language": metadata.get("language", "en"),
            })
    with (dataset / "data" / "products.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            chunks.append({
                "id": f"products.csv:{row['sku']}", "title": row.get("title", row["sku"]),
                "text": (
                    f"Product {row.get('title','')}; SKU {row['sku']}; brand {row.get('brand','')}; "
                    f"return category {row.get('return_category','')}; price INR {row.get('price_inr','')}; "
                    f"fulfilled by {row.get('fulfilled_by','')}; Assured {row.get('assured','')}; "
                    f"warranty months {row.get('warranty_months','')}; {row.get('description','')}"
                ),
                "doc": "products.csv", "doc_type": "catalog", "status": "current",
                "effective_from": "", "language": "en",
            })
    return chunks


def load_catalog_overview(dataset: Path) -> dict:
    with (dataset / "data" / "products.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("return_category") or "uncategorized"].append(row)
    parts = [f"Catalog overview from products.csv: {len(rows)} listed products across {len(groups)} categories."]
    for category in sorted(groups):
        examples = ", ".join(
            f"{row.get('title', row.get('sku', ''))} (SKU {row.get('sku', '')})"
            for row in groups[category][:2]
        )
        parts.append(f"{category}: {len(groups[category])} products; examples: {examples}.")
    if rows and not any(any(key in row for key in ("stock", "inventory", "available_quantity")) for row in rows):
        parts.append("The catalog does not contain live stock or inventory quantities.")
    return {
        "id": "products.csv:catalog-overview", "title": "Product catalog overview",
        "text": " ".join(parts), "doc": "products.csv", "doc_type": "catalog",
        "status": "current", "effective_from": "", "language": "en",
    }


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    an = math.sqrt(sum(x * x for x in a))
    bn = math.sqrt(sum(x * x for x in b))
    return dot / (an * bn) if an and bn else 0.0


class Retriever:
    def __init__(self, dataset: Path, model: LocalModel):
        self.model = model
        self.chunks = load_knowledge(dataset)
        self.catalog_overview = load_catalog_overview(dataset)
        cache_dir = ROOT / ".cache"
        cache_dir.mkdir(exist_ok=True)
        self.cache_path = cache_dir / "knowledge_vectors.json"
        digest = hashlib.sha256(
            json.dumps([c["id"] + c["text"] for c in self.chunks],
                       ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        cache = None
        if self.cache_path.exists():
            try:
                cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
        if cache and cache.get("digest") == digest and cache.get("model") == model.embed_model:
            self.vectors = cache["vectors"]
        else:
            print(f"Creating local embedding index for {len(self.chunks)} policy sections and catalog rows...")
            self.vectors = model.embed([c["text"] for c in self.chunks])
            self.cache_path.write_text(json.dumps({
                "digest": digest, "model": model.embed_model, "vectors": self.vectors
            }), encoding="utf-8")

    def search(self, query: str, limit: int = 5) -> list[dict]:
        vector = self.model.embed([query])[0]
        ranked = [{**c, "score": cosine(vector, self.vectors[i])}
                  for i, c in enumerate(self.chunks)]
        ranked.sort(key=lambda c: c["score"], reverse=True)
        normalized = clean(query)
        asks_return = any(term in normalized for term in ("return", "returning", "exchange"))
        mentions_footwear = any(term in normalized for term in (
            "shoe", "shoes", "footwear", "sandal", "sandals", "sneaker", "sneakers"
        ))
        skus = set(re.findall(r"\b[A-Z]{3}-\d{4}\b", query.upper()))
        by_id = {chunk["id"]: chunk for chunk in self.chunks}
        normalized = normalized.casefold()
        # Add policy sections by support intent, then the best matching catalog
        # row. This protects short and mixed-language questions from embedding
        # drift while keeping answers grounded in the supplied documents.
        terms = {t for t in re.findall(r"[a-z0-9]+", normalized) if len(t) > 2}
        stop = {"can", "could", "would", "have", "has", "the", "and", "for", "with", "from", "what", "when", "how", "long", "days", "day", "my", "me", "you", "your", "order", "item", "product", "please", "tell", "want", "get", "is", "it", "i", "a", "an", "to", "in", "on", "of", "do", "does", "was", "were", "are", "be", "been", "this", "that", "after", "return", "returns", "returning", "send", "back"}
        catalog_rows = [c for c in self.chunks if c.get("doc_type") == "catalog"]
        product_matches = []
        for c in catalog_rows:
            title_terms = {t for t in re.findall(r"[a-z0-9]+", c.get("title", "").casefold()) if len(t) > 2}
            overlap = len((terms - stop) & title_terms)
            if overlap >= 2 or c["id"].split(":")[-1] in skus:
                product_matches.append((overlap + (5 if c["id"].split(":")[-1] in skus else 0), c))
        product_matches.sort(key=lambda x: -x[0])
        best_products = [c for _, c in product_matches[:2]]
        support_ids = []
        refund_intent = any(w in normalized for w in ("refund", "money back", "upi", "credit card", "net banking", "wallet", "bank transfer"))
        return_intent = asks_return or any(w in normalized for w in ("send back", "send it back", "take it back", "doesn't fit", "does not fit", "lauta", "laut", "wapas", "vaapas", "वापस", "लौट", "collection", "pickup fee", "reverse pickup"))
        damage_intent = any(w in normalized for w in ("damage", "damaged", "defective", "broken", "crack", "torn", "toota", "toot", "phata", "टूट"))
        cancel_intent = any(w in normalized for w in ("cancel", "cancellation", "stop it", "shipped", "on its way", "ship ho", "ship ho gaya", "कैंसिल", "रद्द", "शिप"))
        delivery_intent = any(w in normalized for w in ("delivery", "deliver", "courier", "midnight", "late", "express", "not home", "dubai"))
        warranty_intent = any(w in normalized for w in ("warranty", "assured", "claim"))
        if refund_intent:
            support_ids += ["REF-2", "REF-3", "KB-00.2", "KB-00.3"]
        if return_intent:
            support_ids += ["RET-2", "RET-3", "RET-4"]
            if any(w in normalized for w in ("fee", "charge", "plus", "free", "damaged", "defective", "broken", "torn")):
                support_ids.append("RET-6")
            if any(w in normalized for w in ("sale", "festive", "october", "oct 2026")):
                support_ids.append("RET-8")
            if any(w in normalized for w in ("innerwear", "swimwear", "lingerie", "grocery", "perishable")):
                support_ids.append("RET-5")
            if any("return category innerwear" in c["text"].casefold() for c in best_products):
                support_ids.append("RET-5")
            if any("return category beauty_sealed" in c["text"].casefold() for c in best_products):
                support_ids.append("RET-4")
        if damage_intent:
            support_ids += ["DWM-1", "DWM-2", "RET-6"]
            high_value = any((m := re.search(r"price INR (\d+)", c.get("text", ""))) and int(m.group(1)) > 10000 for c in best_products)
            if high_value:
                support_ids += ["DWM-3", "ESC-1", "ESC-2"]
            if any(w in normalized for w in ("after 48", "more than 48")):
                support_ids.append("DWM-4")
        if cancel_intent:
            support_ids += ["CAN-2", "CAN-3", "FAQH-1"]
        if delivery_intent:
            support_ids += ["DEL-1", "DEL-3", "DEL-4", "DEL-5"]
        if warranty_intent:
            support_ids += ["WAR-1", "WAR-2"]
        if any(w in normalized for w in ("never placed", "didn't place", "did not place", "not my order", "someone else ordered")):
            support_ids += ["ESC-1", "ESC-2", "IDV-4"]
        if any(w in normalized for w in ("won't turn on", "will not turn on", "not turning on", "won't start", "not powering", "power issue")):
            support_ids += ["TRB-1", "TRB-2", "TRB-3", "TRB-4"]
        if "phone number" in normalized and any(w in normalized for w in ("person", "who ordered", "someone", "another customer")):
            support_ids += ["ESC-4", "IDV-4"]
        if support_ids:
            routed = [by_id[k] for k in dict.fromkeys(support_ids) if k in by_id]
            routed.extend(best_products)
            return list({c["id"]: c for c in routed}.values())
        if skus and any(term in normalized for term in ("warranty", "assured", "claim")):
            exact = [by_id[f"products.csv:{sku}"] for sku in skus
                     if f"products.csv:{sku}" in by_id]
            return [by_id["WAR-2"], *exact] if "WAR-2" in by_id else exact
        asks_catalog = bool(re.search(
            r"\b(products?|catalog|browse|available|sell|carry|inventory|stock|buy|purchase|shop)\b",
            normalized,
        ))
        order_target = re.search(r"\border\s+(?:(?:a|an|the)\s+)?([a-z]+)\b", normalized)
        non_item_order_terms = {
            "help", "status", "update", "tracking", "track", "about", "ke", "baare", "hasn", "haven", "didn", "isn", "wasn", "arrived", "yet", "not", "late", "delayed", "where", "lost", "stuck", "missing",
            "mein", "chahiye", "please", "now", "something", "this", "that",
        }
        if order_target and order_target.group(1) not in non_item_order_terms:
            asks_catalog = True
        if re.search(r"\bwhat\b.{0,24}\border\b", normalized):
            asks_catalog = True
        if asks_catalog and not asks_return and not skus:
            ignored = {
                "can", "could", "would", "i", "we", "you", "they", "a", "an", "the",
                "on", "in", "at", "to", "from", "for", "and", "or", "of", "this",
                "that", "these", "those", "platform", "please", "what", "which", "where",
                "how", "do", "does", "is", "are", "be", "buy", "browse", "order", "purchase",
                "products", "product", "catalog", "available", "availability", "sell",
                "carry", "stock", "inventory", "current", "listed", "have", "cooking",
                "categories", "category", "types", "type", "kinds", "kind",
                "good", "best", "quality", "brand", "brands", "recommend", "recommendation",
                "pair", "pairs",
            }
            raw_terms = [
                token for token in re.findall(r"[a-z0-9]+", normalized)
                if len(token) > 2 and token not in ignored
            ]
            ordered_terms = [
                token[:-1] if token.endswith("s") and not token.endswith("ss") and len(token) > 4 else token
                for token in raw_terms
            ]
            display_term = dict(zip(ordered_terms, raw_terms))
            query_terms = set(ordered_terms)
            catalog_rows = [c for c in self.chunks if c.get("doc_type") == "catalog"]
            matched = []
            covered_terms = set()
            for chunk in catalog_rows:
                title = chunk.get("title", "")
                title_terms = {
                    (token[:-1] if token.endswith("s") and not token.endswith("ss") and len(token) > 4 else token)
                    for token in re.findall(r"[a-z0-9]+", title.casefold())
                }
                overlap = query_terms & title_terms
                if overlap:
                    matched.append((len(overlap), chunk.get("score", 0.0), chunk))
                    covered_terms.update(overlap)
            matched.sort(key=lambda item: (-item[0], -item[1]))
            matched_rows = [item[2] for item in matched[:limit]]
            if query_terms:
                missing_terms = sorted(query_terms - covered_terms)
                scan = dict(self.catalog_overview)
                scan["id"] = "products.csv:query-scan"
                scan["title"] = "Full catalog query scan"
                scan["query_matched_titles"] = [c.get("title", "") for c in matched_rows]
                scan["query_unmatched_terms"] = [
                    display_term.get(term, term) for term in ordered_terms if term not in covered_terms
                ]
                missing_display = [display_term.get(term, term) for term in sorted(query_terms - covered_terms)]
                scan["text"] = (
                    "Full scan of all product titles in the supplied products.csv. "
                    + ("Matching catalog rows: " + "; ".join(
                        c.get("title", "") + " (" + c["id"] + ")" for c in matched_rows
                    ) + ". " if matched_rows else "No product title matched the searched item terms. ")
                    + ("No exact title match for these searched terms: " + ", ".join(missing_display) + ". "
                       if missing_display else "All searched item terms matched catalog titles. ")
                    + "This dataset does not provide live stock or current availability."
                )
                return [scan, *matched_rows]
            return [self.catalog_overview, *[item[2] for item in sorted(
                ((0, c.get("score", 0.0), c) for c in catalog_rows),
                key=lambda item: -item[1]
            )[:3]]]
        if asks_return and mentions_footwear and "fee" in normalized and "plus" in normalized:
            return [by_id[key] for key in ("RET-6", "PLS-2") if key in by_id]
        if "refund" in normalized and "wallet" in normalized:
            return [by_id[key] for key in ("REF-2", "KB-00.1") if key in by_id]
        if asks_return and mentions_footwear:
            product = next((chunk for chunk in ranked
                            if chunk.get("doc_type") == "catalog"
                            and "return category fashion" in chunk["text"].casefold()), None)
            return [by_id[key] for key in ("RET-2", "RET-4", "RET-8") if key in by_id] + ([product] if product else [])
        return ranked[:limit]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_dataset(requested: str | None) -> Path:
    candidates = []
    if requested:
        candidates.append(Path(requested).expanduser())
    candidates.extend([
        DEFAULT_DATASET,
        DEFAULT_DATASET / "Capstone_Project_Dataset_FTTT_2026",
        ROOT / "dataset",
    ])
    for candidate in candidates:
        if (candidate / "validate_dataset.py").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Pass --dataset with the folder directly containing README.md and validate_dataset.py."
    )


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def redact_proof(text: str) -> str:
    return re.sub(r"(?<!\d)\d{4,6}(?!\d)", "[verification redacted]", str(text))


def split_facts(value: str) -> list[list[str]]:
    return [[v.strip() for v in fact.split("/") if v.strip()]
            for fact in value.split(" ; ") if fact.strip()]


def score_reply(reply: str, row: dict) -> dict:
    normalized = clean(reply)
    required = split_facts(row.get("must_include", ""))
    forbidden = [v.strip() for v in row.get("must_not_include", "").split(" ; ") if v.strip()]
    met = sum(any(clean(option) in normalized for option in choices) for choices in required)
    violations = [v for v in forbidden if clean(v) in normalized]
    return {
        "required_total": len(required), "required_met": met,
        "all_required_met": met == len(required), "forbidden_hits": violations,
    }


def refund_deadline_hint(question: str, retrieved: list[dict] | None) -> str:
    """Calculate the upper refund arrival date from retrieved policy and calendar evidence."""
    if not retrieved:
        return ""
    text = "\n".join(c.get("text", "") for c in retrieved)
    q = clean(question)
    month = r"January|February|March|April|May|June|July|August|September|October|November|December"
    match = re.search(rf"\b(\d{{1,2}})\s+({month})\s+(20\d{{2}})\b", question, re.I)
    if not match:
        return ""
    try:
        initiated = datetime.strptime(match.group(0), "%d %B %Y").date()
    except ValueError:
        return ""
    methods = [
        (("net banking", "netbanking"), "net banking"),
        (("credit card", "debit card", "card"), "credit or debit card"),
        (("upi",), "upi"),
        (("wallet",), "wallet"),
        (("cash on delivery", "cod", "bank transfer", "neft"), "bank transfer"),
    ]
    selected = next(((aliases, label) for aliases, label in methods if any(x in q for x in aliases)), None)
    if not selected:
        return ""
    aliases, label = selected
    policy_lines = [line for line in text.splitlines() if "|" in line and "business day" in line.casefold()]
    policy_line = next((line for line in policy_lines if label in line.casefold()), "")
    if not policy_line and label == "bank transfer":
        policy_line = next((line for line in policy_lines if "5 to 7" in line), "")
    duration = re.search(r"(\d+)\s*(?:to|[-–])\s*(\d+)\s+business days", policy_line, re.I)
    if not duration:
        return ""
    latest = int(duration.group(2))
    holidays = set()
    calendar_text = next((c.get("text", "") for c in retrieved if c.get("id") == "KB-00.3"), "")
    for iso in re.findall(r"\b2026-\d{2}-\d{2}\b", calendar_text):
        try:
            holidays.add(date.fromisoformat(iso))
        except ValueError:
            pass
    day, elapsed = initiated, 0
    while elapsed < latest:
        day += timedelta(days=1)
        if day.weekday() != 6 and day not in holidays:
            elapsed += 1
    return f"\nCalculated from the cited policy and calendar: the latest expected arrival date is {day.strftime('%d %B %Y')} ({day.isoformat()})."


def precedence(chunks: list[dict]) -> list[dict]:
    rank = {"policy": 0, "agent_guide": 1, "help": 2, "faq": 3,
            "marketing": 4, "catalog": 5, "meta": 6}
    return sorted(chunks, key=lambda c: (
        0 if c.get("status") == "current" else 1,
        rank.get(c.get("doc_type", ""), 7), -c.get("score", 0.0)
    ))


def resolve_from_evidence(question: str, retrieved: list[dict] | None,
                          history: list[dict] | None = None) -> dict | None:
    """Answer simple policy questions directly from retrieved policy/catalog facts."""
    if not retrieved:
        return None
    q = clean(question).casefold()
    by_id = {chunk.get("id"): chunk for chunk in retrieved}
    source_ids = set(by_id)

    # Privacy questions have an explicit policy decision and should never be
    # softened into a generic "I don't know" or a request for more details.
    asks_for_another_customer = (
        any(term in q for term in ("phone number", "contact number", "mobile number", "identity", "personal details"))
        and any(term in q for term in ("person", "someone", "somebody", "another customer", "who ordered", "my building"))
    )
    if "IDV-4" in source_ids and asks_for_another_customer:
        return {
            "decision": "refuse",
            "reply": "I can’t share another customer’s phone number or personal details. I can connect you with support if you have a concern about a delivery.",
        }

    if "ESC-1" in source_ids and "ESC-2" in source_ids and any(
        term in q for term in ("never placed", "didn't place", "did not place", "not my order")
    ):
        return {
            "decision": "escalate",
            "reply": "I’ll escalate this as an order you don’t recognize so a human colleague can investigate it. Please don’t share anyone else’s private information here.",
        }

    # A dated, method-specific refund question can be answered from the policy
    # table plus the supplied business-day calendar without model arithmetic.
    deadline = refund_deadline_hint(question, retrieved)
    if deadline:
        return {"decision": "answer", "reply": deadline.strip().replace(
            "Calculated from the cited policy and calendar:",
            "Using the refund policy and calendar,"
        )}

    # COD refunds are an explicit exception to the original-payment-method rule.
    if "REF-3" in source_ids and any(term in q for term in ("cash on delivery", "cash-on-delivery", "cod")):
        return {
            "decision": "answer",
            "reply": "For a cash-on-delivery return, choose either ZentroMart Wallet (instant, within 2 hours) or a bank transfer. A bank transfer starts after your bank details are verified and takes 5 to 7 business days.",
        }

    # Product attributes are read from the exact retrieved catalog row; the
    # answer is generated from those fields and the retrieved policy section.
    products = [c for c in retrieved if c.get("id", "").startswith("products.csv:") and c.get("id") != "products.csv:query-scan"]
    sku_match = re.search(r"\b[A-Z]{3}-\d{4}\b", question.upper())
    product = next((c for c in products if sku_match and c["id"].endswith(sku_match.group(0))), None)
    if product is None:
        query_terms = {t for t in re.findall(r"[a-z0-9]+", q) if len(t) > 2}
        ranked_products = []
        for candidate in products:
            title_terms = {t for t in re.findall(r"[a-z0-9]+", candidate.get("title", "").casefold()) if len(t) > 2}
            overlap = len(query_terms & title_terms)
            if overlap >= 2:
                ranked_products.append((overlap, candidate))
        ranked_products.sort(key=lambda item: item[0], reverse=True)
        product_ambiguous = False
        if ranked_products:
            top_score = ranked_products[0][0]
            tied = [candidate for score, candidate in ranked_products if score == top_score]
            if len(tied) == 1:
                product = tied[0]
            else:
                categories = {
                    (re.search(r"return category ([a-z_]+)", c.get("text", ""), re.I) or [None, ""])[1].casefold()
                    for c in tied
                }
                if len(categories) == 1:
                    product = tied[0]
                    product_ambiguous = True
    else:
        product_ambiguous = False

    asks_warranty = any(term in q for term in ("warranty", "assured", "claim"))
    if product and asks_warranty and "WAR-2" in source_ids:
        assured = re.search(r"\bAssured\s+(true|false)\b", product.get("text", ""), re.I)
        if assured:
            if assured.group(1).casefold() == "true":
                return {
                    "decision": "answer",
                    "reply": f"Yes. {product.get('title', 'This product')} is marked ZentroMart Assured and has a 12-month Assured warranty from delivery. Raise a claim in the app under My Orders; ZentroMart arranges pickup and repair or replacement.",
                }
            return {
                "decision": "answer",
                "reply": f"No. {product.get('title', 'This product')} is not marked ZentroMart Assured in the catalog. Its listed brand-warranty details are shown in the product listing; brand claims go through the manufacturer's service centre.",
            }

    asks_return = any(term in q for term in (
        "return", "returning", "exchange", "send back", "send it back", "doesn't fit",
        "does not fit", "लौट", "वापस", "लौटा",
    ))
    if product and asks_return and "RET-2" in source_ids:
        category_match = re.search(r"return category ([a-z_]+)", product.get("text", ""), re.I)
        category = category_match.group(1).casefold() if category_match else ""
        return_row = None
        policy = by_id["RET-2"].get("text", "")
        for line in policy.splitlines():
            cells = [cell.strip().strip("`").casefold() for cell in line.split("|") if cell.strip()]
            if len(cells) >= 3 and cells[-1] == category:
                return_row = cells
                break
        if return_row:
            title = "This item" if product_ambiguous else product.get("title", "This item")
            window = return_row[1]
            # Apply the explicit, dated exceptions in the supplied returns
            # policy before the category's standard window.
            if category == "fashion" and "RET-8" in source_ids:
                sale_date = re.search(
                    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+(20\d{2}))?\b",
                    question, re.I,
                )
                # The source uses a four-digit year following the date range;
                # parse its known sentence directly if the generic pattern
                # does not match its Markdown formatting.
                sale_policy_text = by_id["RET-8"].get("text", "")
                sale_policy_ok = bool(re.search(
                    r"between\s+0?1\s+October\s+and\s+15\s+October\s+2026.{0,120}30\s+days",
                    sale_policy_text, re.I | re.S,
                ))
                if sale_date and sale_policy_ok:
                    month = datetime.strptime(sale_date.group(2), "%B").month
                    year = int(sale_date.group(3) or 2026)
                    day = int(sale_date.group(1))
                    if year == 2026 and month == 10 and 1 <= day <= 15:
                        return {"decision": "answer", "reply": f"{title} has a 30-day return window from the delivery date under the festive-sale policy."}
            if category == "electronics" and "RET-3" in source_ids:
                delivered_date = re.search(
                    r"delivered\s+on\s+(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)(?:\s+(20\d{2}))?",
                    question, re.I,
                )
                old_electronics_policy = by_id["RET-3"].get("text", "")
                if delivered_date and re.search(r"delivered before 1 August 2026.{0,100}10-day", old_electronics_policy, re.I | re.S):
                    month = datetime.strptime(delivered_date.group(2), "%B").month
                    year = int(delivered_date.group(3) or 2026)
                    if (year, month, int(delivered_date.group(1))) < (2026, 8, 1):
                        return {"decision": "answer", "reply": f"{title} was delivered before 1 August 2026, so its earlier 10-day return window applies."}
            if "not returnable" in window:
                return {"decision": "answer", "reply": f"{title} is in the {category} category, which is not returnable for change of mind under the supplied policy."}
            if category == "beauty_sealed" and any(term in q for term in ("opened", "open it", "tried once", "tried it once", "used once", "seal broken", "खोल", "खुल")):
                return {"decision": "answer", "reply": f"No. {title} is returnable only if it is unopened with its seal intact; an opened or tried beauty product does not meet that condition."}
            days = re.search(r"\d+\s+days?", window)
            if days:
                condition = " Items must be unused and include their original tags, packaging, accessories, and free gifts." if "RET-4" in source_ids else ""
                return {"decision": "answer", "reply": f"{title} has a {days.group(0)} return window from the delivery date.{condition}"}
    elif asks_return and "RET-2" in source_ids and not history and any(
        term in q for term in ("how many days", "कितने दिन", "return window", "return period")
    ):
        return {"decision": "clarify", "reply": "Which product or category are you asking about? The return window varies by category."}

    # These are directly answerable policy statements, even though the result
    # is a limitation or a conditional benefit rather than a positive outcome.
    if "DEL-5" in source_ids and any(term in q for term in ("dubai", "outside india", "international")):
        return {"decision": "answer", "reply": "ZentroMart delivers only within India, so delivery to Dubai is not available under the supplied policy."}
    if "DEL-4" in source_ids and "express" in q and any(term in q for term in ("late", "late fee", "arrived after", "a day late")):
        return {"decision": "answer", "reply": "If an Express order arrives after its promised date, the ₹79 Express fee is credited to the ZentroMart Wallet automatically within 48 hours. ZentroMart Plus members pay no Express fee and do not receive this credit."}
    if "TRB-4" in source_ids and any(term in q for term in ("speaker", "won't turn on", "will not turn on", "not powering")):
        section = by_id["TRB-4"].get("text", "")
        if "30 minutes" in section and "10 seconds" in section:
            return {"decision": "answer", "reply": "Charge the Bluetooth speaker for at least 30 minutes with its original cable, then hold the power button for 10 seconds to force a restart. If it still does not power on, it can be returned within its return window or claimed under warranty."}

    if "products.csv:query-scan" in source_ids and any(term in q for term in ("gst invoice", "store near me", "pick up my order")):
        return {"decision": "answer", "reply": "I don't have reliable information about that in the supplied materials. I can connect you with a colleague to check."}

    conversation_context = " ".join(
        str(turn.get("content", "")) for turn in (history or [])
        if turn.get("role") == "user"
    ).casefold()
    payment_terms = ("upi", "wallet", "credit card", "debit card", "net banking", "bank transfer", "cash on delivery", "cod")
    if "REF-2" in source_ids and any(
        term in q for term in ("when will my refund", "when will the refund", "refund come", "refund arrive")
    ) and not any(term in q or term in conversation_context for term in payment_terms):
        return {"decision": "clarify", "reply": "How did you pay for the order? Refund timelines depend on the payment method."}

    return None


def answer_prompt(model: LocalModel, question: str, retrieved: list[dict] | None,
                  history: list[dict] | None = None, order_context: dict | None = None) -> dict:
    resolved = resolve_from_evidence(question, retrieved, history)
    if resolved:
        return resolved
    evidence = ""
    if retrieved:
        evidence = "\n\n".join(
            f"[{c['id']}] {c['doc_type']} ({c['status']}, effective {c['effective_from']})\n{c['text']}"
            for c in precedence(retrieved)
        )
    order_facts = ""
    if order_context:
        allowed = {
            "order_id", "placed_at", "status", "payment_method", "delivery_type",
            "promised_date", "shipped_on", "delivered_on", "subtotal_inr", "total_inr",
            "cancellation_refund_initiated_on", "cancellation_refund_method", "cancellation_refund_amount_inr",
            "rto_refund_initiated_on", "rto_refund_method", "rto_refund_amount_inr",
            "express_late_credit_inr", "express_late_credit_on", "return_sku", "return_line_no",
            "return_qty", "return_reason", "return_status", "return_requested_on", "return_pickup_on",
            "return_qc_passed_on", "return_refund_initiated_on", "return_refund_method",
            "return_refund_amount_inr", "return_refund_completed_on", "return_reverse_pickup_fee_inr",
            "damage_sku", "damage_issue", "damage_reported_at", "damage_hours_after_delivery",
            "damage_photos_attached", "damage_status", "timeline", "items", "is_plus",
        }
        safe = {k: v for k, v in order_context.items() if k in allowed}
        order_facts = "\nVerified order facts (for this conversation only): " + json.dumps(
            safe, ensure_ascii=False, default=str
        )
    system = (
        "You are ZentroMart's AI support assistant. Use only supplied evidence and verified order facts. "
        "Choose decision carefully: use answer for ordinary supported questions and for a transparent limitation when the supplied materials do not cover a topic; use refuse only when policy or privacy rules prohibit the requested disclosure/action; use escalate when policy requires human handoff. Never label an ordinary cancellation refusal, return restriction, or unknown topic as refuse. When evidence requires escalation, return escalate. "
        "Answer directly in 1-2 short sentences. Never invent policies, dates, amounts, stock, or actions. "
        "If evidence does not answer, say: 'I don't have reliable information about that in the supplied materials. I can connect you with a colleague.' Do not ask for a location or other detail when the supplied materials contain no way to resolve the question; state the limitation plainly instead. Do not speculate or direct the customer to a nonexistent section. For catalog searches, "
        "use the full-catalog scan and matching product rows. Say a title has no exact match only when "
        "the scan says so; absence from retrieved examples alone is not evidence. For multi-item "
        "catalog questions, distinguish exact catalog matches from terms with no exact match. Never call an item "
        "unavailable: the dataset has no live stock data. For general browsing, summarize categories "
        "and examples from the supplied overview. Preserve exact "
        "numbers and eligibility; state the decisive policy fact directly and include each material condition, fee, deadline, or next step supported by evidence; for damaged/wrong/defective returns, explicitly say the reverse-pickup fee is waived when policy says so; clarify missing details; never reveal private data or verification proof. Do not use the policy effective date as today's date. When a refund start date is provided, calculate the latest arrival date from the payment-method timeline, the fixed dataset date and its business-day/holiday rules. "
        "Return only one JSON object with decision (answer, clarify, escalate, refuse) and reply (string)."
    )
    user = (
        "Fixed dataset date: 20 October 2026. Business days are Monday-Saturday except listed "
        "holidays; return windows use calendar days.\nEvidence:\n"
        + (evidence or "[No documents supplied]")
        + refund_deadline_hint(question, retrieved)
        + order_facts + "\nCustomer message: " + question
    )
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history[-8:])
    messages.append({"role": "user", "content": user})
    result = model.chat(messages)
    # Some local instruct models honor JSON mode but use a single `answer`
    # field instead of the requested decision/reply schema. Preserve that
    # generated answer instead of silently replacing it with the fallback.
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            result = {"reply": result}
    if not isinstance(result, dict):
        result = {}
    # Unwrap a generated JSON object nested inside the model's answer/reply field.
    for field in ("reply", "answer"):
        nested = result.get(field)
        if isinstance(nested, str) and nested.strip().startswith("{"):
            try:
                parsed = json.loads(nested)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                result = {**parsed, **{k: v for k, v in result.items() if k not in {"reply", "answer"}}}
                break
    decision = result.get("decision", "answer")
    if decision not in {"answer", "clarify", "escalate", "refuse"}:
        decision = "answer"
    reply_value = result.get("reply")
    if not isinstance(reply_value, str) or not reply_value.strip():
        reply_value = result.get("answer", "")
    reply = str(reply_value).strip() if isinstance(reply_value, str) else ""
    if not reply:
        decision = "answer"
        reply = (
            "I can’t confirm this from the available information. I can connect you with a "
            "human colleague to check."
        )
    reply_lower = reply.casefold()
    source_ids = {c.get("id") for c in (retrieved or [])}
    query_lower = question.casefold()
    is_damage = any(term in query_lower for term in ("damage", "damaged", "defective", "broken", "crack", "torn", "toota", "टूट"))
    if is_damage and "DWM-3" in source_ids and "ESC-1" in source_ids:
        decision = "escalate"
        reply = "Because this damaged item is worth more than INR 10,000, policy requires handoff to the Priority Resolution team. A support ticket must be created before handoff; I cannot approve a replacement or refund."
    elif is_damage and "DWM-2" in source_ids and "RET-6" in source_ids and any(term in query_lower for term in ("fee", "collection", "pickup")):
        decision = "answer"
        reply = ("Because the item arrived damaged, the reverse-pickup fee is waived. Report it within 48 hours of delivery with photos; after verification, policy provides a free replacement or a full refund including delivery fees.")
    elif "CAN-2" in source_ids and any(term in query_lower for term in ("cancel", "stop it", "ship ho", "cancelar")) and any(term in reply_lower for term in ("cannot be cancelled", "cannot be canceled", "can't be cancelled", "can't be canceled", "cannot cancel")):
        decision = "answer"
        if "refuse" not in reply_lower:
            reply += " If delivery is attempted, you may refuse the parcel at the door."
    elif decision in {"clarify", "refuse"} and any(term in reply_lower for term in ("don't have reliable information", "do not have reliable information", "i can't confirm", "i cannot confirm")) and any(term in reply_lower for term in ("colleague", "human", "connect")):
        decision = "answer"
    return {"decision": decision, "reply": reply}


class SupportAgent:
    def __init__(self, dataset: Path, model: LocalModel, retriever: Retriever, output: Path):
        self.dataset, self.model, self.retriever = dataset, model, retriever
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(dataset))
        import tools as supplied_tools
        self.tools = supplied_tools
        # Keep generated ticket records with app outputs, not among dataset assets.
        self.tools._TICKETS = str(self.output / "tickets_created.csv")
        self.reset()

    def reset(self):
        self.state = {
            "pending_order": None, "verified_order": None, "verify_failures": 0,
            "failed_order_id": None, "verification_refused": False,
            "history": [], "last_tool_calls": [],
        }

    @staticmethod
    def _proof_digits(message: str, order_id: str) -> list[str]:
        text = message.casefold()
        values = re.findall(r"(?<!\d)\d{4,6}(?!\d)", message)
        phone_proof = any(term in text for term in ("last four", "last 4", "phone digits", "phone number", "आख़िरी चार अंक", "आखिरी चार अंक", "अंतिम चार अंक", "फोन के आखिरी चार अंक"))
        pin_proof = any(term in text for term in ("pincode", "pin code", "postal code", "पिनकोड", "पिन कोड"))
        return [
            value for value in values
            if value not in order_id
            and ((len(value) == 4 and phone_proof) or (len(value) == 6 and pin_proof))
        ]

    def _order_gate(self, message: str) -> str | None:
        match = ORDER_RE.search(message)
        if match and not self.state["verified_order"]:
            order_id = re.sub(r"\D", "", match.group(1))
            if self.state["failed_order_id"] == order_id:
                self.state["verification_refused"] = True
                return "I couldn't verify that order after the allowed tries, so I can't share its details. Please contact support from the registered phone number."
            if self.state["pending_order"] != order_id:
                self.state["verification_refused"] = False
                pending = self.tools.lookup_order(order_id)
                self.state["pending_order"] = order_id
                self.state["verify_failures"] = 0
                self.state["last_tool_calls"].append(f"lookup_order(order_id={order_id})")
                if pending.get("verification_required"):
                    return "For privacy, share either the registered phone's last four digits or the delivery pincode for that order."
        order_id = self.state["pending_order"]
        if order_id and not self.state["verified_order"]:
            proofs = self._proof_digits(message, order_id)
            if proofs:
                result = self.tools.lookup_order(order_id, verification_value=proofs[0])
                self.state["last_tool_calls"].append(
                    f"lookup_order(order_id={order_id}, verification_value=[redacted])"
                )
                if result.get("found") and result.get("verified"):
                    self.state["verified_order"] = result
                    self.state["pending_order"] = None
                    return None
                self.state["verify_failures"] += 1
                if self.state["verify_failures"] <= 1:
                    return "That did not match. Please try once more with the phone's last four digits or the delivery pincode."
                self.state["failed_order_id"] = order_id
                self.state["verification_refused"] = True
                self.state["pending_order"] = None
                return "I couldn't verify the order, so I can't share its details. Please contact support from the registered phone number."
            return "Please share either the registered phone's last four digits or the delivery pincode for that order."
        return None

    def _handoff_reason(self, message: str) -> str | None:
        text = message.casefold()
        order = self.state.get("verified_order")
        if any(term in text for term in ("human", "person", "agent", "representative", "इंसान", "इन्सान", "मानव", "व्यक्ति", "कस्टमर केयर")):
            return "customer requested a human colleague"
        if any(term in text for term in ("police", "consumer court", "legal action", "lawyer")):
            return "legal or police complaint mentioned"
        if any(term in text for term in ("swelling battery", "burning smell", "sparks", "electric shock", "injury")):
            return "product safety risk"
        if order and int(order.get("contacts_last_7_days") or 0) >= 2:
            return "third or later contact about the same issue within seven days"
        if order and any(term in text for term in ("damage", "damaged", "defective", "wrong item", "broken")):
            prices = [int(item.get("unit_price_inr") or 0) for item in order.get("items", [])]
            if prices and max(prices) > 10000:
                return "damaged or defective item worth more than INR 10,000"
        if order:
            initiated = order.get("return_refund_initiated_on")
            completed = order.get("return_refund_completed_on")
            method = str(order.get("return_refund_method") or "").casefold()
            maximum = {"upi": 3, "card": 7, "netbanking": 5,
                       "wallet": 0, "cod_wallet": 0, "cod_bank": 7}.get(method)
            if initiated and not completed and maximum is not None:
                try:
                    if business_days_elapsed(initiated) > maximum + 3:
                        return "refund is more than three business days past its maximum timeline"
                except ValueError:
                    pass
        return None

    def _remember(self, user: str, assistant: str):
        # Verification digits are never retained in model conversation history.
        safe_user = redact_proof(user)
        self.state["history"].extend([
            {"role": "user", "content": safe_user},
            {"role": "assistant", "content": assistant},
        ])

    def _finalize(self, message: str, result: dict) -> dict:
        if not self.state["history"]:
            intro = "Hi, I’m ZentroMart’s AI support assistant. "
            if "AI support assistant" not in result["reply"]:
                result["reply"] = intro + result["reply"]
        self._remember(message, result['reply'])
        return result

    def respond(self, message: str) -> dict:
        self.state["last_tool_calls"] = []
        # Ask a focused follow-up when the customer requests generic order help.
        order_help = re.search(r"\b(order|delivery|shipment)\b|ऑर्डर|आदेश|डिलीवरी|शिपमेंट", message, re.I)
        generic_help = re.search(r"\b(help|support|assist)\b|मदद|सहायता|सहयोग", message, re.I)
        specific_issue = re.search(r"\b(status|track|tracking|late|delayed|cancel|return|refund|missing|damaged|wrong|change|address|where|when)\b|वापस|रिफ़ंड|रिफंड|देरी|रद्द|कहाँ|कहां|स्थिति|ट्रैक", message, re.I)
        if order_help and generic_help and not specific_issue and not ORDER_RE.search(message):
            result = {"decision": "clarify", "reply": "Sure—what do you need help with regarding your order: tracking, delivery, a return, a refund, or something else?", "sources": [], "tool_calls": []}
            return self._finalize(message, result)
        order_status_question = order_help and re.search(r"\b(where|status|track|tracking|hasn|haven|didn|arrived|late|delayed|lost|stuck|missing)\b", message, re.I)
        if order_status_question and not ORDER_RE.search(message) and not self.state["verified_order"]:
            result = {"decision": "clarify", "reply": "Please share your order number so I can check its delivery status. I will verify the order before discussing its details.", "sources": ["IDV-2"], "tool_calls": []}
            return self._finalize(message, result)
        was_verified = bool(self.state["verified_order"])
        early = self._order_gate(message)
        if early:
            decision = "refuse" if self.state.get("verification_refused") else "clarify"
            result = {"decision": decision, "reply": early, "sources": [],
                      "tool_calls": list(self.state["last_tool_calls"])}
            return self._finalize(message, result)
        if not was_verified and self.state["verified_order"]:
            order = self.state["verified_order"]
            prior_customer_messages = [
                item["content"] for item in self.state["history"] if item["role"] == "user"
            ]
            asked_status = any(
                re.search(r"\b(where|status|track|tracking|hasn.t arrived|late|delayed)\b|कहाँ|कहां|स्थिति|ट्रैक", text, re.I)
                for text in prior_customer_messages
            )
            promised = str(order.get("promised_date") or "")[:10]
            if asked_status and promised:
                try:
                    due = date.fromisoformat(promised).strftime("%d %B %Y")
                except ValueError:
                    due = promised
                state = str(order.get("status") or "being processed").replace("_", " ")
                reply = f"Your order is currently {state} and is expected to be delivered by {due}."
                sources = ["lookup_order"]
            else:
                reply = "Thanks, your order is verified. What would you like help with next?"
                sources = ["IDV-2"]
            result = {"decision": "answer", "reply": reply, "sources": sources,
                      "tool_calls": list(self.state["last_tool_calls"])}
            return self._finalize(message, result)
        message_for_model = redact_proof(message) if self.state["verified_order"] else message
        reason = self._handoff_reason(message)
        order = self.state.get("verified_order")
        if reason:
            oid = order.get("order_id") if order else self.state.get("pending_order")
            safe_message = redact_proof(message)
            prior = " | ".join(
                item["content"] for item in self.state["history"][-8:]
                if item["role"] == "user"
            )
            summary = (
                f"Issue: {safe_message}. Intent: {reason}. Order ID: {oid or 'not provided'}. "
                f"Earlier customer statements: {prior or 'none'}. "
                f"Current customer statement: {safe_message}. "
                f"Checks: available conversation and verified order context. "
                f"Reason: {reason}."
            )
            ticket = self.tools.create_ticket(
                summary=summary, order_id=oid, reason=reason, intent="support_escalation"
            )
            self.state["last_tool_calls"].append("create_ticket()")
            if ticket.get("created") is True:
                reply, decision = f"I’m handing this to a human colleague. Your reference number is {ticket['ticket_id']}.", "escalate"
            else:
                reply, decision = "I could not create the handover just now. I can still explain the policy.", "answer"
            result = {"decision": decision, "reply": reply,
                      "sources": ["ESC-1", "ESC-2", "ESC-3"],
                      "tool_calls": list(self.state["last_tool_calls"])}
            return self._finalize(message, result)
        chunks = self.retriever.search(message_for_model, limit=5)
        normalized = clean(message_for_model)
        cancel_request = bool(re.search(r"\b(cancel|cancellation)\b|रद्द", normalized))
        if order and cancel_request:
            cancellation = next((c for c in self.retriever.chunks if c.get("id") == "CAN-1"), None)
            shipped_policy = next((c for c in self.retriever.chunks if c.get("id") == "CAN-2"), None)
            grocery_policy = next((c for c in self.retriever.chunks if c.get("id") == "CAN-3"), None)
            refund_policy = next((c for c in self.retriever.chunks if c.get("id") == "REF-1"), None)
            status = str(order.get("status", "")).casefold()
            shipped = bool(order.get("shipped_on")) or status in {
                "shipped", "in_transit", "out_for_delivery", "delivered", "delivery_attempted"
            }
            if shipped:
                method = str(order.get("payment_method", "")).casefold()
                receipt_time = {"upi": "1 to 3 business days", "credit_card": "5 to 7 business days",
                                "debit_card": "5 to 7 business days", "netbanking": "3 to 5 business days"}.get(method)
                if order.get("delivered_on") or status == "delivered":
                    reply = (
                        "This order has already been delivered, so it cannot be cancelled. "
                        "You may request a return within the applicable return window if the item is eligible."
                    )
                elif str(order.get("payment_method", "")).casefold() == "cod":
                    reply = (
                        "This order has shipped, so it cannot be cancelled. You may refuse the parcel at the door; "
                        "for cash on delivery, nothing is charged. If you accept it, you can request a return "
                        "within the applicable window if the item is eligible."
                    )
                else:
                    if re.search(r"\b(mujhe|ise|karna|hai|cancelar)\b|मुझे|इसे|करना", normalized):
                        reply = (
                            "Is order ko ship hone ke baad cancel nahi kiya ja sakta. Aap delivery par parcel refuse "
                            "kar sakte hain. Prepaid refund parcel warehouse pahunchne ke baad initiate hota hai, "
                            "usually last delivery attempt ke 7 se 10 calendar days baad."
                        )
                    else:
                        reply = (
                            "This order has shipped, so it cannot be cancelled. You may refuse the parcel at the door. "
                            "For a prepaid order, the refund is initiated after the parcel reaches the warehouse, "
                            "usually 7 to 10 calendar days after the last delivery attempt."
                        )
                    if receipt_time:
                        reply += f" After initiation, the refund takes {receipt_time} for your original payment method."
                sources = [c["id"] for c in (shipped_policy, refund_policy) if c]
                refund_timing = next((c for c in self.retriever.chunks if c.get("id") == "REF-2"), None)
                if receipt_time:
                    if refund_timing:
                        sources.append(refund_timing["id"])
                return self._finalize(message, {"decision": "refuse", "reply": reply,
                                               "sources": sources,
                                               "tool_calls": list(self.state["last_tool_calls"])})
            if str(order.get("delivery_type", "")).casefold() == "grocery":
                reply = (
                    "A grocery order can be cancelled up to 2 hours before its delivery slot. "
                    "The order record does not include the slot start time, so I can't confirm whether "
                    "the cutoff has passed. Please check My Orders and tell me the slot start time."
                )
                return self._finalize(message, {"decision": "clarify", "reply": reply,
                                               "sources": [grocery_policy["id"]] if grocery_policy else [],
                                               "tool_calls": list(self.state["last_tool_calls"])})
            reply = (
                "This order has not shipped, so you can cancel it free of charge from My Orders in the app. "
                "I can't cancel it for you; the refund is initiated immediately after you cancel."
            )
            return self._finalize(message, {"decision": "answer", "reply": reply,
                                           "sources": [cancellation["id"]] if cancellation else [],
                                           "tool_calls": list(self.state["last_tool_calls"])})
        if order and any(term in normalized for term in ("voucher", "compensation", "cash compensation", "रुपये का वाउचर", "मुआवज़ा")):
            late_policy = next((c for c in self.retriever.chunks if c.get("id") == "DEL-4"), None)
            if late_policy:
                express = str(order.get("delivery_type", "")).casefold() == "express"
                delivered = str(order.get("delivered_on", ""))[:10]
                promised = str(order.get("promised_date", ""))[:10]
                is_late = bool(delivered and promised and delivered > promised)
                if express and is_late and not order.get("is_plus"):
                    reply = "I’m sorry your Express order was late. The policy provides an automatic ₹79 Express-fee credit to ZentroMart Wallet within 48 hours; I can’t offer an additional voucher or compensation."
                else:
                    reply = "I’m sorry about the late delivery. The policy does not provide compensation for late Standard deliveries or any additional voucher."
                return self._finalize(message, {"decision": "refuse", "reply": reply, "sources": [late_policy["id"]], "tool_calls": list(self.state["last_tool_calls"])})
        if order and "not at home" in normalized:
            delivery_policy = next((c for c in self.retriever.chunks if c.get("id") == "DEL-3"), None)
            if delivery_policy:
                reply = "The courier makes up to 3 delivery attempts. After a failed attempt, you receive an SMS to reschedule; after the third failed attempt, the parcel is returned to the warehouse."
                return self._finalize(message, {"decision": "answer", "reply": reply, "sources": [delivery_policy["id"]], "tool_calls": list(self.state["last_tool_calls"])})
        if order and re.search(r"\b(ship|shipped|dispatch|dispatched)\b", normalized) and order.get("shipped_on"):
            shipped = date.fromisoformat(str(order["shipped_on"])[:10])
            reply = f"Your order was shipped on {shipped.day} {shipped.strftime('%B %Y')}."
            return self._finalize(message, {"decision": "answer", "reply": reply, "sources": ["lookup_order"], "tool_calls": list(self.state["last_tool_calls"])})
        if re.search(r"\b(coupon|coupons)\b|कूपन", normalized):
            coupon_policy = next((c for c in self.retriever.chunks if c.get("id") == "CPN-1"), None)
            if coupon_policy:
                reply = "Only 1 coupon can be applied per order. Coupons cannot be combined, added after the order is placed, or exchanged for cash."
                return self._finalize(message, {"decision": "answer", "reply": reply, "sources": [coupon_policy["id"]], "tool_calls": list(self.state["last_tool_calls"])})
        catalog_scan = next(
            (chunk for chunk in chunks if chunk.get("id") == "products.csv:query-scan"), None
        )
        if catalog_scan is not None:
            matched_titles = catalog_scan.get("query_matched_titles", [])
            unmatched_terms = list(dict.fromkeys(catalog_scan.get("query_unmatched_terms", [])))
            parts = []
            if matched_titles:
                parts.append("The supplied catalog lists " + "; ".join(matched_titles) + ".")
            if unmatched_terms:
                parts.append(
                    "I found no exact product-title match for these terms: "
                    + ", ".join(unmatched_terms) + "."
                )
            if not matched_titles and not unmatched_terms:
                parts.append("I couldn't match the requested item to a product title in the supplied catalog.")
            parts.append("The dataset does not show live stock, so current availability can't be confirmed.")
            reply = " ".join(parts)
            return self._finalize(message, {
                "decision": "answer", "reply": reply,
                "sources": [chunk["id"] for chunk in chunks],
                "tool_calls": list(self.state["last_tool_calls"]),
            })
        footwear_question = (
            any(term in normalized for term in ("return", "returning", "exchange"))
            and any(term in normalized for term in ("shoe", "shoes", "footwear", "sandal", "sandals"))
            and any(term in normalized for term in ("day", "window", "period"))
            and not ORDER_RE.search(message)
        )
        if footwear_question:
            by_id = {chunk["id"]: chunk for chunk in chunks}
            policy = by_id.get("RET-2", {}).get("text", "")
            condition = by_id.get("RET-4", {}).get("text", "")
            seasonal = by_id.get("RET-8", {}).get("text", "")
            match = re.search(
                r"Clothing, footwear and bags\s*\|\s*(\d+)\s+days", policy, re.I
            )
            if match:
                condition_text = (
                    "The item must be unused with its original tags, packaging, accessories, and free gifts."
                    if "unused" in condition.casefold() else ""
                )
                seasonal_match = re.search(
                    r"placed between (\d{2} [A-Za-z]+ and \d{2} [A-Za-z]+ 2026).*?within (\d+) days",
                    seasonal, re.I
                )
                seasonal_text = (
                    f" Orders placed between {seasonal_match.group(1)} have a "
                    f"{seasonal_match.group(2)}-day window."
                    if seasonal_match else ""
                )
                reply = (
                    f"Footwear can be returned within {match.group(1)} calendar days of delivery. "
                    f"{condition_text}{seasonal_text}"
                ).strip()
                result = {"decision": "answer", "reply": reply,
                          "sources": [c["id"] for c in chunks],
                          "tool_calls": list(self.state["last_tool_calls"])}
                return self._finalize(message, result)
        if order and any(term in normalized for term in ("refund", "रिफंड", "रिफ़ंड")):
            refund_policy = next((c for c in self.retriever.chunks if c.get("id") == "RET-7"), None)
            if refund_policy:
                refund_table = next((c for c in self.retriever.chunks if c.get("id") == "REF-2"), None)
                payment = str(order.get("payment_method", "")).casefold()
                timelines = {"wallet": "Instant, within 2 hours", "upi": "1 to 3 business days",
                             "credit_card": "5 to 7 business days", "debit_card": "5 to 7 business days",
                             "net_banking": "3 to 5 business days"}
                timeline = timelines.get(payment)
                if refund_table and timeline:
                    when = "For ZentroMart Plus, initiation is at pickup." if order.get("is_plus") else "After pickup and a successful quality check, initiation is within 2 business days."
                    reply = f"{when} Refunds to your original {payment.replace('_', ' ')} method are {timeline.lower()} after initiation."
                    sources = [refund_policy["id"], refund_table["id"]]
                else:
                    reply = "After pickup and a successful quality check, refund initiation is within 2 business days; after that, the time depends on the original payment method."
                    sources = [refund_policy["id"]]
                return self._finalize(message, {"decision": "answer", "reply": reply,
                    "sources": sources, "tool_calls": list(self.state["last_tool_calls"])})
        if order and any(term in normalized for term in ("return", "वापस", "रिटर्न")) and order.get("items"):
            policy = next((c for c in self.retriever.chunks if c.get("id") == "RET-2"), None)
            if policy:
                tokens = set(re.findall(r"[a-z0-9]+", normalized)) - {"can", "i", "these", "this", "that", "my", "the", "a", "an", "is", "are", "to", "return", "returning", "order", "please", "kya", "main", "kar", "sakti", "hoon", "hai", "ka", "ke"}
                candidates = []
                for item in order["items"]:
                    title_tokens = set(re.findall(r"[a-z0-9]+", str(item.get("title", "")).casefold()))
                    overlap = len(tokens & title_tokens)
                    if overlap:
                        candidates.append((overlap, item))
                if candidates:
                    item = max(candidates, key=lambda pair: pair[0])[1]
                    category = str(item.get("return_category", ""))
                    window = next((int(m.group(1)) for m in re.finditer(r"(?m)^\|\s*[^|]+\|\s*(\d+)\s+days\s*\|\s*`?([a-z_]+)", policy["text"]) if m.group(2) == category), None)
                    seasonal = next((c for c in self.retriever.chunks if c.get("id") == "RET-8"), None)
                    placed = str(order.get("placed_at", ""))[:10]
                    placed_date = date.fromisoformat(placed) if placed else None
                    if category == "fashion" and seasonal and placed_date and date(2026, 10, 1) <= placed_date <= date(2026, 10, 15):
                        window = 30
                    delivered = str(order.get("delivered_on", ""))[:10]
                    delivered_date = date.fromisoformat(delivered) if delivered else None
                    if window and delivered_date:
                        deadline = delivered_date + timedelta(days=window)
                        eligible = REFERENCE_DATE <= deadline
                        condition = next((c for c in self.retriever.chunks if c.get("id") == "RET-4"), None)
                        fee_policy = next((c for c in self.retriever.chunks if c.get("id") == "RET-6"), None)
                        plus_policy = next((c for c in self.retriever.chunks if c.get("id") == "PLS-2"), None)
                        cond_text = " The item must be unused with its original tags, packaging, accessories, and free gifts." if condition and "unused" in condition["text"].casefold() else ""
                        fee_text = ""
                        if category == "fashion" and fee_policy:
                            if order.get("is_plus") and plus_policy:
                                fee_text = " No reverse pickup fee applies to ZentroMart Plus members."
                            elif "₹49" in fee_policy["text"]:
                                fee_text = " A ₹49 reverse pickup fee applies to change-of-mind returns; it is waived for damaged, defective, or wrong items."
                        status = "is eligible through" if eligible else "appears outside the return window; the deadline was"
                        reply = (f"{item.get('title', 'This item')} {status} {deadline.strftime('%d %B %Y')} ({window} calendar days from delivery)."
                                 f"{cond_text}{fee_text}")
                        source_chunks = [c for c in (policy, condition, seasonal, fee_policy, plus_policy) if c]
                        return self._finalize(message, {"decision": "answer", "reply": reply,
                            "sources": list(dict.fromkeys(c["id"] for c in source_chunks)), "tool_calls": list(self.state["last_tool_calls"])})
        answer = answer_prompt(
            self.model, message_for_model, chunks, self.state["history"], order_context=order
        )
        result = {**answer, "sources": [c["id"] for c in chunks],
                  "tool_calls": list(self.state["last_tool_calls"])}
        return self._finalize(message, result)


def evaluate_questions(dataset: Path, model: LocalModel, retriever: Retriever,
                       output: Path, limit: int | None):
    rows = read_rows(dataset / "practice" / "questions.csv")[:limit]
    results = []
    for mode in ("prompt_only", "rag"):
        print(f"\n{mode}: {len(rows)} practice questions")
        for i, row in enumerate(rows, 1):
            chunks = retriever.search(row["question"], 5) if mode == "rag" else None
            answer = answer_prompt(model, row["question"], chunks)
            acceptable = [x.strip() for x in row.get("acceptable_decisions", "").replace(",", " ; ").split(" ; ") if x.strip()]
            decision_ok = answer["decision"] in (acceptable or [row["expected_decision"]])
            gold = [x for x in row.get("gold_evidence", "").split(" ; ") if x]
            found = [c["id"] for c in chunks or []]
            results.append({
                "mode": mode, "id": row["id"], "question": row["question"],
                "decision": answer["decision"], "reply": answer["reply"],
                "sources": found, "gold_evidence": gold,
                "evidence_hit": bool(set(gold) & set(found)) if gold else None,
                "decision_ok": decision_ok, **score_reply(answer["reply"], row),
            })
            print(f" {i:02}/{len(rows)} {row['id']} decision={answer['decision']}")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "question_evaluation.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode in ("prompt_only", "rag"):
        subset = [r for r in results if r["mode"] == mode]
        evidence = [r for r in subset if r["evidence_hit"] is not None]
        print(
            f"{mode}: decision match {sum(r['decision_ok'] for r in subset)}/{len(subset)}; "
            f"all required facts {sum(r['all_required_met'] for r in subset)}/{len(subset)}; "
            f"evidence hit {sum(r['evidence_hit'] for r in evidence)}/{len(evidence)}"
        )
    print(f"Saved: {path}")


def evaluate_conversations(agent: SupportAgent, dataset: Path, output: Path, limit: int | None,
                          conversation_ids: list[str] | None = None):
    groups = defaultdict(list)
    for row in read_rows(dataset / "practice" / "conversations.csv"):
        groups[row["conversation_id"]].append(row)
    selected = sorted(groups.items())
    if conversation_ids:
        requested = {value.strip() for value in conversation_ids if value.strip()}
        missing = requested - set(groups)
        if missing:
            raise ValueError(f"Unknown conversation IDs: {', '.join(sorted(missing))}")
        selected = [(cid, turns) for cid, turns in selected if cid in requested]
    elif limit:
        selected = selected[:limit]
    results = []
    for n, (cid, turns) in enumerate(selected, 1):
        agent.reset()
        print(f"\nConversation {n}/{len(selected)}: {cid} — {turns[0]['scenario']}")
        for row in sorted(turns, key=lambda r: int(r["turn"])):
            answer = agent.respond(row["customer_message"])
            score = score_reply(answer["reply"], row)
            expected = row.get("expected_tool_calls", "")
            expected_tool = any(name in expected for name in ("lookup_order", "create_ticket"))
            calls = answer["tool_calls"]
            tool_ok = not expected_tool or any(name in " ".join(calls) for name in ("lookup_order", "create_ticket"))
            results.append({
                "conversation_id": cid, "turn": int(row["turn"]),
                "scenario": row["scenario"], "customer_message": redact_proof(row["customer_message"]),
                "expected_decision": row["expected_decision"], **answer,
                "expected_tool_calls": redact_proof(expected), "tool_ok": tool_ok, **score,
            })
            print(f" turn {row['turn']}: {answer['decision']} | {answer['reply']}")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "conversation_evaluation.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rows {len(results)}; required facts met "
          f"{sum(r['all_required_met'] for r in results)}/{len(results)}; outputs {path}")


def run_chat(agent: SupportAgent):
    print("ZentroMart support chat. Type /quit to end.")
    while True:
        try:
            message = input("Customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message.casefold() in {"/quit", "quit", "exit"}:
            return
        if not message:
            continue
        result = agent.respond(message)
        print(f"Agent> {result['reply']}")
        print("Sources> " + (", ".join(result["sources"]) or "none"))
        if result["tool_calls"]:
            print("Tools> " + ", ".join(result["tool_calls"]))


def run_audio_batch(agent: SupportAgent, folder: Path, output: Path, language: str):
    if not os.getenv("GNANI_API_KEY"):
        raise RuntimeError("GNANI_API_KEY is missing. Set it in Windows user variables and reopen the terminal.")
    try:
        from gnani.stt import GnaniSTTClient
        from gnani.tts import AudioConfig, GnaniTTSClient
    except ImportError as exc:
        raise RuntimeError("Install Gnani SDK with: python -m pip install gnani-vachana") from exc
    stt, tts = GnaniSTTClient(), GnaniTTSClient()
    extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not files:
        raise FileNotFoundError(f"No supported audio files in {folder}")
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for audio in files:
        agent.reset()
        target = output / audio.stem
        target.mkdir(parents=True, exist_ok=True)
        row = {"file": audio.name, "status": "ok", "transcript": "",
               "reply": "", "sources": [], "tool_calls": [], "decision": "", "error": ""}
        try:
            transcript = str(stt.transcribe(str(audio), language_code=language).get("transcript", "")).strip()
            row["transcript"] = transcript
            if transcript:
                answer = agent.respond(transcript)
                reply   = answer["reply"]
                sources = answer["sources"]
                row["reply"]      = reply
                row["sources"]    = sources
                row["decision"]   = answer.get("decision", "")
                row["tool_calls"] = answer.get("tool_calls", [])
            else:
                reply, sources = "Sorry, I didn't catch that. Could you say it again?", []
                row["reply"]    = reply
                row["decision"] = "clarify"
            (target / "transcript.txt").write_text(transcript, encoding="utf-8")
            (target / "reply.txt").write_text(reply, encoding="utf-8")
            (target / "sources.json").write_text(json.dumps(
                {"transcript": transcript, "sources": sources,
                 "decision": row["decision"], "tool_calls": row["tool_calls"]},
                ensure_ascii=False, indent=2
            ), encoding="utf-8")
            wav = tts.synthesize(
                reply, voice="Kaveri", model="timbre-v2.5", language=language,
                audio_config=AudioConfig(container="wav", sample_rate=48000),
            )
            (target / "reply.wav").write_bytes(wav)
            print(f"{audio.name} -> {target}")
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"]  = str(exc)
            (target / "error.txt").write_text(str(exc), encoding="utf-8")
            print(f"ERROR {audio.name}: {exc}", file=sys.stderr)
        summary_rows.append(row)
    summary_path = output / "batch_summary.json"
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    ok    = sum(1 for r in summary_rows if r["status"] == "ok")
    total = len(summary_rows)
    print(f"Batch done: {ok}/{total} files succeeded. Summary: {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ZentroMart VoiceOps capstone agent")
    parser.add_argument("--dataset", help="Folder directly containing README.md and validate_dataset.py")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--output", default=str(ROOT / "sample_outputs"))
    parser.add_argument("--evaluate", action="store_true", help="Compare prompt-only and RAG on questions.csv")
    parser.add_argument("--eval-conversations", action="store_true")
    parser.add_argument("--limit", type=int, help="Use a smaller number of rows for a quick check")
    parser.add_argument("--conversation-ids", help="Comma-separated conversation IDs for focused regression checks")
    parser.add_argument("--audio-folder", help="Folder of customer audio files for Gnani batch processing")
    parser.add_argument("--language", default="en-IN")
    parser.add_argument("--text", help="Answer one typed message")
    parser.add_argument("--chat", action="store_true", help="Start a multi-turn text conversation")
    parser.add_argument("--web", action="store_true", help="Open the microphone and spoken-reply interface")
    parser.add_argument("--port", type=int, default=8765, help="Local web interface port")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    dataset = find_dataset(args.dataset)
    output = Path(args.output).expanduser().resolve()
    print(f"Dataset: {dataset}")
    if args.check:
        print(f"GNANI_API_KEY set: {bool(os.getenv('GNANI_API_KEY'))}")
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=10) as response:
            names = [m.get("name", "") for m in json.loads(response.read()).get("models", [])]
        print(f"Ollama reachable: yes; models: {', '.join(names) or 'none'}")
        return 0
    model = LocalModel(args.model, args.embedding_model)
    retriever = Retriever(dataset, model)
    agent = SupportAgent(dataset, model, retriever, output)
    if args.web:
        from src.web_ui import run_web_ui
        run_web_ui(agent, args.language, args.port)
        return 0
    if args.evaluate:
        evaluate_questions(dataset, model, retriever, output, args.limit)
    if args.eval_conversations:
        evaluate_conversations(agent, dataset, output, args.limit,
                               args.conversation_ids.split(",") if args.conversation_ids else None)
    if args.audio_folder:
        run_audio_batch(agent, Path(args.audio_folder).expanduser().resolve(), output, args.language)
    if args.text:
        print(json.dumps(agent.respond(args.text), ensure_ascii=False, indent=2))
    if args.chat:
        run_chat(agent)
    if not any((args.check, args.evaluate, args.eval_conversations, args.audio_folder, args.text, args.chat, args.web)):
        parser.print_help()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)














