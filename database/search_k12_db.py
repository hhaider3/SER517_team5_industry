#!/usr/bin/env python3
"""
search_k12_db_optimized.py

Optimized semantic search over the local K-12 image ChromaDB.

Improvements:
•⁠  ⁠Uses ChromaDB metadata filtering when possible (subject + optional strict grade overlap)
•⁠  ⁠Optional "soft" grade re-ranking when strict filtering is off (default)
•⁠  ⁠Cleaner output (compact/full)
•⁠  ⁠Safer handling of missing metadata keys
•⁠  ⁠Single DB query with includes (metadatas/distances/uris)

Usage:
  python3 search_k12_db_optimized.py "water cycle diagram" --grade 5 --n 6
  python3 search_k12_db_optimized.py "fractions pie chart" --subject Math --grade 4 --n 10
  python3 search_k12_db_optimized.py "blank map" --subject Geography --strict-grade --grade 6 --n 8
  python3 search_k12_db_optimized.py "plant cell" --n 5 --compact
"""

from _future_ import annotations

import argparse
import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

try:
    from chromadb.utils.data_loaders import ImageLoader
except Exception:
    ImageLoader = None

DB_PATH_DEFAULT = "./image_db"
COLLECTION_DEFAULT = "k12_education_images"

_TAG_RE = re.compile(r"<[^>]+>")


def clean_html(s: str) -> str:
    s = _TAG_RE.sub(" ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return default


def build_where(subject: Optional[str], grade: Optional[int], strict_grade: bool) -> Optional[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []

    if subject:
        clauses.append({"subject": subject})

    if strict_grade and grade is not None:
        clauses.append({"grade_min": {"$lte": grade}})
        clauses.append({"grade_max": {"$gte": grade}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def grade_penalty(grade: Optional[int], gmin: Optional[int], gmax: Optional[int]) -> float:
    if grade is None:
        return 0.0
    if gmin is None or gmax is None:
        return 0.15
    if gmin <= grade <= gmax:
        return 0.0
    d = min(abs(grade - gmin), abs(grade - gmax))
    return min(1.5, 0.20 * d)


def combine_score(distance: float, penalty: float) -> float:
    sim = 1.0 / (1.0 + max(0.0, float(distance)))
    return sim - penalty


def get_collection(db_path: str, collection_name: str):
    client = chromadb.PersistentClient(path=db_path)
    embedder = OpenCLIPEmbeddingFunction()

    kwargs = dict(name=collection_name, embedding_function=embedder)
    if ImageLoader:
        kwargs["data_loader"] = ImageLoader()

    return client.get_or_create_collection(**kwargs)


def fmt_range(gmin: Any, gmax: Any) -> str:
    if gmin is None and gmax is None:
        return "N/A"
    return f"{to_int(gmin, 0)} - {to_int(gmax, 12)}"


def print_result(idx: int, score: float, dist: float, uri: str, md: Dict[str, Any], compact: bool):
    subject = md.get("subject", "Unknown")
    topic = md.get("topic", "")
    gmin = md.get("grade_min")
    gmax = md.get("grade_max")
    license_name = md.get("license", "")
    artist = clean_html(md.get("artist", ""))
    credit = clean_html(md.get("credit", ""))
    source_page = md.get("source_page", md.get("source_url", ""))

    print(f"#{idx}  score={score:.3f}  dist={dist:.3f}")
    print(f"File: {uri}")
    print(f"Subject: {subject}")
    if topic:
        print(f"Topic: {topic}")
    print(f"Grades: {fmt_range(gmin, gmax)}")
    if license_name:
        print(f"License: {clean_html(str(license_name))}")

    if not compact:
        if artist:
            print(f"Artist: {artist}")
        if credit:
            print(f"Credit: {credit}")
        if source_page:
            print(f"Source page: {source_page}")

    print()


def main():
    ap = argparse.ArgumentParser(description="Optimized semantic search for local K-12 image DB.")
    ap.add_argument("query", help="Search query text")
    ap.add_argument("--db", default=DB_PATH_DEFAULT, help="ChromaDB persistent path (default: ./image_db)")
    ap.add_argument("--collection", default=COLLECTION_DEFAULT, help="Collection name (default: k12_education_images)")
    ap.add_argument("--n", type=int, default=6, help="Number of results to return (default: 6)")
    ap.add_argument("--subject", default=None, help="Filter by subject (exact match), e.g. Math, Science")
    ap.add_argument("--grade", type=int, default=None, help="Requested grade (used for filtering or re-ranking)")
    ap.add_argument("--strict-grade", action="store_true",
                    help="Only return results whose grade range includes --grade (DB filter).")
    ap.add_argument("--compact", action="store_true", help="Compact output (hide long attribution fields).")
    ap.add_argument("--fetch", type=int, default=None,
                    help="Print only the file path for the Nth result (1-based). Useful for scripting.")

    args = ap.parse_args()

    col = get_collection(args.db, args.collection)

    where = build_where(args.subject, args.grade, args.strict_grade)

    raw_k = args.n if args.strict_grade else max(args.n * 5, args.n)

    res = col.query(
        query_texts=[args.query],
        n_results=raw_k,
        where=where,
        include=["metadatas", "distances", "uris"],
    )

    ids = (res.get("ids") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    mds = (res.get("metadatas") or [[]])[0]
    uris = (res.get("uris") or [[]])[0]

    if not ids:
        print("No results found.")
        return

    ranked: List[Tuple[float, float, str, Dict[str, Any]]] = []

    for dist, uri, md in zip(dists, uris, mds):
        gmin = to_int(md.get("grade_min"), None)
        gmax = to_int(md.get("grade_max"), None)

        pen = 0.0 if args.strict_grade else grade_penalty(args.grade, gmin, gmax)
        score = combine_score(float(dist), pen)

        ranked.append((score, float(dist), uri, md))

    ranked.sort(key=lambda x: x[0], reverse=True)
    ranked = ranked[: args.n]

    if args.fetch is not None:
        i = args.fetch
        if i < 1 or i > len(ranked):
            raise SystemExit(f"--fetch must be between 1 and {len(ranked)}")
        print(ranked[i - 1][2])
        return

    for i, (score, dist, uri, md) in enumerate(ranked, start=1):
        print_result(i, score, dist, uri, md, compact=args.compact)


if _name_ == "_main_":
    main()
