"""
search_k12_db.py (UPDATED)
- Searches your Chroma K–12 image database
- Soft rerank by grade + optional subject filter
- Cleans HTML from Wikimedia attribution fields when printing
"""

from __future__ import annotations

import argparse
import re
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction


DB_PATH = "./image_db"
COLLECTION_NAME = "k12_education_images"


_TAG_RE = re.compile(r"<[^>]+>")

def clean_html(text: Any, max_len: int = 220) -> str:
    if text is None:
        return ""
    s = str(text)
    s = _TAG_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len] + "…"
    return s


def _grade_score(meta: Dict[str, Any], grade: Optional[int]) -> float:
    if grade is None:
        return 0.0
    try:
        gmin = int(meta.get("grade_min", 0))
        gmax = int(meta.get("grade_max", 12))
    except Exception:
        return 0.0

    if gmin <= grade <= gmax:
        center = (gmin + gmax) / 2.0
        return 2.0 - abs(grade - center) * 0.05
    dist = min(abs(grade - gmin), abs(grade - gmax))
    return -0.25 * dist


def _subject_match_score(meta: Dict[str, Any], subject: Optional[str]) -> float:
    if not subject:
        return 0.0
    ms = str(meta.get("subject", "")).strip().lower()
    return 0.75 if ms == subject.strip().lower() else -0.10


def _rerank(
    ids: List[str],
    uris: List[str],
    metas: List[Dict[str, Any]],
    distances: List[float],
    grade: Optional[int],
    subject: Optional[str],
) -> List[Dict[str, Any]]:
    out = []
    for _id, uri, meta, dist in zip(ids, uris, metas, distances):
        sim = -float(dist)  # lower distance -> higher sim
        score = sim + _grade_score(meta, grade) + _subject_match_score(meta, subject)
        out.append(
            {
                "id": _id,
                "uri": uri,
                "distance": float(dist),
                "score": float(score),
                "metadata": meta,
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def search_images(
    query: str,
    n: int = 8,
    grade: Optional[int] = None,
    subject: Optional[str] = None,
) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OpenCLIPEmbeddingFunction(),
    )

    fetch_n = max(n * 4, 20)

    res = collection.query(
        query_texts=[query],
        n_results=fetch_n,
        where={"subject": subject} if subject else None,
        include=["metadatas", "distances", "uris"],
    )

    ids = (res.get("ids") or [[]])[0]
    uris = (res.get("uris") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    if not ids:
        return []

    ranked = _rerank(ids, uris, metas, dists, grade=grade, subject=subject)
    return ranked[:n]


def print_results(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No matches found.")
        return

    for i, r in enumerate(results, start=1):
        m = r["metadata"] or {}

        artist = clean_html(m.get("artist"))
        credit = clean_html(m.get("credit"))

        print(f"\n#{i}  score={r['score']:.3f}  dist={r['distance']:.3f}")
        print("File:", r["uri"])
        print("Subject:", m.get("subject", ""))
        print("Topic:", m.get("topic", ""))
        print("Grades:", m.get("grade_min", ""), "-", m.get("grade_max", ""))
        print("License:", m.get("license", ""))

        if artist:
            print("Artist:", artist)
        if credit:
            print("Credit:", credit)

        if m.get("source_page"):
            print("Source page:", m.get("source_page"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Search text, e.g. 'water cycle diagram'")
    ap.add_argument("--n", type=int, default=8, help="Number of results")
    ap.add_argument("--grade", type=int, default=None, help="Grade level (K=0). Example: --grade 5")
    ap.add_argument("--subject", type=str, default=None, help="Optional subject filter (e.g. Math, Biology)")
    args = ap.parse_args()

    results = search_images(
        query=args.query,
        n=args.n,
        grade=args.grade,
        subject=args.subject,
    )
    print_results(results)


if __name__ == "__main__":
    main()