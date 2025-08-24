# stage2_build_chroma.py
# JSONL (chunks) → Chroma DB 업서트 (LangChain 버전)
# 보고서에 맞춘 구조: LangChain Chroma + OpenAIEmbeddings

import os, glob, json, uuid
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ---------------- Config ----------------
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "rag/.artifacts").replace("\\","/")
PERSIST_DIR  = os.getenv("PERSIST_DIR", "rag/.chroma").replace("\\","/")
COLLECTION   = os.getenv("COLLECTION", "library-all")

CHUNKS_PATH  = os.getenv("CHUNKS_PATH", "")

FILTER_WORKS = [s.strip() for s in os.getenv("FILTER_WORKS","").split(",") if s.strip()]
FILTER_KINDS = [s.strip() for s in os.getenv("FILTER_KINDS","").split(",") if s.strip()]

EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-small")

# ---------------- Helpers ----------------
def latest(path_glob: str) -> str:
    files = sorted(glob.glob(path_glob), key=lambda p: os.path.getmtime(p))
    return files[-1] if files else ""

def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def kind_matches(kind: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    for pat in patterns:
        if pat.endswith("*"):
            if kind.startswith(pat[:-1]): return True
        elif kind == pat:
            return True
    return False

# ---------------- Main ----------------
def main():
    chunks_file = CHUNKS_PATH or latest(f"{ARTIFACT_DIR}/chunks_*.jsonl")
    if not chunks_file:
        raise SystemExit(f"[ERROR] chunks 파일을 찾을 수 없음: {chunks_file}")

    chunks = load_jsonl(chunks_file)

    ids, docs, metas = [], [], []
    seen = set()

    for rec in chunks:
        base_id = rec["id"]
        text = rec["text"]
        meta = rec.get("metadata", {})
        work = meta.get("work_id","unknown")
        kind = meta.get("kind","unknown")

        if FILTER_WORKS and work not in FILTER_WORKS:
            continue
        if not kind_matches(kind, FILTER_KINDS):
            continue
        if not text.strip():
            continue

        # 중복 방지
        _id = base_id
        if _id in seen:
            _id = f"{base_id}::{uuid.uuid4().hex[:6]}"
        seen.add(_id)

        ids.append(_id); docs.append(text); metas.append(meta)

    if not ids:
        raise SystemExit("[WARN] 업서트할 레코드가 없습니다.")

    # ---------------- LangChain Chroma ----------------
    embeddings = OpenAIEmbeddings(model=EMB_MODEL)

    vectordb = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR
    )

    # 업서트 (LangChain에서는 add_texts)
    vectordb.add_texts(texts=docs, metadatas=metas, ids=ids)
    vectordb.persist()

    print(f"[UPSERT] {len(ids)} items → collection='{COLLECTION}' @ {PERSIST_DIR}")

    # ---------------- Test Query ----------------
    q = "기억과 상실의 주제"
    results = vectordb.similarity_search(q, k=3)
    print("[CHECK] sample query:", q)
    for r in results:
        print(" -", r.page_content[:80], "...", r.metadata)

if __name__ == "__main__":
    main()
