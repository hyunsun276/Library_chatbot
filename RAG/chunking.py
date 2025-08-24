# stage1_chunk_and_embed_langchain.py
# 사용법:
#   OPENAI_API_KEY=... python stage1_chunk_and_embed_langchain.py
# 보고서에 맞춘 LangChain 버전: RecursiveCharacterTextSplitter + OpenAIEmbeddings 사용

import os, glob, json
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ---------- Config ----------
DATA_DIR      = os.getenv("DATA_DIR", "RAG/.data")
ARTIFACT_DIR  = os.getenv("ARTIFACT_DIR", "RAG/.artifacts")
MAX_CHARS     = int(os.getenv("MAX_CHARS", "1200"))
OVERLAP       = int(os.getenv("OVERLAP", "200"))   # 보고서대로 200자 overlap
EMB_MODEL     = os.getenv("EMB_MODEL", "text-embedding-3-small")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------- Helpers ----------
def find_files(pattern: str) -> List[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, pattern)))

def read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

# ---------- Main ----------
def main():
    # JSONL 로드
    files = glob.glob(os.path.join(DATA_DIR, "*.jsonl"))
    if not files:
        raise SystemExit(f"[ERROR] {DATA_DIR} 안에 JSONL 파일이 없습니다.")

    docs = []
    for p in files:
        for row in read_jsonl(p):
            text = row.get("text") or row.get("scene_full_text") or row.get("chapter_full_text") or row.get("full_bio") or ""
            if not text.strip():
                continue
            meta: Dict[str, Any] = {k:v for k,v in row.items() if k not in ["text","scene_full_text","chapter_full_text","full_bio"]}
            docs.append({"text": text, "metadata": meta})

    if not docs:
        raise SystemExit("[ERROR] JSONL에서 유효한 텍스트를 찾을 수 없습니다.")

    # ---------- LangChain TextSplitter ----------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS,
        chunk_overlap=OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for d in docs:
        splits = splitter.split_text(d["text"])
        for i, chunk in enumerate(splits):
            chunks.append({
                "id": f"{d['metadata'].get('work_id','unknown')}::{d['metadata'].get('kind','raw')}::{i}",
                "text": chunk,
                "metadata": d["metadata"]
            })

    print(f"[LOAD+SPLIT] docs={len(docs)} → chunks={len(chunks)}")

    # ---------- LangChain Embeddings ----------
    embeddings = OpenAIEmbeddings(model=EMB_MODEL)
    texts = [c["text"] for c in chunks]
    vecs = embeddings.embed_documents(texts)
    dim  = len(vecs[0]) if vecs else 0
    print(f"[EMBED] vectors={len(vecs)}, dim={dim}, model={EMB_MODEL}")

    # ---------- Save ----------
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    chunks_path = os.path.join(ARTIFACT_DIR, f"chunks_{stamp}.jsonl")
    embs_path   = os.path.join(ARTIFACT_DIR, f"embeddings_{stamp}.jsonl")

    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(embs_path, "w", encoding="utf-8") as f:
        for c, v in zip(chunks, vecs):
            f.write(json.dumps({"id": c["id"], "embedding": v}, ensure_ascii=False) + "\n")

    print(f"[SAVE] chunks     → {chunks_path}")
    print(f"[SAVE] embeddings → {embs_path}")
    print("[TIP] 이후 단계: Chroma.from_documents()로 DB 구축 가능")

if __name__ == "__main__":
    main()
