# app.py
# RAG 검색 + 페르소나 주입 + 답변 생성

import os, re
import chromadb
import streamlit as st
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv
from chunking import main as build_chunks
from DB_MAKING import main as build_db

# 앱 시작 시 DB 초기화
build_chunks()
build_db()

# ----------------- 환경변수 -----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
oa = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR   = os.getenv("PERSIST_DIR", os.path.join(BASE_DIR, "RAG", ".chroma"))
COLLECTION    = os.getenv("COLLECTION", "library-all")
MODEL         = os.getenv("MODEL", "gpt-4o")
TOP_K         = int(os.getenv("TOP_K", "6"))
SPOILER_LEVEL = int(os.getenv("SPOILER_LEVEL", "3"))
EMB_MODEL     = "text-embedding-3-small"

WORK_ID_MAP = {
    "지구 끝의 온실": "jigu-ggut-onshil",
    "종의 기원": "jong-ui-giwon",
    "소년이 온다": "so-nyeon-i-onda"
}

# ----------------- DB 체크 & 생성 -----------------
def ensure_chroma_db():
    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        print("[INIT] No Chroma DB found. Building new DB...")
        from stage1_chunk_and_embed import main as chunk_and_embed
        from DB_MAKING import main as make_db
        chunk_and_embed()
        make_db()

ensure_chroma_db()

# ----------------- Chroma 로드 -----------------
PERSIST_DIR = tempfile.mkdtemp()   # 실행할 때마다 임시 폴더 생성
client = chromadb.PersistentClient(path=PERSIST_DIR)
col = client.get_or_create_collection(name=COLLECTION, embedding_function=None)

results = col.get(include=["documents","metadatas"], limit=999999)
all_ids, all_docs, all_metas = results["ids"], results["documents"], results["metadatas"]

bm25 = BM25Okapi([
    [tok for tok in re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", d.lower()).split()]
    for d in all_docs
])
id2doc = {i: (t, m) for i, t, m in zip(all_ids, all_docs, all_metas)}

# ----------------- 검색 함수 -----------------
def reciprocal_rank_fusion(results_lists, k=60):
    scores = {}
    for res in results_lists:
        for rank, doc_id in enumerate(res, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0/(k+rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retrieve(query, top_k, work_id=None):
    if not query or not query.strip():
        return []
    emb = oa.embeddings.create(model=EMB_MODEL, input=query).data[0].embedding
    vec_res = col.query(query_embeddings=[emb], n_results=top_k*3,
                        where={"work_id": work_id} if work_id else None)
    vec_ids = vec_res["ids"][0] if vec_res["ids"] else []

    tokens = [tok for tok in re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", query.lower()).split()]
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k*3]
    bm25_ids = [all_ids[i] for i,_ in ranked if (not work_id or all_metas[i].get("work_id")==work_id)]

    fused = reciprocal_rank_fusion([vec_ids, bm25_ids])
    hits = []
    for did,_ in fused:
        txt, meta = id2doc[did]
        if meta.get("spoiler_level",3) <= SPOILER_LEVEL:
            hits.append((did, txt, meta))
        if len(hits)>=top_k: break
    return hits

# ----------------- 프롬프트 -----------------
def make_prompt(query, hits, work_id=None, speak_as=None, history=[]):
    persona_block = ""
    if speak_as and work_id:
        persos = [txt for i,(txt,meta) in id2doc.items()
                  if meta.get("work_id")==work_id
                  and meta.get("kind") in ["persona","characters_raw"]
                  and (speak_as in meta.get("character",""))]
        if persos:
            persona_block = f"[인물 페르소나: {speak_as}]\n{persos[0]}"

    context_cards = []
    for _,txt,meta in hits:
        title = meta.get("scene_title") or meta.get("chapter_label") or meta.get("kind")
        context_cards.append(f"### {title}\n{txt}")

    system = (
        "당신은 소설 속 인물의 말투를 재현하는 AI입니다.\n"
        "컨텍스트를 근거로 사용하세요.\n"
        "소설 속 인물로 대화하듯 답변하세요 (1~2문장).\n"
    )
    if persona_block:
        system += "\n\n" + persona_block

    msgs = [{"role":"system","content":system}]
    if history: msgs.extend(history[-6:])
    user = f"질문: {query}\n\n[컨텍스트]\n" + "\n\n".join(context_cards[:8])
    msgs.append({"role":"user","content":user})
    return msgs

# ----------------- 답변 생성 -----------------
def generate(messages):
    try:
        resp = oa.responses.create(model=MODEL, input=messages)
        return getattr(resp,"output_text","").strip()
    except Exception:
        comp = oa.chat.completions.create(model=MODEL, messages=messages)
        return comp.choices[0].message.content.strip()

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="📚 소설 캐릭터 챗봇", layout="centered")

st.title("📚 소설 속 인물과 대화하기")

# 작품/인물 선택
work_kor = st.selectbox("작품 선택", ["지구 끝의 온실","종의 기원","소년이 온다"])
st.session_state.work_id = WORK_ID_MAP.get(work_kor)
st.session_state.speak_as = st.text_input("인물 선택 (예: 유진, 동호, 아영 등)", "")

# 채팅 UI
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    role_class = "user-message" if msg["role"]=="user" else "bot-message"
    st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

query = st.text_input("메시지를 입력하세요", key="input")
if st.button("보내기", type="primary") and query.strip():
    hits = hybrid_retrieve(query, TOP_K, st.session_state.work_id)
    msgs = make_prompt(query, hits, work_id=st.session_state.work_id,
                       speak_as=st.session_state.speak_as,
                       history=st.session_state.history)
    ans = generate(msgs)

    st.session_state.history.append({"role":"user","content":query})
    st.session_state.history.append({"role":"assistant","content":ans})
    st.rerun()
