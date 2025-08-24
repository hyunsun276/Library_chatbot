# app.py
# RAG 검색 + 페르소나 주입 + 답변 생성 (Streamlit Cloud 호환)

# -------- SQLite 패치 (Streamlit Cloud용) --------
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# -------- 기본 라이브러리 --------
import os, re, glob, json
import streamlit as st
import chromadb
from rank_bm25 import BM25Okapi
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
oa = OpenAI()

# -------- 기본 설정 --------
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR   = os.getenv("PERSIST_DIR", os.path.join(BASE_DIR, "..", "rag", ".chroma"))
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

# -------- Chroma 로드 --------
client = chromadb.PersistentClient(path=PERSIST_DIR)
col = client.get_or_create_collection(name=COLLECTION, embedding_function=None)

results = col.get(include=["documents","metadatas"], limit=999999)
all_ids, all_docs, all_metas = results["ids"], results["documents"], results["metadatas"]

bm25 = BM25Okapi([
    [tok for tok in re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", d.lower()).split()]
    for d in all_docs
])
id2doc = {i: (t, m) for i, t, m in zip(all_ids, all_docs, all_metas)}

# -------- 검색 함수 --------
def reciprocal_rank_fusion(results_lists, k=60):
    scores = {}
    for res in results_lists:
        for rank, doc_id in enumerate(res, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0/(k+rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_retrieve(query, top_k, work_id=None):
    if not query.strip():
        return []
    emb = oa.embeddings.create(model=EMB_MODEL, input=query).data[0].embedding
    vec_res = col.query(query_embeddings=[emb], n_results=top_k*3,
                        where={"work_id": work_id} if work_id else None)
    vec_ids = vec_res["ids"][0] if vec_res["ids"] else []

    tokens = [tok for tok in re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", query.lower()).split()]
    scores = bm25.get_scores(tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k*3]
    bm25_ids = [
        all_ids[i] for i, _ in ranked
        if (not work_id or all_metas[i].get("work_id") == work_id)
    ]

    fused = reciprocal_rank_fusion([vec_ids, bm25_ids])
    hits = []
    for did, _ in fused:
        txt, meta = id2doc[did]
        if meta.get("spoiler_level", 3) <= SPOILER_LEVEL:
            hits.append((did, txt, meta))
        if len(hits) >= top_k:
            break
    return hits

# -------- 프롬프트 생성 --------
def make_prompt(query, hits, work_id=None, speak_as=None, history=[]):
    persona_block = ""
    if speak_as and work_id:
        persos = [
            txt for i, (txt, meta) in id2doc.items()
            if meta.get("work_id") == work_id
            and meta.get("kind") in ["persona", "characters_raw"]
            and (speak_as in meta.get("character", ""))
        ]
        if persos:
            persona_block = f"[인물 페르소나: {speak_as}]\n{persos[0]}"

    context_cards = []
    for _, txt, meta in hits:
        title = meta.get("scene_title") or meta.get("chapter_label") or meta.get("kind")
        context_cards.append(f"### {title}\n{txt}")

    system = (
        "당신은 소설 속 인물의 말투를 재현하는 AI입니다.\n"
        "컨텍스트를 근거로만 답하세요.\n"
        "대화할 때는 해당 인물의 말투/가치관을 반영해 1~2문장 이내로 자연스럽게 대답하세요."
    )
    if persona_block: system += "\n\n" + persona_block

    msgs = [{"role": "system", "content": system}]
    if history: msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": f"질문: {query}\n\n[컨텍스트]\n" + "\n\n".join(context_cards)})
    return msgs

# -------- 답변 생성 --------
def generate(messages):
    resp = oa.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content.strip()

# -------- Streamlit UI --------
st.set_page_config(page_title="📚 소설 캐릭터 챗봇", layout="centered")

st.markdown("""
<style>
html, body, .stApp { background-color: #CFE7FF !important; }
.chat-container { display: flex; flex-direction: column; padding: 20px; }
.user-message { background-color: #FFEB00; color: #000;
  padding: 10px 14px; border-radius: 18px 0 18px 18px; max-width: 70%;
  align-self: flex-end; margin: 6px 0 6px auto; }
.bot-message { background-color: #FFFFFF; color: #000;
  padding: 10px 14px; border-radius: 0 18px 18px 18px; max-width: 70%;
  align-self: flex-start; margin: 6px auto 6px 0; }
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state: st.session_state.history = []
if "work_id" not in st.session_state: st.session_state.work_id = None
if "speak_as" not in st.session_state: st.session_state.speak_as = None

st.title("📚 소설 속 인물과 대화하기")

work_kor = st.selectbox("작품 선택", ["지구 끝의 온실","종의 기원","소년이 온다"])
st.session_state.work_id = WORK_ID_MAP.get(work_kor)
st.session_state.speak_as = st.text_input("인물 선택 (예: 유진, 동호, 아영 등)", "")

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.history:
    cls = "user-message" if msg["role"]=="user" else "bot-message"
    st.markdown(f'<div class="{cls}">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

query = st.text_input("메시지를 입력하세요", key="input")
if st.button("보내기", type="primary") and query.strip():
    hits = hybrid_retrieve(query, TOP_K, st.session_state.work_id)
    msgs = make_prompt(query, hits, st.session_state.work_id, st.session_state.speak_as, st.session_state.history)
    ans = generate(msgs)
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": ans})
    st.rerun()
