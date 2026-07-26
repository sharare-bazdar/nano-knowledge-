
import streamlit as st
import fitz
import re
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="سامانه یکپارچه‌سازی دانش", page_icon="🔬", layout="wide")

def read_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    return uploaded_file.read().decode("utf-8", errors="ignore")

def chunk_document(text, source_name):
    chunks, idx, section, current = [], 0, "مقدمه", ""
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            current += "\n"
            continue
        is_header = bool(re.match(r"^\d+[\.-]\s|^[۱-۹]\.\s|^(بخش|فصل|ماده)\s", s))
        if is_header:
            if current.strip() and len(current.strip()) > 80:
                chunks.append({"id":f"{source_name}_{idx}","text":current.strip(),
                               "source":source_name,"section":section,"chunk_index":idx})
                idx += 1
            section, current = s[:60], s + "\n"
        else:
            current += s + "\n"
            if len(current) > 800:
                chunks.append({"id":f"{source_name}_{idx}","text":current.strip(),
                               "source":source_name,"section":section,"chunk_index":idx})
                idx += 1
                current = ""
    if current.strip() and len(current.strip()) > 80:
        chunks.append({"id":f"{source_name}_{idx}","text":current.strip(),
                       "source":source_name,"section":section,"chunk_index":idx})
    return chunks

@st.cache_resource
def get_collection():
    client = chromadb.EphemeralClient()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try: client.delete_collection("docs")
    except: pass
    return client.get_or_create_collection("docs",embedding_function=ef,metadata={"hnsw:space":"cosine"})

def add_chunks(col, chunks):
    if chunks:
        col.add(ids=[c["id"] for c in chunks],documents=[c["text"] for c in chunks],
                metadatas=[{"source":c["source"],"section":c["section"],"chunk_index":str(c["chunk_index"])} for c in chunks])

def find_similar(col, query, source_filter, top_k=2):
    res = col.query(query_texts=[query],n_results=top_k,where={"source":source_filter})
    if res and res["documents"] and res["documents"][0]:
        return [{"text":doc,"source":res["metadatas"][0][i]["source"]}
                for i,doc in enumerate(res["documents"][0])]
    return []

ANALYSIS_PROMPT = """دو بخش از دو سند مختلف داری:
--- سند اول ({source_a}) ---
{text_a}
--- سند دوم ({source_b}) ---
{text_b}
فقط به این فرمت جواب بده:
STATUS: [ALIGNED/CONFLICT/GAP/UNRELATED]
TOPIC: [موضوع مشترک در یک جمله فارسی]
DETAIL: [توضیح کوتاه فارسی]
RISK: [LOW/MEDIUM/HIGH]"""

def analyze_pair(client, text_a, text_b, source_a, source_b):
    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":ANALYSIS_PROMPT.format(
                source_a=source_a,text_a=text_a[:600],
                source_b=source_b,text_b=text_b[:600])}],
            max_tokens=300,temperature=0.1)
        raw = res.choices[0].message.content.strip()
        result = {"status":"UNKNOWN","topic":"","detail":"","risk":"LOW",
                  "text_a":text_a[:200],"text_b":text_b[:200],"source_a":source_a,"source_b":source_b}
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("STATUS:"):
                s = line.replace("STATUS:","").strip()
                if s in ["ALIGNED","CONFLICT","GAP","UNRELATED"]: result["status"] = s
            elif line.startswith("TOPIC:"): result["topic"] = line.replace("TOPIC:","").strip()
            elif line.startswith("DETAIL:"): result["detail"] = line.replace("DETAIL:","").strip()
            elif line.startswith("RISK:"):
                r = line.replace("RISK:","").strip()
                if r in ["LOW","MEDIUM","HIGH"]: result["risk"] = r
        return result
    except Exception as e:
        return {"status":"ERROR","topic":"خطا","detail":str(e),"risk":"UNKNOWN",
                "text_a":text_a[:200],"text_b":text_b[:200],"source_a":source_a,"source_b":source_b}

# UI
st.title("🔬 سامانه یکپارچه‌سازی دانش سازمانی")
st.caption("Multi-Source Knowledge Integration System | فاز پایلوت")
st.divider()

with st.sidebar:
    st.header("⚙️ تنظیمات")
    api_key = st.text_input("Groq API Key", type="password")
    st.divider()
    st.subheader("📄 آپلود اسناد")
    name_a = st.text_input("نام سند اول", value="SOP")
    name_b = st.text_input("نام سند دوم", value="مقررات")
    file_a = st.file_uploader("سند اول (PDF یا TXT)", type=["pdf","txt"])
    file_b = st.file_uploader("سند دوم (PDF یا TXT)", type=["pdf","txt"])
    max_pairs = st.slider("تعداد chunk", 5, 30, 15)
    run_btn = st.button("▶ شروع تحلیل", type="primary", use_container_width=True)

tab1, tab2 = st.tabs(["📊 تحلیل تضاد", "💬 چت‌بات"])

with tab1:
    if not file_a or not file_b:
        st.info("👆 دو فایل رو از سایدبار آپلود کن")
        st.stop()

    if run_btn:
        if not api_key:
            st.error("❌ Groq API Key رو وارد کن")
            st.stop()

        groq_client = Groq(api_key=api_key)

        with st.status("در حال پردازش...", expanded=True) as status:
            text_a = read_file(file_a)
            text_b = read_file(file_b)
            chunks_a = chunk_document(text_a, name_a)
            chunks_b = chunk_document(text_b, name_b)
            st.write(f"✅ {name_a}: {len(chunks_a)} chunk | {name_b}: {len(chunks_b)} chunk")
            col = get_collection()
            add_chunks(col, chunks_a)
            add_chunks(col, chunks_b)
            status.update(label="✅ آماده", state="complete")

        st.session_state["col"] = col
        st.session_state["chunks_a"] = chunks_a
        st.session_state["name_a"] = name_a
        st.session_state["name_b"] = name_b
        st.session_state["groq_client"] = groq_client

        results, seen = [], set()
        bar = st.progress(0, text="در حال تحلیل...")
        for i, chunk in enumerate(chunks_a[:max_pairs]):
            bar.progress((i+1)/max_pairs, text=f"chunk {i+1}/{max_pairs}")
            for sim in find_similar(col, chunk["text"], name_b):
                key = (chunk["text"][:50], sim["text"][:50])
                if key in seen: continue
                seen.add(key)
                r = analyze_pair(groq_client, chunk["text"], sim["text"], name_a, name_b)
                if r["status"] != "UNRELATED": results.append(r)
        bar.empty()
        st.session_state["results"] = results

    if "results" in st.session_state:
        results = st.session_state["results"]
        STATUS_FA = {"CONFLICT":"❌ تضاد","GAP":"⚠️ شکاف","ALIGNED":"✅ هم‌راستا","UNKNOWN":"❓","ERROR":"🔴 خطا"}
        RISK_FA = {"HIGH":"🔴 بالا","MEDIUM":"🟡 متوسط","LOW":"🟢 پایین","UNKNOWN":"❓"}

        c1,c2,c3 = st.columns(3)
        c1.metric("❌ تضاد", sum(1 for r in results if r["status"]=="CONFLICT"))
        c2.metric("⚠️ شکاف", sum(1 for r in results if r["status"]=="GAP"))
        c3.metric("✅ هم‌راستا", sum(1 for r in results if r["status"]=="ALIGNED"))
        st.divider()

        for r in results:
            if r["status"] in ["CONFLICT","GAP","ALIGNED"]:
                with st.expander(f"{STATUS_FA.get(r['status'])} | ریسک: {RISK_FA.get(r['risk'])} | {r['topic']}",
                                 expanded=(r["risk"]=="HIGH")):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**📄 {r['source_a']}**")
                        st.info(r["text_a"])
                    with col_b:
                        st.markdown(f"**📄 {r['source_b']}**")
                        st.info(r["text_b"])
                    st.warning(f"💬 {r['detail']}")
                    st.caption("⚠️ نیاز به بررسی کارشناس انسانی دارد")

        import pandas as pd
        if results:
            df = pd.DataFrame([{"وضعیت":STATUS_FA.get(r["status"]),"موضوع":r["topic"],
                "ریسک":RISK_FA.get(r["risk"]),"توضیح":r["detail"],
                f"متن {r['source_a']}":r["text_a"],f"متن {r['source_b']}":r["text_b"]} for r in results])
            st.download_button("⬇️ دانلود گزارش CSV",df.to_csv(index=False).encode("utf-8-sig"),"report.csv","text/csv")

with tab2:
    if "col" not in st.session_state:
        st.info("👆 اول تحلیل رو اجرا کن")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("سوالت رو بنویس..."):
        st.session_state["chat_history"].append({"role":"user","content":question})
        with st.chat_message("user"):
            st.write(question)
        col = st.session_state["col"]
        na = st.session_state["name_a"]
        nb = st.session_state["name_b"]
        gc = st.session_state["groq_client"]
        sim_a = find_similar(col, question, na, top_k=2)
        sim_b = find_similar(col, question, nb, top_k=2)
        context = "\n".join([f"[{na}]: {c['text'][:400]}" for c in sim_a]+[f"[{nb}]: {c['text'][:400]}" for c in sim_b])
        with st.chat_message("assistant"):
            with st.spinner("در حال پاسخ..."):
                try:
                    res = gc.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role":"user","content":f"بر اساس این اسناد جواب بده:\n{context}\n\nسوال: {question}\nجواب فارسی:"}],
                        max_tokens=400,temperature=0.3)
                    answer = res.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"خطا: {e}"
                st.write(answer)
                st.session_state["chat_history"].append({"role":"assistant","content":answer})
