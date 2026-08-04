import streamlit as st
import fitz
import re
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json, os

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
HISTORY_FILE = "history.json"

# رنگ‌های NanoDaru
PRIMARY = "#003f7f"
ACCENT = "#e91e8c"
BG = "#f8f9fc"

st.set_page_config(
    page_title="NanoDaru | سامانه دانش",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown(f"""
<style>
    * {{ font-family: Tahoma, sans-serif; }}
    body {{ direction: rtl; background: {BG}; }}
    .stApp {{ background: {BG}; direction: rtl; }}

    [data-testid="stSidebar"] {{
        background: {PRIMARY};
        direction: rtl;
    }}

    /* رنگ همه متن‌های سایدبار */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stMarkdown {{
        color: white !important;
    }}

    /* input های سایدبار */
    [data-testid="stSidebar"] input {{
        background: white !important;
        color: {PRIMARY} !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: Tahoma !important;
    }}

    /* placeholder */
    [data-testid="stSidebar"] input::placeholder {{
        color: #aaa !important;
    }}

    /* file uploader */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {{
        background: rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        padding: 8px !important;
        border: 1px dashed rgba(255,255,255,0.4) !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] * {{
        color: white !important;
    }}
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {{
        background: white !important;
        color: {PRIMARY} !important;
        border-radius: 6px !important;
    }}

    /* دکمه */
    .stButton > button {{
        background: {ACCENT} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 15px !important;
        padding: 10px !important;
    }}
    .stButton > button:hover {{ opacity: 0.85; }}

    /* هدر */
    .nano-header {{
        background: linear-gradient(135deg, {PRIMARY}, #0066cc);
        padding: 25px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }}
    .nano-header h1 {{ font-size: 24px; margin: 0; }}
    .nano-header p {{ color: rgba(255,255,255,0.7); margin: 5px 0 0; font-size: 13px; }}

    /* کارت آمار */
    .stat-card {{
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin: 5px 0;
    }}
    .stat-card .num {{ font-size: 36px; font-weight: bold; }}
    .stat-card .lbl {{ font-size: 13px; opacity: 0.9; }}

    /* کارت نتیجه */
    .result-card {{
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-right: 4px solid #ddd;
    }}
    .result-card.CONFLICT {{ border-color: #e53935; }}
    .result-card.GAP {{ border-color: #fb8c00; }}
    .result-card.ALIGNED {{ border-color: #43a047; }}

    .stTabs [aria-selected="true"] {{
        color: {PRIMARY} !important;
        border-bottom-color: {ACCENT} !important;
    }}

    #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)




def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    return uploaded_file.read().decode("utf-8", errors="ignore")

def chunk_document(text, source_name):
    chunks, idx, section, current = [], 0, "مقدمه", ""
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            current += "\n"
            continue
        is_header = bool(re.match(
            r'^\d+[\.\-]\s|^[۱-۹]\.\s|^(بخش|فصل|ماده|تبصره|Article|Section)\s|^\d+\.\d+', s))
        if is_header:
            if current.strip() and len(current.strip()) > 80:
                chunks.append({"id": f"{source_name}_{idx}", "text": current.strip(),
                               "source": source_name, "section": section, "chunk_index": idx})
                idx += 1
            section, current = s[:60], s + "\n"
        else:
            current += s + "\n"
            if len(current) > 800:
                chunks.append({"id": f"{source_name}_{idx}", "text": current.strip(),
                               "source": source_name, "section": section, "chunk_index": idx})
                idx += 1
                current = ""
    if current.strip() and len(current.strip()) > 80:
        chunks.append({"id": f"{source_name}_{idx}", "text": current.strip(),
                       "source": source_name, "section": section, "chunk_index": idx})
    return chunks

def get_fresh_collection():
    client = chromadb.EphemeralClient()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    try: client.delete_collection("docs")
    except: pass
    return client.get_or_create_collection("docs", embedding_function=ef,
                                            metadata={"hnsw:space": "cosine"})

def add_chunks(col, chunks):
    if chunks:
        col.add(ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[{"source": c["source"], "section": c["section"],
                            "chunk_index": str(c["chunk_index"])} for c in chunks])

def find_similar(col, query, source_filter, top_k=2):
    try:
        res = col.query(query_texts=[query], n_results=top_k,
                        where={"source": source_filter})
        if res and res["documents"] and res["documents"][0]:
            return [{"text": doc, "source": res["metadatas"][0][i]["source"],
                     "section": res["metadatas"][0][i]["section"]}
                    for i, doc in enumerate(res["documents"][0])]
    except: pass
    return []


ANALYSIS_PROMPT = """تو متخصص تحلیل اسناد دارویی و کنترل کیفیت هستی.

--- سند اول ({source_a}) ---
{text_a}

--- سند دوم ({source_b}) ---
{text_b}

این دو بخش رو با هم مقایسه کن. به اسامی مواد، غلظت‌ها، دماها، بازه‌های زمانی و اعداد دقت کن.

فقط به این فرمت جواب بده:
STATUS: [ALIGNED/CONFLICT/GAP/UNRELATED]
TOPIC: [موضوع در یک جمله فارسی]
DETAIL: [توضیح دقیق فارسی با ذکر اعداد و مقادیر مهم]
RISK: [LOW/MEDIUM/HIGH]

راهنما:
- CONFLICT: اختلاف در اعداد، مقادیر، یا دستورالعمل‌ها
- GAP: یک سند اطلاعاتی داره که دیگری نداره
- ALIGNED: هر دو یک چیز می‌گن
- UNRELATED: ربطی به هم ندارن"""



def analyze_pair(client, text_a, text_b, source_a, source_b):
    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(
                source_a=source_a, text_a=text_a[:600],
                source_b=source_b, text_b=text_b[:600])}],
            max_tokens=300, temperature=0.1)
        raw = res.choices[0].message.content.strip()
        result = {"status": "UNKNOWN", "topic": "", "detail": "", "risk": "LOW",
                  "text_a": text_a[:250], "text_b": text_b[:250],
                  "source_a": source_a, "source_b": source_b}
        for line in raw.split('\n'):
            line = line.strip()
            if line.startswith("STATUS:"):
                s = line.replace("STATUS:", "").strip()
                if s in ["ALIGNED","CONFLICT","GAP","UNRELATED"]: result["status"] = s
            elif line.startswith("TOPIC:"): result["topic"] = line.replace("TOPIC:","").strip()
            elif line.startswith("DETAIL:"): result["detail"] = line.replace("DETAIL:","").strip()
            elif line.startswith("RISK:"):
                r = line.replace("RISK:","").strip()
                if r in ["LOW","MEDIUM","HIGH"]: result["risk"] = r
        return result
    except Exception as e:
        return {"status":"ERROR","topic":"خطا","detail":str(e),"risk":"UNKNOWN",
                "text_a":text_a[:250],"text_b":text_b[:250],
                "source_a":source_a,"source_b":source_b}

def save_history(name_a, name_b, results):
    history = load_history()
    history.append({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "doc_a": name_a, "doc_b": name_b,
        "conflicts": sum(1 for r in results if r["status"]=="CONFLICT"),
        "gaps": sum(1 for r in results if r["status"]=="GAP"),
        "aligned": sum(1 for r in results if r["status"]=="ALIGNED"),
        "results": results
    })
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except: pass
    return []


STATUS_FA = {"CONFLICT":"❌ تضاد","GAP":"⚠️ شکاف",
             "ALIGNED":"✅ هم‌راستا","UNKNOWN":"❓","ERROR":"🔴 خطا"}
RISK_FA = {"HIGH":"🔴 بالا","MEDIUM":"🟡 متوسط","LOW":"🟢 پایین","UNKNOWN":"❓"}
RISK_COLOR = {"HIGH":"#e53935","MEDIUM":"#fb8c00","LOW":"#43a047","UNKNOWN":"#999"}

# هدر
st.markdown(f"""
<div class="nano-header">
  <h1>🔬 سامانه یکپارچه‌سازی دانش سازمانی</h1>
  <p>Multi-Source Knowledge Integration System | NanoDaru Pharmaceutical</p>
</div>
""", unsafe_allow_html=True)

# سایدبار

with st.sidebar:
    st.markdown("### 🔑 API Key")
    api_key = st.text_input("Groq API Key", type="password",
                             label_visibility="collapsed",
                             placeholder="gsk_...")
    st.markdown("---")
    st.markdown("### 📄 اسناد")
    name_a = st.text_input("نام سند اول", value="SOP")
    name_b = st.text_input("نام سند دوم", value="مقررات")
    file_a = st.file_uploader("📎 سند اول (PDF/TXT)", type=["pdf","txt"])
    file_b = st.file_uploader("📎 سند دوم (PDF/TXT)", type=["pdf","txt"])
    st.markdown("---")
    run_btn = st.button("▶ شروع تحلیل", type="primary",
                        use_container_width=True)

    # تاریخچه زیر دکمه
    st.markdown("---")
    st.markdown("### 📋 تاریخچه")
    history = load_history()
    if history:
        for entry in reversed(history[-5:]):
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.1);border-radius:6px;
                        padding:8px;margin:5px 0;font-size:12px'>
              📅 {entry['date']}<br>
              {entry['doc_a']} vs {entry['doc_b']}<br>
              ❌{entry['conflicts']} ⚠️{entry['gaps']} ✅{entry['aligned']}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<small>هنوز تحلیلی نشده</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<small style='opacity:0.6'>⚠️ نیاز به بررسی کارشناس</small>",
                unsafe_allow_html=True)


# تب‌ها
tab1, tab2, tab3, tab4 = st.tabs(["📊 تحلیل تضاد", "📈 داشبورد", "💬 چت‌بات", "📋 تاریخچه"])

# ── تب ۱: تحلیل ───────────────────────────────────────────────
with tab1:
    if not file_a or not file_b:
        st.info("👆 از سایدبار دو فایل آپلود کن و API Key وارد کن")
        st.stop()

    if run_btn:
        if not api_key:
            st.error("❌ API Key وارد کن")
            st.stop()

        groq_client = Groq(api_key=api_key)

        with st.status("در حال پردازش...", expanded=True) as status:
            text_a = read_file(file_a)
            chunks_a = chunk_document(text_a, name_a)
            st.write(f"✅ {name_a}: {len(chunks_a)} بخش")
            text_b = read_file(file_b)
            chunks_b = chunk_document(text_b, name_b)
            st.write(f"✅ {name_b}: {len(chunks_b)} بخش")
            col_db = get_fresh_collection()
            add_chunks(col_db, chunks_a)
            add_chunks(col_db, chunks_b)
            st.write("✅ پایگاه دانش آماده شد")
            status.update(label="✅ آماده", state="complete")

        st.session_state.update({
            "col_db": col_db, "name_a": name_a, "name_b": name_b,
            "groq_client": groq_client, "chat_history": [],
            "chunks_a": chunks_a
        })

        results, seen = [], set()
        MAX_CHUNKS = 50
        total = min(len(chunks_a), MAX_CHUNKS)
        bar = st.progress(0)
        for i, chunk in enumerate(chunks_a[:MAX_CHUNKS]):
            bar.progress((i+1)/total, text=f"تحلیل بخش {i+1} از {total}")
            for sim in find_similar(col_db, chunk["text"], name_b):
                key = (chunk["text"][:50], sim["text"][:50])
                if key in seen: continue
                seen.add(key)
                r = analyze_pair(groq_client, chunk["text"], sim["text"], name_a, name_b)
                if r["status"] not in ["UNRELATED","UNKNOWN"]: results.append(r)
        bar.empty()

        save_history(name_a, name_b, results)
        st.session_state["results"] = results

    if "results" in st.session_state:
        results = st.session_state["results"]
        na = st.session_state.get("name_a","سند اول")
        nb = st.session_state.get("name_b","سند دوم")

        n_c = sum(1 for r in results if r["status"]=="CONFLICT")
        n_g = sum(1 for r in results if r["status"]=="GAP")
        n_a = sum(1 for r in results if r["status"]=="ALIGNED")
        n_h = sum(1 for r in results if r["risk"]=="HIGH")

        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="stat-card" style="background:#e53935"><div class="num">{n_c}</div><div class="lbl">❌ تضاد</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card" style="background:#fb8c00"><div class="num">{n_g}</div><div class="lbl">⚠️ شکاف</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card" style="background:#43a047"><div class="num">{n_a}</div><div class="lbl">✅ هم‌راستا</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="stat-card" style="background:{PRIMARY}"><div class="num">{n_h}</div><div class="lbl">🔴 ریسک بالا</div></div>', unsafe_allow_html=True)

        st.divider()
        f1,f2 = st.columns(2)
        filter_s = f1.multiselect("وضعیت", ["CONFLICT","GAP","ALIGNED"],
                                   default=["CONFLICT","GAP"])
        filter_r = f2.multiselect("ریسک", ["HIGH","MEDIUM","LOW"],
                                   default=["HIGH","MEDIUM"])
        filtered = [r for r in results if r["status"] in filter_s and r["risk"] in filter_r]
        st.caption(f"{len(filtered)} از {len(results)} مورد")

        for r in filtered:
            st.markdown(f"""
            <div class="result-card {r['status']}">
              <b>{STATUS_FA.get(r['status'])} | ریسک: {RISK_FA.get(r['risk'])} | {r['topic']}</b>
              <p style='color:#555;margin:6px 0;font-size:13px'>{r['detail']}</p>
            </div>""", unsafe_allow_html=True)
            with st.expander("نمایش متن‌ها"):
                ca,cb = st.columns(2)
                with ca:
                    st.markdown(f"**📄 {r['source_a']}**")
                    st.info(r["text_a"])
                with cb:
                    st.markdown(f"**📄 {r['source_b']}**")
                    st.info(r["text_b"])

        st.divider()
        df = pd.DataFrame([{
            "وضعیت": STATUS_FA.get(r["status"]),
            "موضوع": r["topic"],
            "ریسک": RISK_FA.get(r["risk"]),
            "توضیح": r["detail"],
            f"متن {na}": r["text_a"],
            f"متن {nb}": r["text_b"],
        } for r in results])
        st.download_button("⬇️ دانلود CSV",
                           df.to_csv(index=False).encode("utf-8-sig"),
                           f"report_{datetime.now().strftime('%Y%m%d')}.csv",
                           "text/csv")

# ── تب ۲: داشبورد Excel + نمودار ─────────────────────────────
with tab2:
    st.subheader("📈 تحلیل داده‌های Excel")

    excel_file = st.file_uploader("فایل Excel آپلود کن", type=["xlsx","xls","csv"])

    if excel_file:
        if excel_file.name.endswith(".csv"):
            df_excel = pd.read_csv(excel_file)
        else:
            df_excel = pd.read_excel(excel_file)

        st.markdown("**پیش‌نمایش داده‌ها:**")
        st.dataframe(df_excel.head(20), use_container_width=True)

        numeric_cols = df_excel.select_dtypes(include='number').columns.tolist()
        all_cols = df_excel.columns.tolist()

        if numeric_cols:
            st.divider()
            st.markdown("**ساخت نمودار:**")
            c1,c2,c3 = st.columns(3)
            chart_type = c1.selectbox("نوع نمودار",
                ["ستونی","خطی","پراکندگی","دایره‌ای","هیستوگرام"])
            x_col = c2.selectbox("محور X", all_cols)
            y_col = c3.selectbox("محور Y", numeric_cols)

            color_col = st.selectbox("رنگ‌بندی بر اساس (اختیاری)",
                                      ["ندارد"] + all_cols)
            color = None if color_col == "ندارد" else color_col

            if chart_type == "ستونی":
                fig = px.bar(df_excel, x=x_col, y=y_col, color=color,
                             color_discrete_sequence=[PRIMARY, ACCENT])
            elif chart_type == "خطی":
                fig = px.line(df_excel, x=x_col, y=y_col, color=color,
                              color_discrete_sequence=[PRIMARY, ACCENT])
            elif chart_type == "پراکندگی":
                fig = px.scatter(df_excel, x=x_col, y=y_col, color=color,
                                 color_discrete_sequence=[PRIMARY, ACCENT])
            elif chart_type == "دایره‌ای":
                fig = px.pie(df_excel, names=x_col, values=y_col,
                             color_discrete_sequence=[PRIMARY, ACCENT, "#0099cc","#ff69b4"])
            else:
                fig = px.histogram(df_excel, x=y_col,
                                   color_discrete_sequence=[PRIMARY])

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_family="Tahoma",
                title_font_color=PRIMARY
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**آمار پایه:**")
            st.dataframe(df_excel[numeric_cols].describe(), use_container_width=True)
        else:
            st.warning("فایل شما ستون عددی ندارد")
    else:
        st.info("یه فایل Excel یا CSV آپلود کن تا نمودار بسازیم")

# ── تب ۳: چت‌بات ──────────────────────────────────────────────
with tab3:
    if "col_db" not in st.session_state:
        st.info("👆 اول از تب تحلیل، فایل‌ها رو آپلود و تحلیل کن")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("از اسناد سوال بپرس..."):
        st.session_state["chat_history"].append({"role":"user","content":question})
        with st.chat_message("user"):
            st.write(question)

        col_db = st.session_state["col_db"]
        na = st.session_state["name_a"]
        nb = st.session_state["name_b"]
        gc = st.session_state["groq_client"]

        sim_a = find_similar(col_db, question, na, top_k=2)
        sim_b = find_similar(col_db, question, nb, top_k=2)
        context = "\n".join(
            [f"[{na}]: {c['text'][:400]}" for c in sim_a] +
            [f"[{nb}]: {c['text'][:400]}" for c in sim_b])

        with st.chat_message("assistant"):
            with st.spinner("..."):
                try:
                    res = gc.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role":"user","content":
                            f"بر اساس این اسناد جواب بده:\n{context}\n\nسوال: {question}\nجواب فارسی:"}],
                        max_tokens=500, temperature=0.3)
                    answer = res.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"خطا: {e}"
                st.write(answer)
                st.session_state["chat_history"].append({"role":"assistant","content":answer})

# ── تب ۴: تاریخچه ─────────────────────────────────────────────
with tab4:
    st.subheader("📋 تاریخچه تحلیل‌ها")
    history = load_history()

    if not history:
        st.info("هنوز تحلیلی انجام نشده")
        st.stop()

    for entry in reversed(history):
        with st.expander(
            f"📅 {entry['date']} | {entry['doc_a']} vs {entry['doc_b']}"):
            c1,c2,c3 = st.columns(3)
            c1.metric("تضاد", entry["conflicts"])
            c2.metric("شکاف", entry["gaps"])
            c3.metric("هم‌راستا", entry["aligned"])




                
