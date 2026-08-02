import streamlit as st
import fitz
import re
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import pandas as pd
from datetime import datetime
import json
import os

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
HISTORY_FILE = "analysis_history.json"

st.set_page_config(
    page_title="سامانه یکپارچه‌سازی دانش",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
body { direction: rtl; font-family: Tahoma; }
.stApp { direction: rtl; }
[data-testid="stSidebar"] { direction: rtl; }
</style>
""", unsafe_allow_html=True)


def read_file(uploaded_file):
    """PDF یا TXT رو می‌خونه و متن برمیگردونه"""
    if uploaded_file.name.lower().endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    return uploaded_file.read().decode("utf-8", errors="ignore")

def chunk_document(text, source_name):
    """
    متن رو به بخش‌های معنادار تقسیم می‌کنه
    اول سعی می‌کنه بر اساس عنوان‌ها تقسیم کنه
    اگه بخش خیلی بزرگ شد، بر اساس اندازه تقسیم می‌کنه
    """
    chunks, idx, section, current = [], 0, "مقدمه", ""
    for line in text.split('\n'):
        s = line.strip()
        if not s:
            current += "\n"
            continue
        is_header = bool(re.match(
            r'^\d+[\.\-]\s|^[۱-۹]\.\s|^(بخش|فصل|ماده|تبصره|Article|Section)\s|^\d+\.\d+',
            s))
        if is_header:
            if current.strip() and len(current.strip()) > 80:
                chunks.append({
                    "id": f"{source_name}_{idx}",
                    "text": current.strip(),
                    "source": source_name,
                    "section": section,
                    "chunk_index": idx
                })
                idx += 1
            section = s[:60]
            current = s + "\n"
        else:
            current += s + "\n"
            if len(current) > 800:
                chunks.append({
                    "id": f"{source_name}_{idx}",
                    "text": current.strip(),
                    "source": source_name,
                    "section": section,
                    "chunk_index": idx
                })
                idx += 1
                current = ""
    if current.strip() and len(current.strip()) > 80:
        chunks.append({
            "id": f"{source_name}_{idx}",
            "text": current.strip(),
            "source": source_name,
            "section": section,
            "chunk_index": idx
        })
    return chunks


def get_fresh_collection():
    """
    هر بار یه collection تازه می‌سازه
    مشکل cache قبلی رو حل می‌کنه
    """
    client = chromadb.EphemeralClient()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL)
    try:
        client.delete_collection("docs")
    except:
        pass
    return client.get_or_create_collection(
        "docs",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

def add_chunks(col, chunks):
    """chunk ها رو به پایگاه دانش اضافه می‌کنه"""
    if not chunks:
        return
    col.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{
            "source": c["source"],
            "section": c["section"],
            "chunk_index": str(c["chunk_index"])
        } for c in chunks]
    )

def find_similar(col, query, source_filter, top_k=3):
    """
    برای یه متن، مشابه‌ترین بخش‌های سند دیگه رو پیدا می‌کنه
    از embedding های معنایی استفاده می‌کنه
    """
    try:
        res = col.query(
            query_texts=[query],
            n_results=top_k,
            where={"source": source_filter}
        )
        if res and res["documents"] and res["documents"][0]:
            return [{
                "text": doc,
                "source": res["metadatas"][0][i]["source"],
                "section": res["metadatas"][0][i]["section"],
                "distance": res["distances"][0][i] if res.get("distances") else 0
            } for i, doc in enumerate(res["documents"][0])]
    except:
        pass
    return []



ANALYSIS_PROMPT = """تو یک متخصص تحلیل اسناد دارویی و سازمانی هستی.
دو بخش از دو سند مختلف داری:

--- سند اول ({source_a}) ---
{text_a}

--- سند دوم ({source_b}) ---
{text_b}

رابطه این دو بخش رو تحلیل کن.
فقط و فقط به این فرمت جواب بده، هیچ چیز اضافه ننویس:

STATUS: [ALIGNED/CONFLICT/GAP/UNRELATED]
TOPIC: [موضوع مشترک در یک جمله فارسی]
DETAIL: [توضیح دقیق فارسی - حداکثر 2 جمله]
RISK: [LOW/MEDIUM/HIGH]

راهنما:
- ALIGNED: هر دو سند یک چیز می‌گن
- CONFLICT: دو سند در یک موضوع حرف‌های متناقض می‌زنن
- GAP: یک سند اطلاعاتی داره که سند دیگه نداره
- UNRELATED: این دو بخش ربطی به هم ندارن"""

def analyze_pair(client, text_a, text_b, source_a, source_b):
    """
    یه جفت chunk رو با LLM تحلیل می‌کنه
    خروجی: وضعیت، موضوع، توضیح، سطح ریسک
    """
    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(
                source_a=source_a, text_a=text_a[:600],
                source_b=source_b, text_b=text_b[:600]
            )}],
            max_tokens=300,
            temperature=0.1
        )
        raw = res.choices[0].message.content.strip()
        result = {
            "status": "UNKNOWN", "topic": "", "detail": "", "risk": "LOW",
            "text_a": text_a[:250], "text_b": text_b[:250],
            "source_a": source_a, "source_b": source_b
        }
        for line in raw.split('\n'):
            line = line.strip()
            if line.startswith("STATUS:"):
                s = line.replace("STATUS:", "").strip()
                if s in ["ALIGNED","CONFLICT","GAP","UNRELATED"]:
                    result["status"] = s
            elif line.startswith("TOPIC:"):
                result["topic"] = line.replace("TOPIC:", "").strip()
            elif line.startswith("DETAIL:"):
                result["detail"] = line.replace("DETAIL:", "").strip()
            elif line.startswith("RISK:"):
                r = line.replace("RISK:", "").strip()
                if r in ["LOW","MEDIUM","HIGH"]:
                    result["risk"] = r
        return result
    except Exception as e:
        return {
            "status": "ERROR", "topic": "خطا", "detail": str(e),
            "risk": "UNKNOWN", "text_a": text_a[:250], "text_b": text_b[:250],
            "source_a": source_a, "source_b": source_b
        }

def run_analysis(groq_client, col, chunks_a, name_a, name_b, max_pairs, progress_bar):
    """pipeline اصلی تحلیل"""
    results, seen = [], set()
    to_process = chunks_a[:max_pairs]

    for i, chunk in enumerate(to_process):
        progress_bar.progress(
            (i+1)/len(to_process),
            text=f"تحلیل chunk {i+1} از {len(to_process)}..."
        )
        similars = find_similar(col, chunk["text"], name_b, top_k=2)
        for sim in similars:
            key = (chunk["text"][:50], sim["text"][:50])
            if key in seen:
                continue
            seen.add(key)
            r = analyze_pair(groq_client, chunk["text"], sim["text"], name_a, name_b)
            if r["status"] not in ["UNRELATED", "UNKNOWN"]:
                results.append(r)

    return results



def save_to_history(name_a, name_b, results):
    """نتایج رو با تاریخ ذخیره می‌کنه"""
    history = load_history()
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "doc_a": name_a,
        "doc_b": name_b,
        "total": len(results),
        "conflicts": sum(1 for r in results if r["status"]=="CONFLICT"),
        "gaps": sum(1 for r in results if r["status"]=="GAP"),
        "aligned": sum(1 for r in results if r["status"]=="ALIGNED"),
        "results": results
    }
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_history():
    """تاریخچه تحلیل‌های قبلی رو می‌خونه"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return []



def generate_csv(results, name_a, name_b):
    """گزارش CSV می‌سازه"""
    STATUS_FA = {
        "CONFLICT":"تضاد", "GAP":"شکاف",
        "ALIGNED":"هم‌راستا", "UNKNOWN":"نامشخص", "ERROR":"خطا"
    }
    RISK_FA = {"HIGH":"بالا", "MEDIUM":"متوسط", "LOW":"پایین", "UNKNOWN":"نامشخص"}

    df = pd.DataFrame([{
        "وضعیت": STATUS_FA.get(r["status"], r["status"]),
        "موضوع": r["topic"],
        "ریسک": RISK_FA.get(r["risk"], r["risk"]),
        "توضیح": r["detail"],
        f"متن {name_a}": r["text_a"],
        f"متن {name_b}": r["text_b"],
    } for r in results])
    return df.to_csv(index=False).encode("utf-8-sig")


STATUS_FA = {
    "CONFLICT":"❌ تضاد", "GAP":"⚠️ شکاف",
    "ALIGNED":"✅ هم‌راستا", "UNKNOWN":"❓", "ERROR":"🔴 خطا"
}
RISK_FA = {
    "HIGH":"🔴 بالا", "MEDIUM":"🟡 متوسط",
    "LOW":"🟢 پایین", "UNKNOWN":"❓"
}
RISK_COLOR = {"HIGH":"#e53935", "MEDIUM":"#fb8c00", "LOW":"#43a047", "UNKNOWN":"#999"}

st.title("🔬 سامانه یکپارچه‌سازی دانش سازمانی")
st.caption("Multi-Source Knowledge Integration System | فاز پایلوت — NanoDaru Pharmaceutical")
st.divider()

# سایدبار
with st.sidebar:
    st.header("⚙️ تنظیمات")
    api_key = st.text_input("🔑 Groq API Key", type="password",
                             help="از console.groq.com رایگان بگیر")
    st.divider()
    st.subheader("📄 آپلود اسناد")
    name_a = st.text_input("نام سند اول", value="SOP")
    name_b = st.text_input("نام سند دوم", value="مقررات")
    file_a = st.file_uploader("سند اول", type=["pdf","txt"])
    file_b = st.file_uploader("سند دوم", type=["pdf","txt"])
    st.divider()
    max_pairs = st.slider("تعداد chunk برای تحلیل", 5, 40, 15,
                          help="بیشتر = دقیق‌تر ولی کندتر")
    run_btn = st.button("▶ شروع تحلیل", type="primary",
                        use_container_width=True)
    st.caption("⚠️ این سیستم مرجع نهایی نیست — نیاز به بررسی کارشناس دارد")

# تب‌ها
tab1, tab2, tab3 = st.tabs(["📊 تحلیل تضاد", "💬 چت‌بات", "📋 تاریخچه"])

# ── تب ۱: تحلیل ───────────────────────────────────────────────
with tab1:
    if not file_a or not file_b:
        st.info("👆 دو فایل رو از سایدبار آپلود کن")
        col1, col2, col3 = st.columns(3)
        col1.metric("قابلیت", "تحلیل تضاد")
        col2.metric("قابلیت", "شناسایی شکاف")
        col3.metric("قابلیت", "چت‌بات هوشمند")
        st.stop()

    if run_btn:
        if not api_key:
            st.error("❌ Groq API Key رو از سایدبار وارد کن")
            st.stop()

        groq_client = Groq(api_key=api_key)

        with st.status("در حال پردازش اسناد...", expanded=True) as status:
            st.write(f"📖 خواندن {name_a}...")
            text_a = read_file(file_a)
            chunks_a = chunk_document(text_a, name_a)
            st.write(f"✅ {name_a}: {len(chunks_a)} بخش")

            st.write(f"📖 خواندن {name_b}...")
            text_b = read_file(file_b)
            chunks_b = chunk_document(text_b, name_b)
            st.write(f"✅ {name_b}: {len(chunks_b)} بخش")

            st.write("🗄️ ایندکس‌گذاری...")
            col_db = get_fresh_collection()
            add_chunks(col_db, chunks_a)
            add_chunks(col_db, chunks_b)
            st.write("✅ پایگاه دانش آماده شد")
            status.update(label="✅ آماده برای تحلیل", state="complete")

        st.session_state["col_db"] = col_db
        st.session_state["name_a"] = name_a
        st.session_state["name_b"] = name_b
        st.session_state["groq_client"] = groq_client
        st.session_state["chat_history"] = []

        bar = st.progress(0, text="در حال تحلیل...")
        results = run_analysis(groq_client, col_db, chunks_a,
                               name_a, name_b, max_pairs, bar)
        bar.empty()

        save_to_history(name_a, name_b, results)
        st.session_state["results"] = results

    if "results" in st.session_state:
        results = st.session_state["results"]
        na = st.session_state.get("name_a", "سند اول")
        nb = st.session_state.get("name_b", "سند دوم")

        if not results:
            st.warning("هیچ بخش مرتبطی پیدا نشد")
            st.stop()

        # آمار
        n_conflict = sum(1 for r in results if r["status"]=="CONFLICT")
        n_gap = sum(1 for r in results if r["status"]=="GAP")
        n_aligned = sum(1 for r in results if r["status"]=="ALIGNED")
        n_high = sum(1 for r in results if r["risk"]=="HIGH")

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("❌ تضاد", n_conflict)
        c2.metric("⚠️ شکاف", n_gap)
        c3.metric("✅ هم‌راستا", n_aligned)
        c4.metric("🔴 ریسک بالا", n_high)
        st.divider()

        # فیلتر
        f1, f2 = st.columns(2)
        with f1:
            filter_status = st.multiselect(
                "فیلتر وضعیت",
                ["CONFLICT","GAP","ALIGNED"],
                default=["CONFLICT","GAP"]
            )
        with f2:
            filter_risk = st.multiselect(
                "فیلتر ریسک",
                ["HIGH","MEDIUM","LOW"],
                default=["HIGH","MEDIUM"]
            )

        filtered = [r for r in results
                    if r["status"] in filter_status
                    and r["risk"] in filter_risk]

        st.caption(f"{len(filtered)} مورد از {len(results)} نمایش داده میشه")

        for r in filtered:
            color = RISK_COLOR.get(r["risk"], "#999")
            with st.expander(
                f"{STATUS_FA.get(r['status'])} | "
                f"ریسک: {RISK_FA.get(r['risk'])} | {r['topic']}",
                expanded=(r["risk"]=="HIGH")
            ):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown(f"**📄 {r['source_a']}**")
                    st.info(r["text_a"])
                with cb:
                    st.markdown(f"**📄 {r['source_b']}**")
                    st.info(r["text_b"])
                st.warning(f"💬 {r['detail']}")
                st.caption("⚠️ این تحلیل نیاز به بررسی کارشناس انسانی دارد")

        st.divider()
        st.download_button(
            "⬇️ دانلود گزارش CSV",
            generate_csv(results, na, nb),
            f"report_{na}_{nb}_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )

# ── تب ۲: چت‌بات ──────────────────────────────────────────────
with tab2:
    if "col_db" not in st.session_state:
        st.info("👆 اول از تب 'تحلیل تضاد' فایل‌ها رو آپلود و تحلیل کن")
        st.stop()

    st.markdown("از اسناد آپلود شده سوال بپرس")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("سوالت رو بنویس..."):
        st.session_state["chat_history"].append(
            {"role": "user", "content": question})
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
            [f"[{nb}]: {c['text'][:400]}" for c in sim_b]
        )

        with st.chat_message("assistant"):
            with st.spinner("در حال پاسخ..."):
                try:
                    res = gc.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "user", "content":
                            f"بر اساس این اسناد به سوال جواب بده:\n{context}\n\n"
                            f"سوال: {question}\n"
                            f"جواب دقیق و کوتاه فارسی:"}],
                        max_tokens=500,
                        temperature=0.3
                    )
                    answer = res.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"خطا: {e}"
                st.write(answer)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": answer})

# ── تب ۳: تاریخچه ─────────────────────────────────────────────
with tab3:
    st.subheader("📋 تاریخچه تحلیل‌ها")
    history = load_history()

    if not history:
        st.info("هنوز هیچ تحلیلی انجام نشده")
        st.stop()

    for entry in reversed(history):
        with st.expander(
            f"📅 {entry['date']} | {entry['doc_a']} vs {entry['doc_b']} | "
            f"تضاد: {entry['conflicts']} | شکاف: {entry['gaps']}"
        ):
            c1,c2,c3 = st.columns(3)
            c1.metric("تضاد", entry["conflicts"])
            c2.metric("شکاف", entry["gaps"])
            c3.metric("هم‌راستا", entry["aligned"])


