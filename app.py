import streamlit as st
import joblib, re

st.set_page_config(page_title="Fake News & Clickbait Detector", page_icon="📰", layout="wide")

@st.cache_resource
def load_models():
    return joblib.load("model.pkl"), joblib.load("clickbait_model.pkl")

fake_model, cb_model = load_models()

def clean_news(t):
    t = str(t)
    t = re.sub(r"^.{0,80}?\(Reuters\)\s*-?\s*", "", t)
    t = re.sub(r"http\S+|pic\.twitter\.com\S+|www\.\S+", "", t)
    t = re.sub(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(,\s*\d{4})?", "", t, flags=re.I)
    t = re.sub(r"(?i)\b(reuters|featured image|getty images|21st century wire|image via)\b", "", t)
    t = t.lower(); t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def clean_cb(t):
    t = str(t).lower()
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

EXAMPLES = {
    "Clickbait headline": ("10 Things You Won't Believe About Your Phone", ""),
    "Normal headline": ("Premium gas discounted for a few hours", ""),
    "Real-style article": ("U.S. Senate passes defense spending bill",
        "The U.S. Senate on Thursday approved a defense spending bill by a vote of 86 to 8, sending the measure to the House of Representatives. Lawmakers from both parties said the bill would fund military pay raises and equipment upgrades. The House is expected to take up the legislation next week, according to a spokesman for the Republican leadership."),
    "Fake-style article": ("BREAKING: Obama Caught Red-Handed In Shocking Scandal The Media Is Hiding",
        "Patriots, wake up! The mainstream media refuses to tell you the truth about what really happened. Sources close to the situation say the corrupt establishment has been covering this up for years. Share this before they delete it!"),
}

if "headline" not in st.session_state: st.session_state.headline = ""
if "article" not in st.session_state: st.session_state.article = ""
if "history" not in st.session_state: st.session_state.history = []

def load_example(name):
    st.session_state.headline, st.session_state.article = EXAMPLES[name]

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About")
    st.write("Two machine-learning models:")
    st.markdown("- **Clickbait detector**: reads the headline\n- **Fake-news detector**: reads headline + article")
    st.caption("Models: TF-IDF + Logistic Regression (clickbait), TF-IDF + handcrafted features + Logistic Regression (fake news).")
    st.divider()
    st.subheader("Try an example")
    for name in EXAMPLES:
        st.button(name, on_click=load_example, args=(name,), use_container_width=True)
    st.divider()
    st.warning("Student ML project. Cross-dataset accuracy is limited, so this is not a substitute for fact-checking.")

# ---------- Main ----------
st.title("📰 Fake News & Clickbait Detector")
st.write("Enter a headline, and optionally the full article, then press **Analyze**.")

st.text_input("Headline", key="headline", placeholder="Type or paste a headline...")
st.text_area("Article text (needed for the fake-news check)", key="article", height=180,
             placeholder="Paste the full article here...")

if st.button("🔍 Analyze", type="primary"):
    h, a = st.session_state.headline.strip(), st.session_state.article.strip()
    if not h:
        st.warning("Please enter a headline.")
    else:
        p_cb = float(cb_model.predict_proba([clean_cb(h)])[0][1])
        p_fk = float(fake_model.predict_proba([clean_news(h + " " + a)])[0][1]) if a else None

        tab1, tab2, tab3 = st.tabs(["🎣 Clickbait", "🕵️ Fake News", "📊 Combined verdict"])

        with tab1:
            c1, c2 = st.columns(2)
            c1.metric("Verdict", "CLICKBAIT" if p_cb >= .5 else "NORMAL")
            c2.metric("Clickbait probability", f"{p_cb:.0%}")
            st.progress(p_cb)
            st.caption("Looks at headline style: numbers, teaser words, direct address, punctuation.")

        with tab2:
            if p_fk is None:
                st.info("Paste the article text to run the fake-news check. The model was trained on full articles.")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Verdict", "FAKE" if p_fk >= .5 else "REAL")
                c2.metric("Fake probability", f"{p_fk:.0%}")
                st.progress(p_fk)
                st.caption("Looks at the writing style and words of the whole article.")

        with tab3:
            if p_fk is None:
                st.info("Add article text to see the combined verdict.")
            elif p_fk >= .5 and p_cb >= .5:
                st.error("Fake-style article with a clickbait headline.")
            elif p_fk >= .5:
                st.error("Fake-style article, but the headline is not clickbait.")
            elif p_cb >= .5:
                st.warning("Real-style article, but the headline is clickbait.")
            else:
                st.success("Real-style article with a normal headline.")

        st.session_state.history.append({
            "Headline": h[:60],
            "Clickbait": f"{p_cb:.0%}",
            "Fake": f"{p_fk:.0%}" if p_fk is not None else "-"})

if st.session_state.history:
    st.subheader("Session history")
    st.dataframe(st.session_state.history[::-1], use_container_width=True)