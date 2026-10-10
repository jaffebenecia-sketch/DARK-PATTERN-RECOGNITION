"""Dark Pattern Radar - Streamlit frontend (neo-brutalist). Run: streamlit run app.py"""
import html
import numpy as np, pandas as pd, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="Dark Pattern Radar", page_icon="◈", layout="wide")
esc = html.escape
PER_PAGE = 12
EX = "Hurry! Only 2 left in stock — this exclusive deal ends in 10 minutes."

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Space+Mono:wght@400;700&display=swap');
:root{--bg:#FFF6E0;--ink:#111;--yellow:#FFE14D;--pink:#FF6B9D;--blue:#4D9DFF;--green:#5BE37D;--white:#fff}
html,body,.stApp,[class*="css"]{font-family:'Space Grotesk',sans-serif;color:var(--ink)}
.stApp{background:var(--bg);background-image:radial-gradient(#11111122 1.2px,transparent 1.2px);background-size:22px 22px}
[data-testid="stHeader"],[data-testid="stToolbar"],footer,#MainMenu{display:none!important}
.block-container{max-width:1200px;padding:28px 22px 60px}
h1,h2,h3{font-family:'Space Grotesk',sans-serif;font-weight:700;letter-spacing:-.02em;margin:0;padding:0}
.top{display:flex;align-items:center;justify-content:space-between;gap:18px;margin-bottom:26px;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:14px}
.logo{width:56px;height:56px;background:var(--pink);border:4px solid var(--ink);box-shadow:5px 5px 0 var(--ink);display:grid;place-items:center;font-size:30px;font-weight:700}
.eyebrow{font:700 12px 'Space Mono',monospace;text-transform:uppercase;letter-spacing:.12em}
.brand h1{font-size:34px;line-height:1}
.status{font:700 13px 'Space Mono',monospace;background:var(--green);border:3px solid var(--ink);box-shadow:4px 4px 0 var(--ink);padding:6px 12px}
.hero{background:var(--yellow);border:4px solid var(--ink);box-shadow:10px 10px 0 var(--ink);padding:34px 36px;margin-bottom:34px}
.tag{display:inline-block;background:var(--ink);color:var(--yellow);font:700 12px 'Space Mono',monospace;padding:4px 10px;letter-spacing:.1em;margin-bottom:14px}
.hero h2{font-size:clamp(28px,4.4vw,52px);line-height:1.02;max-width:820px;margin-bottom:14px}
.hero p{font-size:17px;max-width:680px;margin:0;font-weight:500}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:22px;margin-bottom:34px}
.stat{border:4px solid var(--ink);box-shadow:6px 6px 0 var(--ink);padding:16px 18px}
.stat:nth-child(1){background:var(--blue)}.stat:nth-child(2){background:var(--pink)}.stat:nth-child(3){background:var(--green)}.stat:nth-child(4){background:var(--white)}
.stat label{display:block;font:700 12px 'Space Mono',monospace;text-transform:uppercase;letter-spacing:.08em}
.stat strong{display:block;font-size:44px;line-height:1.1;margin:4px 0}
.stat small{font:400 12px 'Space Mono',monospace}
[class*="st-key-card"]{background:var(--white);border:4px solid var(--ink)!important;border-radius:0!important;box-shadow:8px 8px 0 var(--ink);padding:22px!important}
.card-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:14px}
.card-head h3{font-size:24px}.sub{font:400 13px 'Space Mono',monospace;margin-top:2px}
.pill{font:700 12px 'Space Mono',monospace;background:var(--yellow);border:3px solid var(--ink);padding:3px 10px;white-space:nowrap}
textarea{border:3px solid var(--ink)!important;border-radius:0!important;background:var(--bg)!important;font:400 15px 'Space Mono',monospace!important;color:var(--ink)!important}
[data-baseweb="textarea"],[data-baseweb="base-input"],[data-baseweb="input"]{border-radius:0!important;border:0!important;background:transparent!important}
[data-baseweb="input"] input{border:3px solid var(--ink)!important;background:var(--bg)!important;font-family:'Space Mono',monospace!important;color:var(--ink)!important}
[data-baseweb="select"]>div{border:3px solid var(--ink)!important;border-radius:0!important;background:var(--bg)!important;font-family:'Space Mono',monospace}
.stButton>button,.stDownloadButton>button{background:var(--yellow);color:var(--ink);border:3px solid var(--ink);border-radius:0;box-shadow:4px 4px 0 var(--ink);font:700 15px 'Space Grotesk',sans-serif;padding:8px 18px;transition:transform .08s,box-shadow .08s}
.stButton>button:hover,.stDownloadButton>button:hover{background:var(--pink);border-color:var(--ink);color:var(--ink);transform:translate(-2px,-2px);box-shadow:6px 6px 0 var(--ink)}
.stButton>button:active,.stDownloadButton>button:active{transform:translate(4px,4px);box-shadow:0 0 0 var(--ink)}
.stButton>button:disabled{background:#ddd;box-shadow:none;opacity:.6}
label p,[data-testid="stWidgetLabel"] p{color:var(--ink)!important;opacity:1!important;font:700 12px 'Space Mono',monospace!important;text-transform:uppercase;letter-spacing:.08em}
.verdict{border:4px solid var(--ink);padding:14px 16px;margin-top:16px;box-shadow:5px 5px 0 var(--ink)}
.verdict.dark{background:var(--pink)}.verdict.safe{background:var(--green)}
.verdict b{font-size:22px;display:block}.verdict span{font:700 13px 'Space Mono',monospace}
.meter{height:22px;border:3px solid var(--ink);background:var(--white);margin:12px 0}
.meter div{height:100%;background:var(--ink)}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:6px}
.chip{border:3px solid var(--ink);padding:3px 9px;font:700 13px 'Space Mono',monospace}
.chip.up{background:var(--pink)}.chip.down{background:var(--green)}
.legend{font:400 12px 'Space Mono',monospace;margin-top:10px}
.steps{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.step{border:3px solid var(--ink);padding:12px;background:var(--bg)}
.step .num{display:inline-block;background:var(--ink);color:var(--yellow);font:700 14px 'Space Mono',monospace;padding:1px 8px;margin-bottom:6px}
.step b{display:block;font-size:17px}.step small{font:400 12px 'Space Mono',monospace}
.barrow{display:grid;grid-template-columns:130px 1fr 48px;gap:10px;align-items:center;margin:8px 0;font:700 13px 'Space Mono',monospace}
.bartrack{height:20px;border:3px solid var(--ink);background:var(--white)}
.barfill{height:100%;background:var(--blue);border-right:3px solid var(--ink)}
.tablewrap{overflow-x:auto;border:3px solid var(--ink);margin-top:14px}
table.data{width:100%;border-collapse:collapse;font-size:14px;background:var(--white)}
table.data th{background:var(--ink);color:var(--yellow);font:700 12px 'Space Mono',monospace;text-transform:uppercase;text-align:left;padding:8px 10px}
table.data td{border-top:2px solid var(--ink);padding:8px 10px;vertical-align:top}
table.data td.t{max-width:560px}
.label{border:2px solid var(--ink);padding:1px 8px;font:700 12px 'Space Mono',monospace;white-space:nowrap}
.label.bad{background:var(--pink)}.label.good{background:var(--green)}
.footer{font:700 12px 'Space Mono',monospace;margin-top:34px;text-align:center}
.pageinfo{font:700 13px 'Space Mono',monospace;text-align:center;padding-top:8px}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def load():  # same model/split as pipeline.py
    d = pd.read_csv("data/dataset.tsv", sep="\t").drop_duplicates("text")
    d["cat"] = d["Pattern Category"]
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(d, groups=d.page_id))
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                      LogisticRegression(max_iter=1000, C=3, random_state=42))
    m.fit(d.iloc[tr].text, d.iloc[tr].label)
    acc = (m.predict(d.iloc[te].text) == d.iloc[te].label).mean()
    return m, d, acc


m, data, acc = load()
ss = st.session_state
ss.setdefault("txt", "")
ss.setdefault("page", 0)

st.markdown(f"""
<div class="top"><div class="brand"><div class="logo">◈</div><div><div class="eyebrow">Responsible commerce AI</div><h1>Dark Pattern Radar</h1></div></div><div class="status">● Live ML model</div></div>
<div class="hero"><span class="tag">TEXT CLASSIFICATION · E-COMMERCE</span><h2>Spot the pressure tactics hiding in a product page.</h2><p>Paste interface copy, run a TF-IDF + logistic regression detector trained on real shopping-site text, see which words drove the call, and browse the labeled dataset.</p></div>
<div class="stats">
<div class="stat"><label>Dataset records</label><strong>{len(data):,}</strong><small>Unique real text segments</small></div>
<div class="stat"><label>Dark-pattern examples</label><strong>{int(data.label.sum()):,}</strong><small>Label = 1</small></div>
<div class="stat"><label>Pattern categories</label><strong>{data.cat[data.label == 1].nunique()}</strong><small>Non-neutral classes</small></div>
<div class="stat"><label>Held-out accuracy</label><strong>{acc:.0%}</strong><small>Split by page_id</small></div></div>
""", unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left, st.container(key="card_analyze"):
    st.markdown('<div class="card-head"><div><h3>Analyze interface text</h3><div class="sub">Paste copy from a product page, cart, or checkout.</div></div><span class="pill">Live detector</span></div>', unsafe_allow_html=True)
    text = st.text_area("Text to inspect", key="txt", height=120,
                        placeholder="Example: Hurry! Only 2 left in stock — deal ends in 10 minutes.")
    b1, b2, b3 = st.columns([1, 1, 1])
    b1.button("Analyze", use_container_width=True)
    b2.button("Try example", use_container_width=True, on_click=lambda: ss.update(txt=EX))
    b3.button("Clear", use_container_width=True, on_click=lambda: ss.update(txt=""))
    if text.strip():
        p = m.predict_proba([text])[0, 1]
        dark = p >= .5
        st.markdown(f'<div class="verdict {"dark" if dark else "safe"}"><b>{"Potential dark pattern detected" if dark else "No strong signal detected"}</b><span>P(dark pattern) = {p:.0%}</span></div><div class="meter"><div style="width:{p * 100:.0f}%"></div></div>', unsafe_allow_html=True)
        v, clf = m[0], m[1]
        x = v.transform([text]); names = v.get_feature_names_out()
        c = sorted(((names[j], x[0, j] * clf.coef_[0][j]) for j in x.nonzero()[1]), key=lambda t: -abs(t[1]))[:10]
        if c:
            chips = "".join(f'<span class="chip {"up" if w > 0 else "down"}">{esc(n)} {w:+.2f}</span>' for n, w in c)
            st.markdown(f'<label class="field-label" style="font:700 12px Space Mono;text-transform:uppercase">Why? Word contributions</label><div class="chips">{chips}</div><div class="legend">Pink pushes toward dark pattern · green pushes toward not dark.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="legend">No known words in this text; the model has nothing to go on.</div>', unsafe_allow_html=True)

with right, st.container(key="card_pipeline"):
    st.markdown('''<div class="card-head"><div><h3>Detection pipeline</h3><div class="sub">From raw copy to an explainable flag.</div></div><span class="pill">4 stages</span></div>
<div class="steps"><div class="step"><div class="num">01</div><b>Input</b><small>Paste interface text</small></div>
<div class="step"><div class="num">02</div><b>Vectorize</b><small>TF-IDF, 1-2 grams</small></div>
<div class="step"><div class="num">03</div><b>Classify</b><small>Logistic regression</small></div>
<div class="step"><div class="num">04</div><b>Explain</b><small>Per-word weights</small></div></div>''', unsafe_allow_html=True)
    counts = data.cat.value_counts()
    bars = "".join(f'<div class="barrow"><span>{esc(c)}</span><div class="bartrack"><div class="barfill" style="width:{max(2, n / len(data) * 100):.1f}%"></div></div><b>{n}</b></div>' for c, n in counts.items())
    st.markdown(f'<div class="card-head" style="margin-top:22px"><div><h3>Category mix</h3></div></div>{bars}', unsafe_allow_html=True)

st.markdown("<div style='height:34px'></div>", unsafe_allow_html=True)
with st.container(key="card_explorer"):
    st.markdown('<div class="card-head"><div><h3>Dataset explorer</h3><div class="sub">Search, filter, and export the labeled records.</div></div></div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([3, 1, 1])
    q = f1.text_input("Search", placeholder="Search text, category, page id…", key="q")
    lab = f2.selectbox("Label", ["All", "Dark pattern", "Not dark"])
    cat = f3.selectbox("Category", ["All"] + sorted(data.cat.unique()))
    df = data
    if q.strip(): df = df[df.text.str.contains(q.strip(), case=False, regex=False) | df.cat.str.contains(q.strip(), case=False, regex=False) | (df.page_id.astype(str) == q.strip())]
    if lab != "All": df = df[df.label == (lab == "Dark pattern")]
    if cat != "All": df = df[df.cat == cat]
    pages = max(1, -(-len(df) // PER_PAGE))
    ss.page = min(ss.page, pages - 1)
    rows = "".join(f'<tr><td>{r.page_id}</td><td class="t">{esc(r.text)}</td><td><span class="label {"bad" if r.label else "good"}">{"Dark" if r.label else "Not dark"}</span></td><td>{esc(r.cat)}</td></tr>'
                   for r in df.iloc[ss.page * PER_PAGE:(ss.page + 1) * PER_PAGE].itertuples())
    st.markdown(f'<div class="card-head" style="margin:10px 0 0"><span class="pill">{len(df):,} records</span></div><div class="tablewrap"><table class="data"><tr><th>Page</th><th>Text</th><th>Label</th><th>Category</th></tr>{rows}</table></div>', unsafe_allow_html=True)
    n1, n2, n3, n4 = st.columns([1, 2, 1, 2])
    if n1.button("← Prev", disabled=ss.page == 0, use_container_width=True): ss.page -= 1; st.rerun()
    n2.markdown(f'<div class="pageinfo">Page {ss.page + 1} / {pages}</div>', unsafe_allow_html=True)
    if n3.button("Next →", disabled=ss.page >= pages - 1, use_container_width=True): ss.page += 1; st.rerun()
    n4.download_button("Export CSV", df.drop(columns="Pattern Category").to_csv(index=False), "dark_patterns_filtered.csv", "text/csv", use_container_width=True)

st.markdown('<div class="footer">Dark Pattern Radar · Data: yamanalab/ec-darkpattern · Model trains locally on startup</div>', unsafe_allow_html=True)
