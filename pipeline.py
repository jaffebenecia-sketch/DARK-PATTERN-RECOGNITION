"""Dark-pattern text classifier: TF-IDF + LogisticRegression, audit by subgroup, top-feature explainability."""
import pandas as pd, numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score

d = pd.read_csv("data/dataset.tsv", sep="\t").drop_duplicates("text")
d["cat"] = d["Pattern Category"]
# split by page_id so segments of one site never straddle train/test
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(d, groups=d.page_id))
train, test = d.iloc[tr], d.iloc[te]
m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True),
                  LogisticRegression(max_iter=1000, C=3, random_state=42))
m.fit(train.text, train.label)
p = m.predict(test.text)
print(f"train={len(train)} test={len(test)}")
print(classification_report(test.label, p, target_names=["Not dark", "Dark"], digits=3))

# audit 1: per-category recall (dark) / specificity (not dark)
print("== per pattern category (share predicted correctly) ==")
test = test.assign(pred=p, ok=(p == test.label))
print(test.groupby("cat").ok.agg(["mean", "size"]).round(3).sort_values("size", ascending=False))
# audit 2: text style = short (<=5 words) vs long
test["style"] = np.where(test.text.str.split().str.len() <= 5, "short (<=5 words)", "long (>5 words)")
print("== by text length style ==")
for s, g in test.groupby("style"):
    print(f"{s:20s} n={len(g):4d} acc={accuracy_score(g.label, g.pred):.3f} f1={f1_score(g.label, g.pred):.3f}")

# explainability: top features
v, c = m[0].get_feature_names_out(), m[1].coef_[0]
o = np.argsort(c)
print("== top DARK features =="); print([(v[i], round(c[i], 2)) for i in o[::-1][:15]])
print("== top NOT-DARK features =="); print([(v[i], round(c[i], 2)) for i in o[:15]])
print("== sample errors ==")
print(test[~test.ok][["text", "label", "cat"]].head(8).to_string(max_colwidth=80))
