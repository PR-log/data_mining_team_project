"""build_v6.py — Macro-F1 최대화 본격 시도.
레버: (1) 특성 공학  (2) class_weight 스킴  (3) ★ 클래스별 임계값 최적화(좌표상승, Macro-F1 직접 최대화).
모두 train 내부 split 으로만 튜닝 → 누수 없음. best 가 v5(0.207) 넘으면 predict_v6.csv 생성.
"""
import warnings; warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import hanwoo_pipeline as hp

T0 = time.time()
TRAIN_N = 600_000
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)

log("load...")
train = hp.read_csv_safe("hanwoo_train.csv", nrows=TRAIN_N, usecols=hp.TEST_AVAILABLE_COLS + ["LAST_GRADE"])
test_raw = hp.read_csv_safe("test_hanwoo.csv")
area = hp.read_csv_safe("hanwoo_area.csv")
death = hp.read_csv_safe("hanwoo_death.csv")
lineage = hp.read_csv_safe("hanwoo_lineage.csv", usecols=["CATTLE_NO", "KPN_NO"])
weather = hp.read_csv_safe("hanwoo_weather.csv")

parts, numf, catf = hp.build_submission_frame(train, test_raw, weather_df=weather,
                                              area_df=area, death_df=death, lineage_df=lineage)
log(f"frame built. train={parts['train'].shape}")

# ---------- 특성 공학 ----------
def engineer(d):
    d = d.copy()
    w = pd.to_numeric(d["WEIGHT"], errors="coerce")
    a = pd.to_numeric(d["AGE"], errors="coerce").clip(lower=1)
    d["weight_per_age"] = w / a
    by = pd.to_datetime(d["BIRTH_YMD"], errors="coerce")
    d["birth_year"] = by.dt.year
    d["birth_month"] = by.dt.month
    return d

parts = {k: engineer(v) for k, v in parts.items()}
numf2 = numf + ["weight_per_age", "birth_year", "birth_month"]
if "abatt_year" in parts["train"].columns:
    numf2 += ["abatt_year"]
catf2 = catf + (["eupmyeondong"] if "eupmyeondong" in parts["train"].columns else [])
numf2 = [c for c in numf2 if c in parts["train"].columns]
catf2 = [c for c in catf2 if c in parts["train"].columns]
feats = numf2 + catf2
log(f"features: {len(feats)} (num={len(numf2)}, cat={len(catf2)})")

# ---------- 3-way split : fit / tune(임계값) / test(평가) ----------
d = parts["train"]
X, y = d[feats], d["LAST_GRADE"]
Xfit, Xtmp, yfit, ytmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
Xtune, Xtest, ytune, ytest = train_test_split(Xtmp, ytmp, test_size=0.50, random_state=42, stratify=ytmp)
# 임계값 튜닝용 tune set 은 속도 위해 최대 40k
if len(Xtune) > 40000:
    Xtune, _, ytune, _ = train_test_split(Xtune, ytune, train_size=40000, random_state=42, stratify=ytune)
log(f"split: fit={len(Xfit)} tune={len(Xtune)} test={len(Xtest)}")


def class_weights(yy, scheme):
    vc = yy.value_counts(); N = len(yy); K = len(vc)
    if scheme == "balanced":
        return {c: N / (K * n) for c, n in vc.items()}
    if scheme == "sqrt":
        w = {c: np.sqrt(N / n) for c, n in vc.items()}
        m = np.mean(list(w.values()))
        return {c: v / m for c, v in w.items()}
    return None


def train_model(cw):
    pre = hp.build_ordinal_preprocess(numf2, catf2)
    mdl = LGBMClassifier(n_estimators=800, num_leaves=127, learning_rate=0.04,
                         subsample=0.85, colsample_bytree=0.8, min_child_samples=40,
                         class_weight=cw, n_jobs=-1, random_state=42, verbosity=-1)
    pipe = Pipeline([("p", pre), ("m", mdl)])
    pipe.fit(Xfit, yfit)
    return pipe


def optimize_weights(proba, ytrue, classes, passes=3):
    """클래스별 곱가중치 좌표상승으로 Macro-F1 직접 최대화."""
    yt = np.asarray(ytrue)
    w = np.ones(proba.shape[1])
    grid = [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.6, 3.4]
    def mf1(ww): return f1_score(yt, classes[np.argmax(proba * ww, axis=1)], average="macro")
    best = mf1(w)
    for _ in range(passes):
        for c in range(len(w)):
            bc, bf = w[c], best
            for g in grid:
                w[c] = g
                f = mf1(w)
                if f > bf:
                    bf, bc = f, g
            w[c] = bc; best = bf
    return w, best


results = []
for scheme in ["balanced", "sqrt"]:
    t = time.time()
    pipe = train_model(class_weights(yfit, scheme))
    classes = pipe.classes_
    p_tune = pipe.predict_proba(Xtune)
    p_test = pipe.predict_proba(Xtest)
    base = f1_score(ytest, classes[np.argmax(p_test, axis=1)], average="macro")
    w, _ = optimize_weights(p_tune, ytune, classes)
    pred_opt = classes[np.argmax(p_test * w, axis=1)]
    opt = f1_score(ytest, pred_opt, average="macro")
    acc = (pred_opt == ytest.values).mean()
    results.append(dict(scheme=scheme, base=base, opt=opt, acc=acc, w=w, pipe=pipe))
    log(f"[{scheme}] argmax MacroF1={base:.4f} | +임계값최적화={opt:.4f} (acc={acc:.3f}) [{time.time()-t:.0f}s]")

best = max(results, key=lambda r: r["opt"])
log("=" * 60)
log(f"BEST = {best['scheme']} + 임계값최적화  →  Macro-F1 {best['opt']:.4f}")
log(f"(v5 기준 0.207 대비 {best['opt']-0.207:+.4f})")

# ---------- best 가 v5 넘으면 predict_v6 생성 ----------
if best["opt"] >= 0.207:
    log("v5 초과 → 전체 train 재학습 후 predict_v6.csv 생성")
    pre = hp.build_ordinal_preprocess(numf2, catf2)
    mdl = LGBMClassifier(n_estimators=800, num_leaves=127, learning_rate=0.04,
                         subsample=0.85, colsample_bytree=0.8, min_child_samples=40,
                         class_weight=class_weights(y, best["scheme"]), n_jobs=-1,
                         random_state=42, verbosity=-1)
    final = Pipeline([("p", pre), ("m", mdl)])
    final.fit(X, y)
    classes = final.classes_
    p_test_real = final.predict_proba(parts["test"][feats])
    w_best = best["w"]
    test_pred = classes[np.argmax(p_test_real * w_best, axis=1)]
    sub = test_raw.copy(); sub["LAST_GRADE"] = test_pred
    sub.to_csv("predict_v6.csv", index=False, encoding="utf-8-sig")
    np.save("v6_class_weights.npy", w_best)
    log(f"saved predict_v6.csv  {sub.shape}")
    log("LAST_GRADE 분포:\n" + sub["LAST_GRADE"].value_counts().to_string())
else:
    log("v5 를 넘지 못함 → v5 유지 (predict_v6 미생성)")

log("DONE_BUILD_V6")
