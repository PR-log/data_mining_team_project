"""
make_submission.py
==================
검증/제출용 파이프라인. test_hanwoo.csv 에는 도체 측정값(BACKFAT/REA/WINDEX/WGRADE/
INSFAT/YUKSAK/FATSAK/TISSUE/GROWTH/COST_AMT)이 없으므로, train/test 양쪽에 공통으로
존재하는 feature 만 사용한다.

사용 가능 원본 feature: sido, sigungu, eupmyeondong, stn, ABATT_DATE, JUDGE_DATE,
                        JUDGE_SEX, WEIGHT, AGE, BIRTH_YMD, CATTLE_NO, FARM_UNIQUE_NO
파생: 기상 rolling(stn+ABATT_DATE), 농장규모(FARM_UNIQUE_NO), 혈통KPN(CATTLE_NO),
      폐사(FARM_UNIQUE_NO), 날짜(계절/사육일수/판정지연)

산출: 1) 내부 train/valid split 으로 현실 Macro-F1 추정
      2) 전체 train 으로 재학습 후 test 예측 → 제출 파일 저장 (UTF-8-BOM)
"""
import warnings; warnings.filterwarnings("ignore")
import time, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

import hanwoo_pipeline as hp

T0 = time.time()
BASE = Path(".")
TRAIN_N = 800_000       # None=전체 train. 첫 검증은 표본으로 빠르게.
WINDOWS = (30, 90, 180)
TARGET = "LAST_GRADE"

# test 와 공통으로 존재하는 원본 컬럼만 사용
TEST_AVAILABLE = ["sido", "sigungu", "eupmyeondong", "stn", "ABATT_DATE", "JUDGE_DATE",
                  "JUDGE_SEX", "WEIGHT", "AGE", "BIRTH_YMD", "CATTLE_NO", "FARM_UNIQUE_NO"]

print(f"== make_submission (TRAIN_N={TRAIN_N}) ==")

# ---------- 로드 ----------
train = hp.read_csv_safe(BASE / "hanwoo_train.csv", nrows=TRAIN_N,
                         usecols=TEST_AVAILABLE + [TARGET])
test_raw = hp.read_csv_safe(BASE / "test_hanwoo.csv")     # 원본 그대로(출력용)
test = test_raw.copy()
area = hp.read_csv_safe(BASE / "hanwoo_area.csv")
death = hp.read_csv_safe(BASE / "hanwoo_death.csv")
lineage = hp.read_csv_safe(BASE / "hanwoo_lineage.csv", usecols=["CATTLE_NO", "KPN_NO"])
weather = hp.read_csv_safe(BASE / "hanwoo_weather.csv")
print(f"train={train.shape} test={test.shape}  ({time.time()-T0:.0f}s)")


# ---------- 기본 정제 ----------
def prep_base(df):
    df = hp.normalize_missing(df)
    for c in ["WEIGHT", "AGE", "stn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return hp.add_date_features(df)

train = prep_base(train).dropna(subset=[TARGET]).reset_index(drop=True)
test = prep_base(test).reset_index(drop=True)
parts = {"train": train, "test": test}

# ---------- 보조 데이터 병합 (행 1:1 유지) ----------
area_table = hp.prepare_area_table(area)
death_table = hp.prepare_death_table(death)
lineage_table = hp.prepare_lineage_table(lineage)
for k in parts:
    parts[k] = hp.merge_area_features(parts[k], area_table)
    parts[k] = hp.merge_death_features(parts[k], death_table)
    parts[k] = hp.merge_lineage_features(parts[k], lineage_table)

# ---------- 기상 rolling (shift(1), leakage-safe) ----------
wd = hp.prepare_weather_daily(weather)
wf = hp.build_weather_rolling_features(wd, WINDOWS)
wf["stn"] = pd.to_numeric(wf["stn"], errors="coerce").astype("Int64")
for k in parts:
    parts[k] = hp.merge_weather_features(parts[k], wf)
print(f"merged. train={parts['train'].shape} test={parts['test'].shape}  ({time.time()-T0:.0f}s)")

# ---------- train 기준 결측 보간 ----------
# WEIGHT/AGE: train median
for col in ["WEIGHT", "AGE"]:
    med = pd.to_numeric(parts["train"][col], errors="coerce").median()
    for k in parts:
        parts[k][col] = pd.to_numeric(parts[k][col], errors="coerce").fillna(med)

# cow_year, AREA: 지역 median fallback (train 기준)
for col in ["cow_year", "AREA"]:
    spec = hp.fit_fallback_imputer(parts["train"], col, [["sigungu"], ["sido"]], kind="median", numeric=True)
    for k in parts:
        parts[k] = hp.apply_fallback_imputer(parts[k], spec)
for k in parts:
    parts[k]["log_cow_year"] = np.log1p(pd.to_numeric(parts[k]["cow_year"], errors="coerce").clip(lower=0))
    parts[k]["log_area"] = np.log1p(pd.to_numeric(parts[k]["AREA"], errors="coerce").clip(lower=0))

# KPN 빈도: train 기준
kpn_freq_map = parts["train"]["KPN_NO"].dropna().astype("string").value_counts().to_dict()
for k in parts:
    kpn = parts[k]["KPN_NO"].astype("string")
    parts[k]["kpn_known"] = kpn.notna().astype("int8")
    parts[k]["kpn_freq"] = kpn.map(kpn_freq_map).fillna(0).astype("int32")
    parts[k]["log_kpn_freq"] = np.log1p(parts[k]["kpn_freq"])

# 기상 rolling 결측: train median
weather_cols = [c for c in parts["train"].columns if any(c.endswith(f"_{w}d") for w in WINDOWS)]
for col in weather_cols:
    med = pd.to_numeric(parts["train"][col], errors="coerce").median()
    for k in parts:
        parts[k][col] = pd.to_numeric(parts[k][col], errors="coerce").fillna(med)

# ---------- feature 정의 ----------
numeric_features = (["WEIGHT", "AGE", "rearing_days", "judge_delay_days",
                     "cow_year", "log_cow_year", "log_area", "farm_death_count",
                     "kpn_known", "kpn_freq", "log_kpn_freq"] + weather_cols)
numeric_features = [c for c in numeric_features if c in parts["train"].columns]
categorical_features = ["JUDGE_SEX", "sido", "sigungu", "stn", "abatt_month", "abatt_season"]
categorical_features = [c for c in categorical_features if c in parts["train"].columns]
features = numeric_features + categorical_features
print(f"features: {len(features)} (num={len(numeric_features)}, cat={len(categorical_features)})")
print("numeric:", numeric_features)


def build_lgbm():
    from lightgbm import LGBMClassifier
    pre = hp.build_ordinal_preprocess(numeric_features, categorical_features)
    from sklearn.pipeline import Pipeline
    return Pipeline([("preprocess", pre),
                     ("model", LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                                              subsample=0.85, colsample_bytree=0.85,
                                              class_weight="balanced", n_jobs=-1,
                                              random_state=42, verbosity=-1))])


# ---------- 1) 내부 검증 (현실 Macro-F1) ----------
Xtr = parts["train"][features]
ytr = parts["train"][TARGET]
X_tr, X_va, y_tr, y_va = train_test_split(Xtr, ytr, test_size=0.2, random_state=42, stratify=ytr)
m = build_lgbm()
t = time.time()
m.fit(X_tr, y_tr)
val_pred = m.predict(X_va)
f1 = f1_score(y_va, val_pred, average="macro")
print(f"\n★ 내부검증 현실 Macro-F1 (도체변수 제외, test와 동일 조건) = {f1:.4f}   (fit {time.time()-t:.0f}s)")
rep = pd.DataFrame(classification_report(y_va, val_pred, output_dict=True, zero_division=0)).T
print(rep[["precision", "recall", "f1-score", "support"]].round(3).to_string())

# ---------- 2) 전체 train 재학습 → test 예측 → 제출 ----------
final = build_lgbm()
final.fit(Xtr, ytr)
test_pred = final.predict(parts["test"][features])

submission = test_raw.copy()
submission[TARGET] = test_pred
out_path = BASE / "submission_접수번호.csv"
submission.to_csv(out_path, index=False, encoding="utf-8-sig")   # UTF-8 BOM
print(f"\n저장: {out_path}  shape={submission.shape}")
print("제출 LAST_GRADE 분포:")
print(submission[TARGET].value_counts().to_string())
print(f"\nTOTAL {time.time()-T0:.0f}s")
print("DONE_SUBMISSION")
