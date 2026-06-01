# model_v5.ipynb 생성기 (제출용 실모델). python _make_v5_notebook.py
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md(r"""# Model V5 — 제출용 LAST_GRADE 예측 (test 호환 실모델)

## 버전 규칙
- **model_v4** : 도체변수(INSFAT/WINDEX/WGRADE 등) 포함. 내부 Macro-F1 0.98. **그러나 제출 불가** — `test_hanwoo.csv` 에 도체변수가 없음. → 참고/보관용.
- **model_v5 (이 노트북)** : ★ **검증 데이터에 실제로 존재하는 feature 만** 사용하는 제출용 실모델.
- 앞으로 큰 변경이 생기면 v6, v7 … 로 새로 만든다.

## 왜 v5 가 필요한가
`test_hanwoo.csv` 는 13개 컬럼뿐이고, 도체 측정값 10개(BACKFAT, REA, WINDEX, WGRADE,
INSFAT, YUKSAK, FATSAK, TISSUE, GROWTH, COST_AMT)가 **전부 빠져 있다.**
따라서 제출 모델은 아래 feature 만 쓸 수 있다.

- 개체: `WEIGHT`(도체중), `AGE`, `JUDGE_SEX`
- 위치/시점: `sido`, `sigungu`, `stn`, 도축월, 계절, 사육일수, 판정지연일
- 기상: `stn`+`ABATT_DATE` 기준 도축 전 30/90/180일 rolling (THI·폭염·강수 등)
- 농장: 사육두수(`cow_year`), 면적(`log_area`), 폐사수(`farm_death_count`)
- 혈통: KPN 빈도(`kpn_known`, `kpn_freq`, `log_kpn_freq`)

## 현실 점수 (이미 측정됨, 80만 행 기준)
**내부검증 Macro-F1 = 0.2054** — 0.98이 아니다. 등급을 결정하는 근내지방도(`INSFAT`)·
육량지수(`WINDEX`)가 test에 없기 때문이며, **모든 참가자가 같은 조건**이다.
점수는 20점짜리이고, 승부는 정성평가 80점(분석·아이디어)에서 난다.
""")

md("## 0. 임포트 & 설정")
code(r"""import warnings; warnings.filterwarnings("ignore")
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from lightgbm import LGBMClassifier

import hanwoo_pipeline as hp

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(".")
TARGET = "LAST_GRADE"
TRAIN_N = 800_000     # 첫 검증은 표본. 최종 제출은 None(전체)로.
WINDOWS = (30, 90, 180)
SUBMISSION_PATH = BASE / "predict_v5.csv"   # 제출 시 본인 접수번호.csv 로 변경""")

md("## 1. 데이터 로드 (train + 검증셋 test_hanwoo)")
code(r"""t0 = time.time()
train = hp.read_csv_safe(BASE / "hanwoo_train.csv", nrows=TRAIN_N,
                         usecols=hp.TEST_AVAILABLE_COLS + [TARGET])
test_raw = hp.read_csv_safe(BASE / "test_hanwoo.csv")     # 원본(출력 보존용)
area = hp.read_csv_safe(BASE / "hanwoo_area.csv")
death = hp.read_csv_safe(BASE / "hanwoo_death.csv")
lineage = hp.read_csv_safe(BASE / "hanwoo_lineage.csv", usecols=["CATTLE_NO", "KPN_NO"])
weather = hp.read_csv_safe(BASE / "hanwoo_weather.csv")
print(f"train={train.shape}  test={test_raw.shape}  ({time.time()-t0:.0f}s)")

# test 에 없는 도체변수 확인
train_full_cols = list(pd.read_csv(BASE / "hanwoo_train.csv", nrows=0).columns)
missing = [c for c in train_full_cols if c not in test_raw.columns]
print("★ test 에 없는(=사용 불가) 컬럼:", missing)""")

md("## 2. 특성 생성 (test 호환) — `build_submission_frame`\n기상/농장/혈통/날짜 병합 + train 기준 결측 보간을 한 번에 처리한다.")
code(r"""parts, numeric_features, categorical_features = hp.build_submission_frame(
    train, test_raw, weather_df=weather, area_df=area, death_df=death,
    lineage_df=lineage, windows=WINDOWS, target=TARGET)
features = numeric_features + categorical_features
print(f"train={parts['train'].shape}  test={parts['test'].shape}")
print(f"features: {len(features)} (num={len(numeric_features)}, cat={len(categorical_features)})")
print("categorical:", categorical_features)""")

md("## 3. 내부 검증 — 현실 Macro-F1\ntrain 을 80/20 으로 나눠 test 와 동일 조건(도체변수 없음)에서 점수를 추정한다.")
code(r"""def build_lgbm():
    pre = hp.build_ordinal_preprocess(numeric_features, categorical_features)
    return Pipeline([("preprocess", pre),
                     ("model", LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05,
                                              subsample=0.85, colsample_bytree=0.85,
                                              class_weight="balanced", n_jobs=-1,
                                              random_state=42, verbosity=-1))])

X = parts["train"][features]; y = parts["train"][TARGET]
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
m = build_lgbm(); t = time.time(); m.fit(X_tr, y_tr)
val_pred = m.predict(X_va)
f1 = f1_score(y_va, val_pred, average="macro")
print(f"★ 현실 Macro-F1 = {f1:.4f}  (fit {time.time()-t:.0f}s)")
rep = pd.DataFrame(classification_report(y_va, val_pred, output_dict=True, zero_division=0)).T
rep[["precision", "recall", "f1-score", "support"]].round(3)""")

code(r"""# 등급별 F1 막대그래프
g = [i for i in rep.index if i not in ("accuracy", "macro avg", "weighted avg")]
f1s = rep.loc[g, "f1-score"].sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(f1s.index, f1s.values, color="steelblue")
ax.axvline(f1, color="red", ls="--", label=f"Macro-F1={f1:.3f}")
ax.set_xlabel("F1"); ax.set_title("등급별 F1 (test 호환 feature 만)"); ax.legend()
plt.tight_layout(); plt.show()""")

md("## 4. 전체 train 재학습 → test 예측 → 제출 파일 생성\n검증셋의 `LAST_GRADE` 만 채워 원본 13개 컬럼 그대로 저장 (UTF-8 BOM).")
code(r"""final = build_lgbm()
final.fit(X, y)
test_pred = final.predict(parts["test"][features])

submission = test_raw.copy()
submission[TARGET] = test_pred
submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")  # UTF-8 BOM
print("저장:", SUBMISSION_PATH, submission.shape)
print(submission[TARGET].value_counts().to_string())
submission.head()""")

md(r"""## 5. 결론 & 다음 버전(v6) 아이디어

- **현실 Macro-F1 ≈ 0.21** (80만 행 기준). 도체 측정값이 없으니 상한이 낮고, 모든 참가자가 동일 조건.
- 잘 맞히는 등급: `등외`(F1 0.67), `3B`(0.44) — 저체중·고령과 연결돼 `WEIGHT`/`AGE` 로 포착됨.
- 못 맞히는 등급: `1B`/`1+B`/`1C` 등 — 근내지방도(`INSFAT`)가 없어 1++/1+/1 미세 구분이 어려움.

### v6 후보 (큰 변경 시 새 노트북)
1. `class_weight` 튜닝 — 'balanced' 가 다수 등급(1B/1+B) 재현율을 0.04까지 떨어뜨림. 가중치 조정/threshold 보정.
2. 타깃 단순화 실험 — 육질(1++/1+/1/2/3)·육량(A/B/C) 분리 예측 후 결합.
3. 농장 단위 과거 등급 분포(target encoding, fold 내부) — 농장별 생산성 경향 반영.
4. 전체 train(`TRAIN_N=None`)으로 재학습.

> 단, 점수는 20점이므로 과투자 금지. 핵심은 정성 80점(생산성 분석 노트북).
""")

nb.cells = cells
nb.metadata = {"language_info": {"name": "python"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}
with open("model_v5.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("model_v5.ipynb written:", len(cells), "cells")
