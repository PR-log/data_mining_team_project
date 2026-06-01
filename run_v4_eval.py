"""model_v4 종합 평가 스크립트 (테스트 + 결과 저장).
실행: python run_v4_eval.py
결과는 model_v4_outputs/ 에 저장된다.
"""
import warnings; warnings.filterwarnings("ignore")
import time, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from hanwoo_pipeline import (
    Config, load_raw, build_feature_frame, run_experiment,
    per_class_report, per_sex_macro_f1, get_feature_columns, available_models,
)

T0 = time.time()
SAMPLE_N = 200_000
cfg = Config(use_sample=True, sample_n=SAMPLE_N, use_wgrade_feature=False)
out_dir = Path(cfg.output_dirname); out_dir.mkdir(exist_ok=True)

print(f"== model_v4 evaluation (sample={SAMPLE_N:,}) ==")
print("available models:", available_models(cfg))

data = load_raw(cfg)
parts, fitted = build_feature_frame(cfg, data)
print("parts:", {k: v.shape for k, v in parts.items()}, f"  ({time.time()-T0:.0f}s)")

# --- 1) 정직한 구성(WGRADE 제외) : RF / LightGBM / XGBoost ---
models = [m for m in ["random_forest", "lightgbm", "xgboost"] if m in available_models(cfg)]
comp, preds, trained, (numf, catf) = run_experiment(
    cfg, parts, model_names=models, strategies=("unified", "gender_split"))
comp.insert(0, "config", "no_wgrade")
print("\n--- no_wgrade comparison ---")
print(comp.to_string())

# --- 2) WGRADE 누수 ablation : LightGBM unified, WGRADE 포함 vs 제외 ---
cfg_wg = Config(use_sample=True, sample_n=SAMPLE_N, use_wgrade_feature=True)
comp_wg, preds_wg, _, _ = run_experiment(
    cfg_wg, parts, model_names=["lightgbm"], strategies=("unified",))
comp_wg.insert(0, "config", "with_wgrade")
print("\n--- WGRADE ablation (lightgbm unified) ---")
print(comp_wg.to_string())

all_comp = pd.concat([comp, comp_wg], ignore_index=True)
all_comp.to_csv(out_dir / "model_v4_metrics.csv", index=False, encoding="utf-8-sig")

# --- best (no_wgrade) 모델로 상세 리포트 ---
best = comp.iloc[0]
best_key = f"{best['model']}_{best['strategy']}"
y_test = parts["test"][cfg.target].to_numpy()
y_pred = preds["test"][best_key]

rep = per_class_report(y_test, y_pred)
rep.to_csv(out_dir / "model_v4_best_per_class.csv", encoding="utf-8-sig")
sex_rep = per_sex_macro_f1(y_test, y_pred, parts["test"]["JUDGE_SEX"])
sex_rep.to_csv(out_dir / "model_v4_best_per_sex.csv", index=False, encoding="utf-8-sig")

labels = sorted(pd.unique(y_test))
cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels),
                  index=[f"true_{x}" for x in labels],
                  columns=[f"pred_{x}" for x in labels])
cm.to_csv(out_dir / "model_v4_best_confusion.csv", encoding="utf-8-sig")

with open(out_dir / "model_v4_features.json", "w", encoding="utf-8") as f:
    json.dump({"numeric": numf, "categorical": catf}, f, ensure_ascii=False, indent=2)

print(f"\n=== BEST (no_wgrade): {best_key}  test_macro_f1={best['test_macro_f1']:.4f} ===")
print("\nper-class F1 (test):")
print(rep[["precision", "recall", "f1-score", "support"]].round(3).to_string())
print("\nper-sex macro-F1 (test):")
print(sex_rep.round(4).to_string())
print(f"\nTOTAL TIME: {time.time()-T0:.0f}s")
print("DONE_V4_EVAL")
