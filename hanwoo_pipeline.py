"""
hanwoo_pipeline.py
==================
한우 도체 최종등급(LAST_GRADE) 예측을 위한 공통 전처리/모델링 모듈.

model_v1_1 / model_v1_1_model_compare / model_v3 에 복사-붙여넣기로 흩어져 있던
전처리 함수를 한 곳으로 모았다. model_v4.ipynb 는 이 모듈만 import 해서 쓴다.

핵심 개선점 (이전 버전 대비)
---------------------------------
1. 코드 중복 제거 : 모든 전처리/모델 빌드 로직을 이 모듈 하나로 통합.
2. 경로 자동 탐지 : BASE_DIR 하드코딩 제거. flat / 하위폴더 레이아웃 모두 지원.
3. 타깃 누수 차단 : WGRADE 는 LAST_GRADE 의 육량등급(A/B/C) 글자와 96% 동일하므로
                    기본적으로 feature 에서 제외 (Config.use_wgrade_feature=False).
4. 기상 관측소 좌표 파일이 없으면 공간 보간을 자동으로 건너뜀 (graceful).
5. '수'(수소) 는 표본이 0.6% 로 매우 적어 분리 모델에서 제외하고 통합 모델 예측으로 fallback.
6. LightGBM / XGBoost 를 선택적으로 사용 (미설치 시 자동 skip).
7. 모든 group 통계 결측 보간은 train split 에서만 fit → valid/test 누수 차단.
8. 기상 rolling feature 는 shift(1) 후 계산 → 도축일 당일/미래 정보 누수 차단.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# ----------------------------------------------------------------------------
# 선택적 의존성 (미설치 시 자동 skip)
# ----------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


# ============================================================================
# 설정
# ============================================================================
@dataclass
class Config:
    base_dir: Path | None = None          # None 이면 자동 탐지
    use_sample: bool = True
    sample_n: int = 250_000
    random_state: int = 42
    target: str = "LAST_GRADE"

    # ---- feature 정책 ----
    use_carcass_features: bool = True     # 도체 측정값(BACKFAT/REA/INSFAT...). 과제 정의상 사용.
    use_wgrade_feature: bool = False      # ★ 누수: LAST_GRADE 육량등급과 96% 동일 → 기본 제외
    use_weather_features: bool = True
    use_area_features: bool = True
    use_lineage_features: bool = True
    use_death_features: bool = True
    weather_windows: tuple = (30, 90, 180)

    # ---- 학습 전략 ----
    split_sexes: tuple = ("암", "거세")   # '수'는 너무 적어 분리 제외 → unified fallback
    test_size: float = 0.15
    valid_size: float = 0.15

    # ---- 모델 하이퍼파라미터 ----
    rf_n_estimators: int = 120
    rf_max_depth: int = 20
    rf_min_samples_leaf: int = 2
    et_n_estimators: int = 200
    xgb_n_estimators: int = 400
    lgbm_n_estimators: int = 500

    output_dirname: str = "model_v4_outputs"


MISSING_TOKENS = [-99, -99.0, "-99", "-99.0", "", " "]
NUMERIC_BASE_COLS = [
    "WEIGHT", "BACKFAT", "REA", "WINDEX", "INSFAT", "YUKSAK",
    "FATSAK", "TISSUE", "GROWTH", "COST_AMT", "AGE",
]
CARCASS_NUMERIC_COLS = ["BACKFAT", "REA", "WINDEX", "INSFAT", "YUKSAK", "FATSAK", "TISSUE", "GROWTH"]


# ============================================================================
# 경로 / 로딩
# ============================================================================
def detect_base_dir(explicit=None) -> Path:
    """hanwoo_train.csv 가 있는 디렉터리를 자동으로 찾는다."""
    if explicit is not None:
        return Path(explicit)
    here = Path.cwd()
    candidates = [here, here / "hanwoo", here.parent, here.parent / "hanwoo"]
    for c in candidates:
        if (c / "hanwoo_train.csv").exists() or (c / "hanwoo_train" / "hanwoo_train.csv").exists():
            return c
    return here


def _resolve(base: Path, *names) -> Path | None:
    """flat 레이아웃과 하위폴더 레이아웃을 모두 시도해서 첫 번째로 존재하는 경로 반환."""
    for name in names:
        p = base / name
        if p.exists():
            return p
    return None


def read_csv_safe(path, **kwargs):
    """인코딩을 utf-8-sig → utf-8 → cp949 순으로 시도."""
    path = Path(path)
    last_error = None
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def load_raw(cfg: Config) -> dict:
    """원본 데이터프레임들을 dict 로 로드. 없는 파일은 None."""
    base = detect_base_dir(cfg.base_dir)
    nrows = cfg.sample_n if cfg.use_sample else None

    train_path = _resolve(base, "hanwoo_train.csv", "hanwoo_train/hanwoo_train.csv")
    if train_path is None:
        raise FileNotFoundError(f"hanwoo_train.csv 를 {base} 에서 찾을 수 없습니다.")

    data = {
        "base_dir": base,
        "train": read_csv_safe(train_path, nrows=nrows),
        "area": None, "death": None, "lineage": None, "weather": None, "station": None,
    }

    if cfg.use_area_features:
        p = _resolve(base, "hanwoo_area.csv")
        data["area"] = read_csv_safe(p) if p else None
    if cfg.use_death_features:
        p = _resolve(base, "hanwoo_death.csv")
        data["death"] = read_csv_safe(p) if p else None
    if cfg.use_lineage_features:
        p = _resolve(base, "hanwoo_lineage.csv", "hanwoo_lineage/hanwoo_lineage.csv")
        # 혈통은 CATTLE_NO, KPN_NO 만 필요
        data["lineage"] = read_csv_safe(p, usecols=["CATTLE_NO", "KPN_NO"]) if p else None
    if cfg.use_weather_features:
        p = _resolve(base, "hanwoo_weather.csv")
        data["weather"] = read_csv_safe(p) if p else None
        # 관측소 좌표 파일(위도/경도)은 이 환경엔 없을 수 있음 → 있으면만 사용
        sp = next(iter(base.glob("stn_station*.csv")), None)
        data["station"] = read_csv_safe(sp) if sp else None

    return data


# ============================================================================
# 기본 정제 / 날짜 feature
# ============================================================================
def normalize_missing(df):
    return df.copy().replace(MISSING_TOKENS, np.nan)


def add_date_features(df):
    out = df.copy()
    out["ABATT_DATE"] = pd.to_datetime(out["ABATT_DATE"], errors="coerce")
    out["JUDGE_DATE"] = pd.to_datetime(out["JUDGE_DATE"], errors="coerce")
    out["BIRTH_YMD"] = pd.to_datetime(out["BIRTH_YMD"].astype("string"), format="%Y%m%d", errors="coerce")
    out["abatt_year"] = out["ABATT_DATE"].dt.year.astype("Int64")
    out["abatt_month"] = out["ABATT_DATE"].dt.month.astype("Int64")
    out["abatt_ym"] = out["ABATT_DATE"].dt.to_period("M").astype("string")
    out["judge_delay_days"] = (out["JUDGE_DATE"] - out["ABATT_DATE"]).dt.days
    out["rearing_days"] = (out["ABATT_DATE"] - out["BIRTH_YMD"]).dt.days
    month = out["abatt_month"].astype("float")
    conditions = [month.isin([3, 4, 5]), month.isin([6, 7, 8]), month.isin([9, 10, 11])]
    out["abatt_season"] = np.select(conditions, ["spring", "summer", "fall"], default="winter")
    return out


def prepare_train(train_df) -> pd.DataFrame:
    train = normalize_missing(train_df)
    for col in NUMERIC_BASE_COLS + ["stn"]:
        if col in train.columns:
            train[col] = pd.to_numeric(train[col], errors="coerce")
    return add_date_features(train)


# ============================================================================
# 농장 면적 / 사육두수 feature
# ============================================================================
def prepare_area_table(area_df):
    if area_df is None:
        return None
    out = normalize_missing(area_df)
    for col in ["C2023", "C2024", "C2025", "AREA"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out.loc[out["AREA"] <= 0, "AREA"] = np.nan
    return out.groupby("FARM_UNIQUE_NO", as_index=False)[["C2023", "C2024", "C2025", "AREA"]].median()


def merge_area_features(df, area_table):
    if area_table is None:
        out = df.copy()
        out["cow_year"] = np.nan
        out["AREA"] = np.nan
        return out
    out = df.merge(area_table, on="FARM_UNIQUE_NO", how="left")
    year = out["abatt_year"].astype("float")
    out["cow_year"] = np.select(
        [year.eq(2023), year.eq(2024), year.eq(2025)],
        [out["C2023"], out["C2024"], out["C2025"]],
        default=np.nan,
    )
    return out.drop(columns=["C2023", "C2024", "C2025"])


# ============================================================================
# 폐사 feature (농장 단위 집계)
# ============================================================================
def prepare_death_table(death_df):
    if death_df is None:
        return None
    d = normalize_missing(death_df)
    return d.groupby("FARM_UNIQUE_NO").size().reset_index(name="farm_death_count")


def merge_death_features(df, death_table):
    out = df.copy()
    if death_table is None:
        out["farm_death_count"] = 0
        return out
    out = out.merge(death_table, on="FARM_UNIQUE_NO", how="left")
    out["farm_death_count"] = out["farm_death_count"].fillna(0).astype("int32")
    return out


# ============================================================================
# 혈통 KPN feature (frequency 는 imputer 단계에서 train 기준으로 계산)
# ============================================================================
def prepare_lineage_table(lineage_df):
    if lineage_df is None:
        return None
    out = normalize_missing(lineage_df[["CATTLE_NO", "KPN_NO"]].copy())
    return out.drop_duplicates("CATTLE_NO", keep="first")


def merge_lineage_features(df, lineage_table):
    if lineage_table is None:
        out = df.copy()
        out["KPN_NO"] = np.nan
        return out
    return df.merge(lineage_table, on="CATTLE_NO", how="left")


# ============================================================================
# 기상 일별 정제 + rolling feature (leakage-safe)
# ============================================================================
def haversine_np(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_nearest_station_map(station_df, k=8):
    """관측소 좌표가 있으면 인접 관측소 맵을 만든다. 없으면 빈 dict."""
    if station_df is None or not {"stn", "위도", "경도"}.issubset(station_df.columns):
        return {}
    coord = station_df[["stn", "위도", "경도"]].dropna().drop_duplicates("stn").copy()
    coord["stn"] = pd.to_numeric(coord["stn"], errors="coerce")
    coord = coord.dropna(subset=["stn"])
    coord["stn"] = coord["stn"].astype(int)
    values = coord[["stn", "위도", "경도"]].to_numpy()
    nearest = {}
    for stn, lat, lon in values:
        dist = haversine_np(lat, lon, values[:, 1], values[:, 2])
        order = np.argsort(dist)
        nearest[int(stn)] = [int(values[i, 0]) for i in order if int(values[i, 0]) != int(stn)][:k]
    return nearest


def fill_station_month_then_global(weather_df, value_cols):
    out = weather_df.copy()
    out["month"] = out["date"].dt.month
    for col in value_cols:
        out[col] = out[col].fillna(out.groupby(["stn", "month"])[col].transform("median"))
        out[col] = out[col].fillna(out.groupby("month")[col].transform("median"))
        out[col] = out[col].fillna(out[col].median())
    return out.drop(columns=["month"])


def prepare_weather_daily(weather_df):
    if weather_df is None:
        return None
    out = normalize_missing(weather_df)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["stn"] = pd.to_numeric(out["stn"], errors="coerce").astype("Int64")
    weather_cols = ["ta_max", "rn_day", "ta_min", "rhm_avg", "ws_davg"]
    for col in weather_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_was_missing"] = out[col].isna().astype("int8")
    # 강수량은 결측을 0 으로 보지 않고 보간 대상에 포함 (원자료가 -99 = 미관측)
    out = out.sort_values(["stn", "date"]).reset_index(drop=True)
    # 연속형 기온/풍속은 관측소별 짧은 선형보간
    for col in ["ta_max", "ta_min", "ws_davg"]:
        out[col] = out.groupby("stn")[col].transform(
            lambda s: s.interpolate(method="linear", limit=3, limit_area="inside")
        )
    # 관측소-월 중앙값 → 월 중앙값 → 전체 중앙값 fallback
    out = fill_station_month_then_global(out, weather_cols)
    # 기본 품질 검사
    swapped = out["ta_max"] < out["ta_min"]
    if swapped.any():
        tmp = out.loc[swapped, "ta_max"].copy()
        out.loc[swapped, "ta_max"] = out.loc[swapped, "ta_min"]
        out.loc[swapped, "ta_min"] = tmp
    out["rhm_avg"] = out["rhm_avg"].clip(0, 100)
    out["rn_day"] = out["rn_day"].clip(lower=0)
    out["ws_davg"] = out["ws_davg"].clip(lower=0)
    return out


def build_weather_rolling_features(weather_daily, windows):
    if weather_daily is None:
        return None
    w = weather_daily.sort_values(["stn", "date"]).copy()
    w["ta_avg"] = (w["ta_max"] + w["ta_min"]) / 2
    # THI(온습도지수) — 가축 더위 스트레스 지표
    w["thi"] = (1.8 * w["ta_avg"] + 32) - (0.55 - 0.0055 * w["rhm_avg"]) * (1.8 * w["ta_avg"] - 26)
    w["rain_day"] = (w["rn_day"] > 0).astype("int8")
    w["hot_day"] = (w["ta_max"] >= 30).astype("int8")
    w["heatwave_day"] = (w["ta_max"] >= 33).astype("int8")
    w["tropical_night"] = (w["ta_min"] >= 25).astype("int8")
    w["thi72_day"] = (w["thi"] >= 72).astype("int8")
    w["thi78_day"] = (w["thi"] >= 78).astype("int8")
    base_cols = ["ta_max", "ta_min", "ta_avg", "rhm_avg", "ws_davg", "thi"]
    sum_cols = ["rn_day", "rain_day", "hot_day", "heatwave_day", "tropical_night", "thi72_day", "thi78_day"]
    pieces = []
    for stn, g in w.groupby("stn", sort=False):
        g = g.set_index("date").sort_index()
        shifted = g[base_cols + sum_cols].shift(1)   # ★ 도축일 당일/미래 누수 차단
        feat = pd.DataFrame(index=g.index)
        feat["stn"] = stn
        for window in windows:
            min_periods = max(7, int(window * 0.6))
            roll = shifted.rolling(window=window, min_periods=min_periods)
            for col in base_cols:
                feat[f"{col}_mean_{window}d"] = roll[col].mean()
            for col in sum_cols:
                feat[f"{col}_sum_{window}d"] = roll[col].sum()
        pieces.append(feat.reset_index())
    return pd.concat(pieces, ignore_index=True).rename(columns={"date": "ABATT_DATE"})


def merge_weather_features(df, weather_features_df):
    if weather_features_df is None:
        return df.copy()
    out = df.copy()
    out["stn"] = pd.to_numeric(out["stn"], errors="coerce").astype("Int64")
    return out.merge(weather_features_df, on=["stn", "ABATT_DATE"], how="left")


# ============================================================================
# train 기준 group 결측 보간 (leakage-safe)
# ============================================================================
def _key_frame(df, group_cols):
    keys = df[list(group_cols)].copy()
    for col in group_cols:
        keys[col] = keys[col].astype("string").fillna("__NA__")
    return keys


def fit_group_stat(df, value_col, group_cols, kind="median", numeric=True):
    keys = _key_frame(df, group_cols)
    values = (pd.to_numeric(df[value_col], errors="coerce") if numeric
              else df[value_col].astype("string").replace("__NA__", np.nan))
    work = keys.copy()
    work["_value"] = values
    work = work.dropna(subset=["_value"])
    if work.empty:
        return pd.DataFrame(columns=list(group_cols) + ["_fill"])
    if kind == "median":
        return work.groupby(list(group_cols), as_index=False)["_value"].median().rename(columns={"_value": "_fill"})
    if kind == "mode":
        return (work.groupby(list(group_cols), as_index=False)["_value"]
                .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
                .rename(columns={"_value": "_fill"}))
    raise ValueError(kind)


def fit_fallback_imputer(train_df, value_col, group_chain, kind="median", numeric=True):
    if numeric:
        observed = pd.to_numeric(train_df[value_col], errors="coerce").dropna()
        global_fill = observed.median() if len(observed) else 0.0
    else:
        observed = train_df[value_col].dropna().astype("string")
        global_fill = observed.mode().iloc[0] if not observed.mode().empty else "UNKNOWN"
    return {
        "value_col": value_col,
        "numeric": numeric,
        "global_fill": global_fill,
        "tables": [(cols, fit_group_stat(train_df, value_col, cols, kind=kind, numeric=numeric))
                   for cols in group_chain],
    }


def apply_fallback_imputer(df, spec):
    out = df.copy()
    col = spec["value_col"]
    out[col] = (pd.to_numeric(out[col], errors="coerce") if spec["numeric"]
                else out[col].astype("string").replace("__NA__", np.nan))
    for group_cols, table in spec["tables"]:
        missing = out[col].isna()
        if not missing.any() or table.empty:
            continue
        keys = _key_frame(out.loc[missing], group_cols)
        fill = keys.merge(table, on=list(group_cols), how="left")["_fill"].to_numpy()
        out.loc[missing, col] = out.loc[missing, col].fillna(pd.Series(fill, index=out.index[missing]))
    out[col] = out[col].fillna(spec["global_fill"])
    return out


def fit_and_apply_imputers(parts, cfg: Config):
    """parts = {'train','valid','test'}. train 기준으로 fit 후 모두에 apply."""
    fitted = {}
    out = {k: v.copy() for k, v in parts.items()}

    for key in out:
        out[key]["carcass_missing_count"] = out[key][CARCASS_NUMERIC_COLS].isna().sum(axis=1)

    # WGRADE (feature 로 쓰든 안 쓰든 보간은 해 둔다; 정책은 feature 선택 단계에서)
    spec = fit_fallback_imputer(out["train"], "WGRADE",
                                [["JUDGE_SEX", "AGE", "abatt_year"], ["JUDGE_SEX", "AGE"], ["JUDGE_SEX"]],
                                kind="mode", numeric=False)
    fitted["WGRADE"] = spec
    for key in out:
        out[key] = apply_fallback_imputer(out[key], spec)

    for col in CARCASS_NUMERIC_COLS:
        spec = fit_fallback_imputer(out["train"], col,
                                    [["JUDGE_SEX", "AGE", "abatt_year"], ["JUDGE_SEX", "AGE"], ["JUDGE_SEX"]],
                                    kind="median", numeric=True)
        fitted[col] = spec
        for key in out:
            out[key] = apply_fallback_imputer(out[key], spec)

    for key in out:
        out[key]["growth_ge8"] = (pd.to_numeric(out[key]["GROWTH"], errors="coerce") >= 8).astype("int8")

    spec = fit_fallback_imputer(out["train"], "COST_AMT",
                                [["JUDGE_SEX", "abatt_ym", "WGRADE"], ["JUDGE_SEX", "abatt_ym"], ["JUDGE_SEX"]],
                                kind="median", numeric=True)
    fitted["COST_AMT"] = spec
    for key in out:
        out[key] = apply_fallback_imputer(out[key], spec)

    for col in ["cow_year", "AREA"]:
        if col in out["train"].columns:
            spec = fit_fallback_imputer(out["train"], col, [["sigungu"], ["sido"]], kind="median", numeric=True)
            fitted[col] = spec
            for key in out:
                out[key] = apply_fallback_imputer(out[key], spec)

    for key in out:
        out[key]["log_cow_year"] = np.log1p(pd.to_numeric(out[key]["cow_year"], errors="coerce").clip(lower=0))
        out[key]["log_area"] = np.log1p(pd.to_numeric(out[key]["AREA"], errors="coerce").clip(lower=0))

    # KPN frequency : train split 에서만 집계
    if "KPN_NO" in out["train"].columns:
        kpn_freq_map = out["train"]["KPN_NO"].dropna().astype("string").value_counts().to_dict()
    else:
        kpn_freq_map = {}
    fitted["kpn_freq_map"] = kpn_freq_map
    for key in out:
        if "KPN_NO" not in out[key].columns:
            out[key]["KPN_NO"] = np.nan
        kpn = out[key]["KPN_NO"].astype("string")
        out[key]["kpn_known"] = kpn.notna().astype("int8")
        out[key]["kpn_freq"] = kpn.map(kpn_freq_map).fillna(0).astype("int32")
        out[key]["log_kpn_freq"] = np.log1p(out[key]["kpn_freq"])

    return out, fitted


# ============================================================================
# 전체 feature 파이프라인 (split 전까지)
# ============================================================================
def build_feature_frame(cfg: Config, data: dict):
    """원본 dict → split 된 parts(dict) 반환. (train 기준 보간 포함)"""
    train = prepare_train(data["train"]).dropna(subset=[cfg.target])

    # split 먼저 (보간을 train 기준으로 하기 위해)
    rs = cfg.random_state
    temp_frac = cfg.valid_size + cfg.test_size
    train_idx, temp_idx = train_test_split(
        train.index, test_size=temp_frac, random_state=rs, stratify=train[cfg.target])
    rel_test = cfg.test_size / temp_frac
    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=rel_test, random_state=rs, stratify=train.loc[temp_idx, cfg.target])
    parts = {
        "train": train.loc[train_idx].copy(),
        "valid": train.loc[valid_idx].copy(),
        "test": train.loc[test_idx].copy(),
    }

    # 농장/폐사/혈통 merge (행 1:1 유지)
    area_table = prepare_area_table(data.get("area"))
    death_table = prepare_death_table(data.get("death"))
    lineage_table = prepare_lineage_table(data.get("lineage"))
    for key in parts:
        parts[key] = merge_area_features(parts[key], area_table)
        parts[key] = merge_death_features(parts[key], death_table)
        parts[key] = merge_lineage_features(parts[key], lineage_table)

    # 기상 rolling merge
    weather_features = None
    if cfg.use_weather_features:
        weather_daily = prepare_weather_daily(data.get("weather"))
        weather_features = build_weather_rolling_features(weather_daily, cfg.weather_windows)
        if weather_features is not None:
            weather_features["stn"] = pd.to_numeric(weather_features["stn"], errors="coerce").astype("Int64")
    for key in parts:
        parts[key] = merge_weather_features(parts[key], weather_features)

    # train 기준 결측 보간
    parts, fitted = fit_and_apply_imputers(parts, cfg)
    return parts, fitted


def get_feature_columns(cfg: Config, df) -> tuple[list, list]:
    numeric_features = ["WEIGHT", "AGE", "judge_delay_days", "rearing_days", "carcass_missing_count"]
    if cfg.use_carcass_features:
        numeric_features += ["BACKFAT", "REA", "WINDEX", "INSFAT", "YUKSAK", "FATSAK",
                             "TISSUE", "GROWTH", "COST_AMT", "growth_ge8"]
    if cfg.use_area_features:
        numeric_features += ["cow_year", "log_cow_year", "log_area"]
    if cfg.use_death_features:
        numeric_features += ["farm_death_count"]
    if cfg.use_lineage_features:
        numeric_features += ["kpn_known", "kpn_freq", "log_kpn_freq"]
    if cfg.use_weather_features:
        numeric_features += [c for c in df.columns if any(c.endswith(f"_{w}d") for w in cfg.weather_windows)]
    numeric_features = [c for c in numeric_features if c in df.columns]

    categorical_features = ["sido", "sigungu", "stn", "abatt_month", "abatt_season"]
    if cfg.use_wgrade_feature:
        categorical_features = ["WGRADE"] + categorical_features  # ★ 누수 주의
    categorical_features = [c for c in categorical_features if c in df.columns]
    return numeric_features, categorical_features


# ============================================================================
# 모델 빌더
# ============================================================================
def _make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_onehot_preprocess(numeric_features, categorical_features, scale_numeric=False):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler(with_mean=False)))
    cat = Pipeline([
        ("to_string", FunctionTransformer(
            lambda x: x.astype("string").fillna("UNKNOWN").astype(str), feature_names_out="one-to-one")),
        ("onehot", _make_onehot()),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric_features),
        ("cat", cat, categorical_features),
    ], remainder="drop")


def build_ordinal_preprocess(numeric_features, categorical_features):
    """HistGB / LightGBM 용 (one-hot 대신 ordinal)."""
    cat = Pipeline([
        ("to_string", FunctionTransformer(
            lambda x: x.astype("string").fillna("UNKNOWN").astype(str), feature_names_out="one-to-one")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1, dtype=np.float64)),
    ])
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", cat, categorical_features),
    ], remainder="drop")


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost 처럼 정수 라벨이 필요한 모델용 래퍼."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_enc)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, X):
        pred = np.asarray(self.estimator_.predict(X)).astype(int)
        return self.label_encoder_.inverse_transform(pred)


def available_models(cfg: Config) -> list:
    models = ["random_forest", "extra_trees", "hist_gb"]
    if XGBClassifier is not None:
        models.append("xgboost")
    if LGBMClassifier is not None:
        models.append("lightgbm")
    return models


def build_model(model_name, cfg: Config, numeric_features, categorical_features, random_state=None):
    rs = cfg.random_state if random_state is None else random_state

    if model_name == "random_forest":
        return Pipeline([
            ("preprocess", build_onehot_preprocess(numeric_features, categorical_features)),
            ("model", RandomForestClassifier(
                n_estimators=cfg.rf_n_estimators, max_depth=cfg.rf_max_depth,
                min_samples_leaf=cfg.rf_min_samples_leaf, random_state=rs,
                n_jobs=-1, class_weight="balanced_subsample")),
        ])
    if model_name == "extra_trees":
        return Pipeline([
            ("preprocess", build_onehot_preprocess(numeric_features, categorical_features)),
            ("model", ExtraTreesClassifier(
                n_estimators=cfg.et_n_estimators, max_depth=None,
                min_samples_leaf=cfg.rf_min_samples_leaf, random_state=rs,
                n_jobs=-1, class_weight="balanced_subsample")),
        ])
    if model_name == "hist_gb":
        return Pipeline([
            ("preprocess", build_ordinal_preprocess(numeric_features, categorical_features)),
            ("model", HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.07, max_depth=None,
                l2_regularization=1.0, random_state=rs)),
        ])
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost 미설치")
        xgb = XGBClassifier(
            n_estimators=cfg.xgb_n_estimators, max_depth=7, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, objective="multi:softmax",
            eval_metric="mlogloss", tree_method="hist", n_jobs=-1,
            random_state=rs, verbosity=0)
        return Pipeline([
            ("preprocess", build_onehot_preprocess(numeric_features, categorical_features)),
            ("model", LabelEncodedClassifier(xgb)),
        ])
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm 미설치")
        lgbm = LGBMClassifier(
            n_estimators=cfg.lgbm_n_estimators, num_leaves=63, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, class_weight="balanced",
            n_jobs=-1, random_state=rs, verbosity=-1)
        return Pipeline([
            ("preprocess", build_ordinal_preprocess(numeric_features, categorical_features)),
            ("model", lgbm),
        ])
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# 학습/평가 루프
# ============================================================================
def run_experiment(cfg: Config, parts: dict, model_names=None, strategies=("unified", "gender_split"),
                   verbose=True):
    """model_names 별로 unified / gender_split 전략을 학습·평가하고 결과 DataFrame 반환."""
    numeric_features, categorical_features = get_feature_columns(cfg, parts["train"])
    features = numeric_features + categorical_features
    if model_names is None:
        model_names = available_models(cfg)

    X = {k: parts[k][features].copy() for k in parts}
    y = {k: parts[k][cfg.target].copy() for k in parts}

    rows = []
    preds = {"valid": {}, "test": {}}
    trained = {}

    for mi, name in enumerate(model_names):
        if verbose:
            print(f"\n{'='*70}\n[{name}]")
        try:
            unified_valid = unified_test = None
            if "unified" in strategies or "gender_split" in strategies:
                model = build_model(name, cfg, numeric_features, categorical_features,
                                    random_state=cfg.random_state + mi * 100)
                t0 = time.time()
                model.fit(X["train"], y["train"])
                unified_valid = model.predict(X["valid"])
                unified_test = model.predict(X["test"])
                dt = time.time() - t0
                vf = f1_score(y["valid"], unified_valid, average="macro")
                tf = f1_score(y["test"], unified_test, average="macro")
                trained[name] = model
                preds["valid"][f"{name}_unified"] = unified_valid
                preds["test"][f"{name}_unified"] = unified_test
                rows.append(dict(model=name, strategy="unified", valid_macro_f1=vf,
                                 test_macro_f1=tf, seconds=dt, status="ok"))
                if verbose:
                    print(f"  unified      valid={vf:.4f} test={tf:.4f} ({dt:.0f}s)")

            if "gender_split" in strategies:
                vpred = pd.Series(unified_valid, index=X["valid"].index)
                tpred = pd.Series(unified_test, index=X["test"].index)
                t0 = time.time()
                for si, sex in enumerate(cfg.split_sexes):
                    tr_mask = parts["train"]["JUDGE_SEX"].eq(sex)
                    va_mask = parts["valid"]["JUDGE_SEX"].eq(sex)
                    te_mask = parts["test"]["JUDGE_SEX"].eq(sex)
                    if tr_mask.sum() < 100:
                        continue
                    sm = build_model(name, cfg, numeric_features, categorical_features,
                                     random_state=cfg.random_state + mi * 100 + si + 1)
                    sm.fit(X["train"][tr_mask.values], y["train"][tr_mask.values])
                    if va_mask.any():
                        vpred.loc[va_mask.values] = sm.predict(X["valid"][va_mask.values])
                    if te_mask.any():
                        tpred.loc[te_mask.values] = sm.predict(X["test"][te_mask.values])
                dt = time.time() - t0
                vf = f1_score(y["valid"], vpred, average="macro")
                tf = f1_score(y["test"], tpred, average="macro")
                preds["valid"][f"{name}_gender_split"] = vpred.to_numpy()
                preds["test"][f"{name}_gender_split"] = tpred.to_numpy()
                rows.append(dict(model=name, strategy="gender_split", valid_macro_f1=vf,
                                 test_macro_f1=tf, seconds=dt, status="ok"))
                if verbose:
                    print(f"  gender_split valid={vf:.4f} test={tf:.4f} ({dt:.0f}s)")

        except Exception as e:  # pragma: no cover
            rows.append(dict(model=name, strategy="failed", valid_macro_f1=np.nan,
                             test_macro_f1=np.nan, seconds=np.nan, status=repr(e)))
            if verbose:
                print(f"  FAILED: {e!r}")

    comparison = (pd.DataFrame(rows)
                  .sort_values(["test_macro_f1", "valid_macro_f1"], ascending=False)
                  .reset_index(drop=True))
    return comparison, preds, trained, (numeric_features, categorical_features)


def per_class_report(y_true, y_pred):
    rep = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T
    return rep


def per_sex_macro_f1(y_true, y_pred, sex):
    y_true = pd.Series(np.asarray(y_true))
    y_pred = pd.Series(np.asarray(y_pred))
    sex = pd.Series(np.asarray(sex))
    rows = []
    for g in sorted(sex.dropna().unique()):
        m = (sex == g).to_numpy()
        rows.append(dict(sex=g, n=int(m.sum()),
                         macro_f1=f1_score(y_true[m], y_pred[m], average="macro")))
    return pd.DataFrame(rows)
