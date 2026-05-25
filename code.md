# 파일 불러오기
```
import pandas as pd
import numpy as np
```
```
weather = pd.read_csv(
    "hanwoo_weather.csv")

train = pd.read_csv(
    "hanwoo_train.csv")

lineage = pd.read_csv(
    "hanwoo_lineage.csv")

area = pd.read_csv(
    "hanwoo_area.csv")
death = pd.read_csv(
    "hanwoo_death.csv")
```

```
print(weather.columns)
print(train.columns)
print(lineage.columns)
print(area.columns)
print(death.columns)
```

# 결측치 확인
```
datasets = {
    "weather": weather, 
    "train": train, 
    "lineage": lineage, 
    "area": area, 
    "death": death
}

for name, df in datasets.items(): # 결측치 확인용

    summary = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "minus99_count": (df == -99).sum()
    })

    print(f"\n================ {name} ================")
    print(f"행: {df.shape[0]}, 열: {df.shape[1]}")
    print(summary)
```
```
print((lineage == -99).sum()) # lineage는 문자형 데이터라 -99를 NaN 처리 해야할 듯, 중요한 변수는 아님
print((lineage == "-99").sum())
```

# train 데이터와 weather 데이터를 연결(merge)
```
df = pd.merge( # train과 weather를 merge
    train,
    weather,
    left_on=["stn", "JUDGE_DATE"],
    right_on=["stn", "date"],
    how="left"
)
```
```
df.shape # 결과 확인, stn 컬럼은 weather와 train 데이터 모두 있어서 총 컬럼은 30개가 아닌 29개
```

```
df.head() # train 쪽 : 지역, 성별 체중, 등급, 개체 정보
# weather 쪽, 최고기온, 최저기온, 강수량, 습도, 풍속
```

```
df.isnull().sum()
```


### 결측치 처리
```
# train 데이터와 weather 데이터를 연결(merge) 후 가장 먼저 할 일!
df = pd.merge(train, weather, left_on=["stn", "JUDGE_DATE"], right_on=["stn", "date"], how="left")

# 1. -99 결측치 우선 처리
df = df.replace([-99, "-99", -99.0], np.nan) 
```

# 파생변수

```
df["temp_avg"] = ( # 평균기온
    df["ta_max"] +
    df["ta_min"]
) / 2
df["temp_gap"] = ( # 일교차
    df["ta_max"] -
    df["ta_min"]
)
df["heatwave"] = ( # 폭염 여부
    df["ta_max"] >= 33
).astype(int) 
df["coldwave"] = ( # 한파 여부
    df["ta_min"] <= -10
).astype(int)
```


# 종속변수 (Y) 선택
```
# WGRADE 고유값 확인 # 육량등급(A/B/C)
print(df["WGRADE"].unique())
```


# LAST_GRADE 고유값 확인 # 육질등급(1++, 1+, 1, 2, 3)
print(df["LAST_GRADE"].unique())
print(df["WGRADE"].value_counts()) # 결측치 5363개 존재(-99.0)

print(df["LAST_GRADE"].value_counts()) # 결측치 거의 안 보임 # LAST_GRADE를 Y로 사용
# 육질등급별 주요 변수 평균 비교
# 육질등급별 체중
df.groupby("LAST_GRADE")["WEIGHT"].mean()
# 육질등급별 평균기온
df.groupby("LAST_GRADE")["temp_avg"].mean()
# 육질등급별 등지방두께(BACKFAT)
df.groupby("LAST_GRADE")["BACKFAT"].mean()
# 등심단면적(REA)
df.groupby("LAST_GRADE")["REA"].mean()
# 육량지수(WINDEX) 중요
df.groupby("LAST_GRADE")["WINDEX"].mean()
# 나이
df.groupby("LAST_GRADE")["AGE"].mean()
# 기상 변수 비교
# 육질등급별 평균 강수량
df.groupby("LAST_GRADE")["rn_day"].mean()
# 육질등급별 평균 풍속
df.groupby("LAST_GRADE")["ws_davg"].mean()
# 육질등급별 일교차
df.groupby("LAST_GRADE")["temp_gap"].mean()
# 시각화
# 육질등급별 체중 그래프
import matplotlib.pyplot as plt

df.groupby("LAST_GRADE")["WEIGHT"].mean().plot(
    kind="bar"
)

plt.xlabel("LAST_GRADE")
plt.ylabel("Average Weight")

plt.show()
# 육질등급별 평균기온 그래프
df.groupby("LAST_GRADE")["temp_avg"].mean().plot(
    kind="bar"
)

plt.xlabel("LAST_GRADE")
plt.ylabel("Average Temperature")

plt.show()
# 상관관계 분석
numeric_cols = [
    "WEIGHT",
    "BACKFAT",
    "REA",
    "WINDEX",
    "AGE",
    "temp_avg",
    "temp_gap",
    "rn_day",
    "ws_davg"
]

corr_matrix = df[numeric_cols].corr()

print(corr_matrix)
# 히트맵 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))

plt.imshow(corr_matrix)

plt.colorbar()

plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=90
)

plt.yticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns
)

plt.show()
# 폭염 영향 분석 (최고 기온이 33도 이상이면 1, 아니면 0으로 설정)
# 폭염 여부별 평균 체중
df.groupby("heatwave")["WEIGHT"].mean()
# 폭염 여부별 평균 육량지수
df.groupby("heatwave")["WINDEX"].mean()
# 폭염 여부별 평균 등급 분포
pd.crosstab(
    df["heatwave"],
    df["LAST_GRADE"]
)
# 산점도
# -99를 NaN 처리
df = df.replace(-99, np.nan) # 숫자형
df = df.replace("-99", np.nan) # 문자형
# 파생변수 재생성 (평균기온)
df["temp_avg"] = (
    df["ta_max"] +
    df["ta_min"]
) / 2
# 일교차
df["temp_gap"] = (
    df["ta_max"] -
    df["ta_min"]
)
# 체중 vs 평균기온
plt.figure(figsize=(6, 4))

plt.scatter(
    df["temp_avg"],
    df["WEIGHT"],
    alpha=0.3
)

plt.xlabel("Average Temperature")
plt.ylabel("Weight")

plt.show()
# 육량지수 vs 평균기온
plt.figure(figsize=(6, 4))

plt.scatter(
    df["temp_avg"], 
    df["WINDEX"],
    alpha=0.3
)

plt.xlabel("Average Temperature")
plt.ylabel("WINDEX")

plt.show()
# 지역별 분석
# 시도별 평균 체중
df.groupby("sido")["WEIGHT"].mean().sort_values()
# 시도별 평균 육량지수
df.groupby("sido")["WINDEX"].mean().sort_values()
# 계절 변수
# 날짜형 변환
df["JUDGE_DATE"] = pd.to_datetime(
    df["JUDGE_DATE"]
)
# 월 추출
df["month"] = df["JUDGE_DATE"].dt.month
# 계절 생성
def get_season(month):

    if month in [3, 4, 5]:
        return "spring"

    elif month in [6, 7, 8]:
        return "summer"

    elif month in [9, 10, 11]:
        return "fall"

    else:
        return "winter"

df["season"] = df["month"].apply(get_season)
# 확인 코드
df[["JUDGE_DATE", "month", "season"]].head()
# 계절별 평균 체중
df.groupby("season")["WEIGHT"].mean()
# 계절별 평균 육량지수
df.groupby("season")["WINDEX"].mean()
# 계절별 평균 기온
df.groupby("season")["temp_avg"].mean()
# area 데이터 연결(train + weather 데이터에 area도 merge)
# 추가로 해야할 듯한 부분
df = pd.merge( # area는 농가 규모
    df,
    area,
    on="FARM_UNIQUE_NO",
    how="left"
)
df[["FARM_UNIQUE_NO", "C2025", "AREA"]].head()
# 사육밀도 생성 # 밀도 면적에 따른 소 마리수 체크
df["density"] = ( 
    df["C2025"] / df["AREA"] 
)
df["density"].isnull().sum()
# 결측치(NaN) 제외한 데이터만 따로 사용
density_df = df.dropna( 
    subset=["density"]
)
density_df.groupby(
    "LAST_GRADE"
)["density"].mean()
density_df["density"].describe()
