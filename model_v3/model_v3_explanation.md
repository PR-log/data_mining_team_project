# Model V3 설명 자료

이 문서는 `model_v3.ipynb`를 팀원에게 설명하기 위한 요약 자료입니다.

`model_v3`의 목적은 `model_v1_1`에서 정리한 전처리와 feature 생성 방식을 유지한 상태에서, 모델 알고리즘만 바꿔 최종 성능을 비교하는 것입니다.

## 1. 문제 정의

- 예측 대상: `LAST_GRADE`
- 문제 유형: 다중분류
- 평가 지표: Macro-F1
- 검증 방식: `train / validation / test = 70 / 15 / 15`
- split 방식: `LAST_GRADE` 기준 stratified split

Macro-F1을 사용하기 때문에 표본이 많은 등급만 잘 맞히는 것보다, `등외`, `3A`, `3B`, `3C`처럼 표본이 적은 등급도 골고루 맞히는 것이 중요합니다.

## 2. 전체 처리 흐름

`model_v3`는 다음 순서로 진행됩니다.

1. 데이터 로드
2. 결측치 기호 `-99`를 실제 결측값으로 변환
3. 날짜 feature 생성
4. train/validation/test 분할
5. 농장 규모 feature 생성
6. 혈통 KPN 요약 feature 생성
7. 기상 rolling feature 생성
8. train 기준 결측치 보완
9. 모델별 학습 및 비교
10. validation/test Macro-F1 저장
11. 클래스별 성능과 오답 분석 저장

## 3. 결측치 처리

### 3.1 기본 결측값 변환

공모전 데이터에서 결측은 주로 `-99`로 표시되어 있습니다.

따라서 먼저 아래 값들을 모두 `NaN`으로 바꾸었습니다.

```text
-99
-99.0
'-99'
'-99.0'
빈 문자열
```

이후 모든 결측치 보완 기준은 train split에서만 계산했습니다. validation/test 정보를 전처리 기준에 섞으면 data leakage가 생길 수 있기 때문입니다.

### 3.2 도체 수치형 결측

도체 관련 수치형 변수는 등급 예측에 매우 강한 변수입니다.

사용한 주요 변수는 다음과 같습니다.

```text
WEIGHT
BACKFAT
REA
WINDEX
INSFAT
YUKSAK
FATSAK
TISSUE
GROWTH
AGE
```

이 변수들은 결측이 있다고 행을 삭제하지 않았습니다. 결측 행이 특정 등급, 특히 `등외`와 관련될 가능성이 있기 때문입니다.

대신 한 행에서 도체 수치형 결측이 몇 개인지 나타내는 변수를 만들었습니다.

```text
carcass_missing_count
```

이 변수는 "값이 비어 있었다"는 사실 자체를 모델이 학습할 수 있게 합니다.

나머지 수치형 결측은 다음 기준으로 중앙값 보완을 적용했습니다.

```text
JUDGE_SEX + AGE + 도축연도 기준 median
```

해당 그룹에 값이 없으면 더 넓은 기준의 median으로 fallback 하도록 처리했습니다.

### 3.3 COST_AMT 결측

`COST_AMT`는 낙찰가격입니다. 등급과 관련성이 매우 높기 때문에 최종 성능 모델에서는 사용했습니다.

결측은 다음 순서로 채웠습니다.

```text
1. JUDGE_SEX + 도축연월 + WGRADE 기준 median
2. 없으면 JUDGE_SEX + 도축연월 기준 median
3. 그래도 없으면 전체 median
```

`WGRADE`는 최종등급이 아니라 육량등급입니다. 즉, `LAST_GRADE`를 직접 사용한 것이 아니므로 target leakage는 아닙니다.

### 3.4 GROWTH 처리

`GROWTH`는 성숙도 변수입니다.

성숙도 8, 9는 등급 판정에서 불리하게 작용할 수 있기 때문에 원래 `GROWTH` 값과 함께 다음 파생 변수를 추가했습니다.

```text
growth_ge8
```

의미는 다음과 같습니다.

```text
GROWTH >= 8 이면 1
그 외에는 0
```

### 3.5 농장 면적과 사육두수 결측

농장 데이터에서는 다음 변수를 사용했습니다.

```text
AREA
C2023
C2024
C2025
```

도축연도에 맞는 사육두수를 가져와 `cow_year`를 만들고, 분포가 한쪽으로 치우친 문제를 줄이기 위해 log 변환을 적용했습니다.

사용한 파생 변수는 다음과 같습니다.

```text
cow_year
log_cow_year
log_area
```

결측은 지역 단위 median을 우선 사용하고, 없으면 전체 median으로 보완했습니다.

### 3.6 혈통 데이터 결측

혈통 데이터는 모든 개체에 매칭되지 않습니다.

따라서 복잡한 혈통 관계를 모두 사용하지 않고, KPN 정보만 요약해서 사용했습니다.

```text
kpn_known
kpn_freq
log_kpn_freq
```

- `kpn_known`: KPN 정보가 있으면 1, 없으면 0
- `kpn_freq`: train 데이터에서 해당 KPN이 등장한 빈도
- `log_kpn_freq`: KPN 빈도의 log 변환

### 3.7 기상 결측과 rolling feature

기상 데이터는 도축 당일 날씨보다 도축 전 일정 기간 동안 누적된 환경 영향이 중요할 수 있다고 보았습니다.

그래서 도축일 이전의 날씨를 다음 기간으로 rolling 요약했습니다.

```text
30일
90일
180일
```

예시는 다음과 같습니다.

```text
도축 전 30일 평균 최고기온
도축 전 90일 총 강수량
도축 전 180일 열대야 일수
도축 전 30일 평균 THI
```

기상 결측은 변수 성격에 따라 다르게 처리했습니다.

```text
기온/풍속: 짧은 결측은 같은 지점 기준 선형보간
긴 결측: 주변 관측지점 활용
강수량/습도: 주변 관측지점 기반 보완
```

## 4. 사용 feature

`model_v3`에서 사용한 원본 feature 수는 총 80개입니다.

| 구분 | 개수 |
|---|---:|
| 수치형 feature | 74 |
| 범주형 feature | 6 |
| 전체 원본 feature | 80 |
| 기상 rolling feature | 54 |

범주형 feature는 다음 6개입니다.

```text
WGRADE
sido
sigungu
stn
abatt_month
abatt_season
```

중요한 점은 `JUDGE_SEX`를 통합모델의 입력 feature로 직접 넣지 않았다는 것입니다.

`JUDGE_SEX`는 모델 입력값이 아니라, 성별 분리 전략에서 데이터를 나누는 기준으로만 사용했습니다.

## 5. 성별 분리 전략

`model_v3`에서는 모든 모델에 대해 두 가지 전략을 비교했습니다.

### 5.1 Unified

전체 데이터를 하나의 모델로 학습합니다.

```text
암 + 거세 + 수 전체를 하나의 모델로 학습
```

단, 통합모델 feature에는 `JUDGE_SEX`를 넣지 않았습니다.

### 5.2 Gender Split

성별에 따라 일부 모델을 따로 학습합니다.

```text
암: 암 전용 모델
거세: 거세 전용 모델
수: 표본이 너무 적어서 unified 모델 예측을 fallback으로 사용
```

`수`는 데이터 수가 매우 적기 때문에 따로 모델을 만들면 불안정해질 수 있습니다. 그래서 `수`는 unified 모델의 예측을 그대로 사용했습니다.

## 6. 비교한 모델

`model_v3`에서 비교한 모델은 다음과 같습니다.

| 모델 | 설명 | 사용 이유 |
|---|---|---|
| RandomForest | 여러 decision tree를 독립적으로 학습해 평균/투표하는 bagging 계열 모델 | v1 계열 기준선 |
| Linear SVM | 선형 경계로 클래스를 구분하는 SVM | 단순 선형 모델과 tree 계열 모델 비교 |
| XGBoost | gradient boosting tree 모델 | 복잡한 비선형 관계와 등급 경계를 잘 학습할 가능성 |
| CatBoost | categorical feature 처리에 강한 boosting 모델 | 범주형 변수가 많기 때문에 후보로 포함 |

SVM은 일반적인 RBF kernel SVM이 아니라 `LinearSVC`를 사용했습니다. 전체 데이터가 매우 커서 kernel SVM은 현실적으로 학습 시간이 너무 길기 때문입니다.

## 7. 모델별 결과

현재 `model_v3_outputs/model_v3_metrics.csv` 기준 결과입니다.

| 모델 | 전략 | Valid Macro-F1 | Test Macro-F1 | 학습 시간 |
|---|---|---:|---:|---:|
| XGBoost | gender_split | 0.985401 | 0.984853 | 약 8.4분 |
| XGBoost | unified | 0.985170 | 0.984597 | 약 8.4분 |
| RandomForest | unified | 0.949371 | 0.949530 | 약 34.2분 |
| RandomForest | gender_split | 0.948613 | 0.948850 | 약 21.1분 |
| Linear SVM | gender_split | 0.809819 | 0.809539 | 약 27.9분 |
| Linear SVM | unified | 0.785745 | 0.786252 | 약 10.8분 |
| CatBoost | failed | - | - | 실패 |

가장 좋은 모델은 다음입니다.

```text
XGBoost + gender_split
Valid Macro-F1 = 0.985401
Test Macro-F1 = 0.984853
```

## 8. 결과 해석

### 8.1 XGBoost가 가장 좋았던 이유

XGBoost는 tree boosting 계열 모델입니다.

RandomForest는 여러 나무를 독립적으로 학습해서 투표하는 방식입니다. 반면 XGBoost는 이전 모델이 틀린 부분을 다음 tree가 보완하는 방식으로 순차적으로 학습합니다.

이 문제는 등급 경계가 단순하지 않고, 여러 변수의 조합으로 결정됩니다.

예를 들어 다음 변수들이 함께 영향을 줄 수 있습니다.

```text
도체중
등지방두께
등심단면적
육량지수
근내지방도
성숙도
낙찰가격
도축 전 기상 조건
농장 규모
```

따라서 XGBoost처럼 변수 간 비선형 관계와 복잡한 경계를 잘 학습하는 모델이 RandomForest나 Linear SVM보다 좋은 성능을 보인 것으로 해석할 수 있습니다.

### 8.2 Gender Split의 효과

XGBoost에서는 unified와 gender_split 차이가 매우 작았습니다.

```text
XGBoost unified test = 0.984597
XGBoost gender_split test = 0.984853
차이 = +0.000256
```

즉, XGBoost에서는 성별 분리보다 모델 자체의 학습 능력이 더 큰 성능 개선 요인으로 보입니다.

다만 gender_split이 가장 높은 점수를 기록했기 때문에 최종 후보는 XGBoost gender_split으로 선택했습니다.

### 8.3 Linear SVM 결과

Linear SVM은 XGBoost나 RandomForest보다 낮은 성능을 보였습니다.

이는 등급 예측 문제가 단순한 선형 경계로 나누기 어렵다는 것을 의미합니다.

따라서 최종 모델은 선형 모델보다 tree 기반 비선형 모델이 더 적합하다고 판단했습니다.

## 9. 클래스별 성능

XGBoost gender_split 모델의 validation 기준 주요 클래스 F1은 다음과 같습니다.

| 등급 | F1-score |
|---|---:|
| 3A | 0.940199 |
| 3B | 0.964075 |
| 2A | 0.968622 |
| 3C | 0.972474 |
| 2B | 0.977901 |
| 등외 | 0.989666 |

가장 어려운 등급도 F1이 0.94 이상입니다. Macro-F1 기준에서 매우 안정적인 결과입니다.

## 10. 주요 오답

XGBoost gender_split의 주요 오답은 대부분 인접 등급 사이에서 발생했습니다.

| 실제 등급 | 예측 등급 | 오답 수 |
|---|---|---:|
| 2B | 2A | 722 |
| 2A | 2B | 476 |
| 1B | 1A | 462 |
| 3B | 3A | 441 |
| 3A | 3B | 404 |
| 1+B | 1+A | 276 |
| 1A | 1B | 234 |

완전히 엉뚱한 등급으로 틀리는 경우보다는, 등급 경계가 가까운 클래스끼리 혼동하는 경우가 많았습니다.

이는 모델이 전체적인 등급 구조는 잘 학습했지만, 인접 등급 경계에서는 여전히 구분이 어렵다는 의미입니다.

## 11. CatBoost 에러

CatBoost는 실행 중 실패했습니다.

에러 메시지는 다음과 같습니다.

```text
CatBoostError('bad allocation')
```

이 에러는 보통 메모리 할당 실패를 의미합니다.

즉, CatBoost가 import되지 않아서 실패한 것이 아니라, 전체 데이터로 학습하는 과정에서 필요한 RAM을 확보하지 못해 실패한 것으로 볼 수 있습니다.

가능한 원인은 다음과 같습니다.

```text
전체 데이터 행 수가 많음
다중분류 클래스 수가 많음
범주형 feature가 포함됨
CatBoost가 내부적으로 추가 메모리를 많이 사용함
```

CatBoost를 다시 시도하려면 다음 방법을 고려할 수 있습니다.

```text
학습 데이터 일부 샘플링
iterations 감소
depth 감소
고유값이 많은 범주형 변수 일부 제거
used_ram_limit 설정
```

다만 XGBoost가 이미 높은 성능을 보였기 때문에, 현재 단계에서는 CatBoost를 반드시 살릴 필요는 낮습니다.

## 12. 최종 선택

최종 후보는 다음 모델입니다.

```text
XGBoost + gender_split
```

선택 이유는 다음과 같습니다.

1. 전체 모델 중 validation/test Macro-F1이 가장 높음
2. RandomForest 대비 큰 성능 개선을 보임
3. Linear SVM보다 훨씬 높은 성능을 보여 비선형 모델이 적합함을 확인
4. 클래스별 F1도 안정적임
5. 주요 오답이 대부분 인접 등급 사이에서 발생해 예측 구조가 자연스러움

보고서에는 다음처럼 설명할 수 있습니다.

```text
RandomForest 기반 기준 모델을 구축한 뒤, 동일한 전처리와 feature 조건에서 SVM, XGBoost, CatBoost를 비교하였다. CatBoost는 메모리 부족으로 학습에 실패했으며, Linear SVM은 낮은 성능을 보였다. XGBoost는 tree boosting 구조를 통해 도체 변수, 가격, 기상 rolling feature 간의 비선형 관계를 효과적으로 학습했고, 최종적으로 XGBoost gender_split 모델이 Test Macro-F1 0.984853으로 가장 높은 성능을 보였다.
```


