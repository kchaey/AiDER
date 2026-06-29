"""
=============================================================================
05_lifestyle_fusion.py
파이프라인 7단계: 생활습관·증상 정보 모델링 + 이미지 특징 결합
=============================================================================

다이어그램 7번에 대응. 두 가지를 합칩니다.
  (A) 이미지 기반 특징  : 5~6단계에서 나온 질환 확률 + 병변 수/면적 등
  (B) 생활습관 tabular  : 나이, 성별, 병변 발생 부위, 증상 지속 기간,
                          통증/가려움, 화장품 변경, 수면, 스트레스, 식습관 등

목표 라벨(target)은 상황에 따라 선택:
  - 여드름 '중증도 등급' (ACNE04는 0~3 Hayashi 등급 제공)  ← 권장
  - 또는 '예상 질환' / '치료 반응' 등

[중요한 현실 안내]
공개 데이터 중 '이미지'와 '생활습관'이 같은 환자로 연결된 것은 없습니다.
그래서 두 경로를 둡니다.
  경로1) 실제 설문 tabular 데이터(CSV)를 보유한 경우 -> load_real_tabular()
  경로2) 데이터가 없을 때 프로토타입용 합성 데이터 생성 -> make_synthetic()

필요 패키지: pip install xgboost scikit-learn pandas numpy
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------------------------------
# 7-A. 생활습관 특징 정의 (다이어그램 '입력 변수 예시')
# --------------------------------------------------------------------------
LIFESTYLE_COLS = [
    "age", "sex", "lesion_site", "duration_weeks", "pain_level", 
    "cosmetic_change", "sleep_hours", "stress_level", "diet_score", "fever",
    "hormone_change",  # 추가: 호르몬 변화 (0/1)
    "alcohol_3days",   # 추가: 3일 내 음주 (0/1)
    "shaving"
]


# --------------------------------------------------------------------------
# 경로1: 실제 tabular 데이터 로드
# --------------------------------------------------------------------------
def load_real_tabular(csv_path):
    """
    설문/임상 CSV를 로드. 컬럼명이 LIFESTYLE_COLS와 'severity'를 포함한다고 가정.
    (사우디·DLQI 등 공개 연구 설문 구조를 본떠 직접 구축한 CSV 사용)
    """
    df = pd.read_csv(csv_path)
    return df


# --------------------------------------------------------------------------
# 경로2: 합성 데이터 생성 (프로토타입/데모용)
# --------------------------------------------------------------------------
def make_synthetic(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(13, 40, n),
        "sex": rng.integers(0, 2, n),
        "lesion_site": rng.integers(0, 5, n), # 0~4 (얼굴, 목, 가슴, 등, 복합)
        "duration_weeks": rng.integers(1, 52, n),
        "pain_level": rng.integers(0, 11, n), # UI에 맞춰 0~10 스케일로 변경
        "cosmetic_change": rng.integers(0, 2, n),
        "sleep_hours": rng.normal(6.5, 1.3, n).clip(3, 10),
        "stress_level": rng.integers(0, 11, n),
        "diet_score": rng.integers(0, 11, n),
        "fever": rng.integers(0, 2, n),
        "hormone_change": rng.integers(0, 2, n), # 추가
        "alcohol_3days": rng.integers(0, 2, n),   # 추가
        "shaving": rng.integers(0, 2, n),         # 추가
    })

    # 잠재 중증도 점수: 알려진 위험요인에 가중치
    latent = (
        0.05 * df["age"] + 0.25 * df["stress_level"] + 0.30 * df["diet_score"]
        - 0.40 * df["sleep_hours"] + 0.20 * df["duration_weeks"] / 10 + 0.2 * df["pain_level"]
        + 0.3 * df["hormone_change"] + 0.3 * df["alcohol_3days"] # 새로운 가중치 반영
        + rng.normal(0, 1.5, n)
    )
    df["severity"] = pd.qcut(latent, q=4, labels=[0, 1, 2, 3]).astype(int)
    return df


# --------------------------------------------------------------------------
# 7-B. 이미지 특징 결합
# --------------------------------------------------------------------------
def attach_image_features(df, image_feat_df=None):
    if image_feat_df is not None and "patient_id" in df.columns:
        return df.merge(image_feat_df, on="patient_id", how="left")

    # 데모: 이미지 특징을 중증도와 약한 상관으로 합성
    rng = np.random.default_rng(0)
    df = df.copy()
    df["img_prob_acne"] = (df["severity"] / 3 * 0.6
                           + rng.normal(0, 0.15, len(df))).clip(0, 1)
    df["n_lesions"] = (df["severity"] * 8
                       + rng.integers(0, 10, len(df))).clip(0, None)
    df["total_lesion_area"] = (df["n_lesions"]
                               * rng.integers(80, 200, len(df)))
    return df


# --------------------------------------------------------------------------
# 7-C. XGBoost 학습 (이미지 + 생활습관 통합)
# --------------------------------------------------------------------------
def train_fusion_model(df, target="severity"):
    """
    XGBoost 다중분류기로 중증도(또는 질환) 예측.
    반환: (모델, 테스트 리포트)
    """
    from xgboost import XGBClassifier

    feature_cols = [c for c in df.columns if c != target]
    X = df[feature_cols].values
    y = df[target].values

    # 라벨 정수화
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, zero_division=0)

    # 특징 중요도 (어떤 생활습관이 중요한지 해석)
    importance = dict(sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda kv: kv[1], reverse=True
    ))

    return model, {"accuracy": acc, "report": report, "importance": importance}


# --------------------------------------------------------------------------
# 전체 7단계 데모 실행
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) tabular 확보 (실제 CSV가 있으면 load_real_tabular로 교체)
    df = make_synthetic(n=3000)

    # 2) 이미지 특징 결합
    df = attach_image_features(df)

    # 3) 통합 모델 학습
    model, result = train_fusion_model(df, target="severity")

    print(f"정확도: {result['accuracy']:.3f}\n")
    print("분류 리포트:\n", result["report"])
    print("특징 중요도 (상위):")
    for k, v in list(result["importance"].items())[:8]:
        print(f"  {k:20s} {v:.4f}")



# --------------------------------------------------------------------------
# 전체 7단계 데모 실행 (수정본)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("1. 데이터를 생성/로드합니다...")
    df = make_synthetic(n=3000)
    df = attach_image_features(df)

    print("2. XGBoost 통합 모델을 학습합니다...")
    model, result = train_fusion_model(df, target="severity")

    print(f"정확도: {result['accuracy']:.3f}")
    
    # [추가된 핵심 코드] 학습된 모델을 파일로 저장!
    model.save_model("best_fusion_model.json")
    print("✅ 모델이 'best_fusion_model.json'으로 저장되었습니다!")
