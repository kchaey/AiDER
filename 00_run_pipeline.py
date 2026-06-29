import importlib.util
import numpy as np
import pandas as pd


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_pipeline(image_input, lifestyle_inputs=None, skip_survey=False, 
                 yolo_weights="runs/detect/acne04_yolov8s/weights/best.pt", 
                 clf_model=None, fusion_model=None):
    """
    image_input      : 파일 경로(str 또는 Path) 또는 카메라/갤러리 스트림 이미지
    lifestyle_inputs : 사용자 설문 데이터 dict (음주, 호르몬 요인 포함)
    skip_survey      : 문진표 작성 건너뛰기 여부 (True 시 기본값 대체)
    """
    from ultralytics import YOLO

    # 외부 모듈 동적 로드
    qc = _load("qc", "02_quality_check.py")
    det = _load("det", "03_yolo_detect_and_roi.py")
    cls = _load("cls", "04_cnn_classify_and_aggregate.py")

    # -----------------------------------------------------------------------
    # [기능 1] 2단계: 이미지 입력 및 품질 검사 (카메라/갤러리 공용)
    # -----------------------------------------------------------------------
    ok, info = qc.check_quality(image_input)
    if not ok:
        return {
            "status": "재촬영 요청",
            "message": "더 정확한 분석을 위해 초점이 흐리거나 어두운 사진은 제한됩니다.",
            "quality_detail": info
        }

    # -----------------------------------------------------------------------
    # [기능 2] 문진표 전처리 및 확장 (Skip 로직 포함)
    # -----------------------------------------------------------------------
    # 확장된 문진 변수 컬럼 정의
    extended_lifestyle_cols = [
        "age", "sex", "lesion_site", "duration_weeks", "pain_level",
        "cosmetic_change", "sleep_hours", "stress_level", "diet_score", "fever",
        "recent_alcohol",  # 최근 음주 여부 (0 또는 1)
        "hormone_factor"   # 생리/호르몬 영향 여부 (0 또는 1)
    ]
    
    if skip_survey or lifestyle_inputs is None:
        # 문진표를 건너뛰거나 입력 안 했을 경우 안전한 대표 기본값(대표치) 지정
        default_lifestyle = {
            "age": 25, "sex": 0, "lesion_site": 0, "duration_weeks": 4, "pain_level": 1,
            "cosmetic_change": 0, "sleep_hours": 7.0, "stress_level": 5, "diet_score": 5, "fever": 0,
            "recent_alcohol": 0, "hormone_factor": 0
        }
        final_lifestyle = default_lifestyle
        survey_status = "Skipped (Default values applied)"
    else:
        # 입력된 문진 데이터 반영하고, 누락된 항목은 기본값 처리
        final_lifestyle = {}
        default_vals = [25, 0, 0, 4, 1, 0, 7.0, 5, 5, 0, 0, 0]
        for col, default in zip(extended_lifestyle_cols, default_vals):
            final_lifestyle[col] = lifestyle_inputs.get(col, default)
        survey_status = "Completed"

    # -----------------------------------------------------------------------
    # 3~4단계: YOLOv8 병변 검출 및 ROI 추출
    # -----------------------------------------------------------------------
    yolo = YOLO(yolo_weights)
    detections = det.detect_lesions(yolo, image_input)
    if not detections:
        return {
            "status": "완료",
            "message": "검출된 피부 병변이 없습니다. 정상 피부이거나 미세 병변입니다.",
            "n_lesions": 0,
            "top_diseases": [{"disease": "정상 혹은 판단 불가", "prob": 1.0}],
            "severity_grade": 0
        }

    rois = det.extract_rois(image_input, detections)

    # -----------------------------------------------------------------------
    # 5~6단계: EfficientNet 이미지 분석 (질환 분류 및 다중 ROI 가중치 통합)
    # -----------------------------------------------------------------------
    # predict_image_disease 결과 내부에서 다중 ROI 가중평균 확률 계산 수행됨
    disease_results = cls.predict_image_disease(clf_model, rois)

    # -----------------------------------------------------------------------
    # [기능 3] 상위 5개 예비 질환 실시간 정렬 및 가공
    # -----------------------------------------------------------------------
    raw_probs = disease_results["probs"]  # {"여드름": 0.65, "모낭염": 0.20, ...}
    
    # 확률이 높은 순서대로 정렬하여 상위 5개 리스트업
    sorted_diseases = sorted(raw_probs.items(), key=lambda x: x[1], reverse=True)
    top_5_diseases = [
        {"rank": idx + 1, "disease": name, "probability_pct": round(prob * 100, 2)}
        for idx, (name, prob) in enumerate(sorted_diseases[:5])
    ]

    # -----------------------------------------------------------------------
    # [기능 4] 7단계: 멀티모달 데이터 결합 및 XGBoost 최종 중증도 판정
    # -----------------------------------------------------------------------
    img_features = {
        "img_prob_acne": raw_probs.get("여드름", 0.0),
        "n_lesions": len(rois),
        "total_lesion_area": sum(r["area"] for r in rois),
    }
    
    # 문진표 데이터 + 이미지 추출 특징 융합
    fusion_row = {**final_lifestyle, **img_features}
    
    # 모델 입력용 DataFrame 변환 (학습 시 컬럼 순서 유지 필수)
    # 실제 학습 모델 환경에 맞추어 컬럼 리스트 순서대로 정렬하여 매핑
    feature_vector = pd.DataFrame([fusion_row])
    
    # XGBoost를 통한 중증도(0~3) 등급 예측
    if fusion_model is not None:
        severity_pred = int(fusion_model.predict(feature_vector.values)[0])
    else:
        # 데모용 룰베이스 기반 결합 매핑 가상 로직 (모델 미로드 시)
        severity_pred = min(3, int(len(rois) // 5))

    return {
        "status": "완료",
        "survey_status": survey_status,
        "n_lesions": len(rois),
        "top_5_diseases": top_5_diseases,       # 상위 5개 예비 질환 실시간 아웃풋
        "dominant_disease": top_5_diseases[0]["disease"],
        "severity_grade": severity_pred,       # Hayashi 등급 (0~3)
        "quality_check": "PASS" if ok else "FAIL"
    }


if __name__ == "__main__":
    import os
    import pprint

    print("=== 통합 파이프라인 단독 실행 테스트 ===")
    
    # 1. 테스트할 이미지 파일 이름 (같은 폴더에 이 이름의 사진이 있어야 합니다)
    test_image_path = "test.jpg"

    if not os.path.exists(test_image_path):
        print(f"[오류] '{test_image_path}' 파일이 없습니다. 테스트할 얼굴 사진을 같은 폴더에 넣어주세요.")
    else:
        # 2. YOLO 가중치 자동 설정 (학습된 모델이 없으면 임시로 기본 모델 사용)
        yolo_weight = "runs/detect/acne04_yolov8s/weights/best.pt"
        if not os.path.exists(yolo_weight):
            print("[안내] 학습된 YOLO 모델을 찾을 수 없어 기본 모델(yolov8n.pt)로 임시 테스트를 진행합니다.")
            yolo_weight = "yolov8n.pt"

        # 3. 피부 질환 분류 모델 임시 로드 (04번 스크립트 활용)
        print("1. AI 모델을 로드하는 중...")
        cls_module = _load("cls", "04_cnn_classify_and_aggregate.py")
        dummy_classifier = cls_module.build_classifier()

        # 4. 가상의 사용자 문진표 데이터 생성
        sample_lifestyle = {
            "age": 24, "sex": 0, "lesion_site": 0, "duration_weeks": 2, "pain_level": 3,
            "cosmetic_change": 1, "sleep_hours": 5.5, "stress_level": 8, "diet_score": 7, "fever": 0,
            "recent_alcohol": 1, "hormone_factor": 1
        }

        # 5. 파이프라인 실행!
        print("2. 이미지 품질 검사 및 병변 분석 중...\n")
        final_result = run_pipeline(
            image_input=test_image_path,
            lifestyle_inputs=sample_lifestyle,
            skip_survey=False,
            yolo_weights=yolo_weight,
            clf_model=dummy_classifier,
            fusion_model=None # 통합 모델이 없으면 내부 데모 로직이 자동으로 작동합니다
        )

        # 6. 최종 결과 보기 좋게 출력
        print("=============================")
        print("        최종 분석 결과        ")
        print("=============================")
        pprint.pprint(final_result, sort_dicts=False)
