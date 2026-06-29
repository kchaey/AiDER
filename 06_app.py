
import streamlit as st
import importlib.util
import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------
# 모듈 동적 로드
# --------------------------------------------------------------------------
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

qc = load_module("qc", "02_quality_check.py")
det = load_module("det", "03_yolo_detect_and_roi.py")
cls = load_module("cls", "04_cnn_classify_and_aggregate.py")
fusion = load_module("fusion", "05_lifestyle_fusion_up.py") # 필요한 경우 로드

# --------------------------------------------------------------------------
# UI 레이아웃 시작
# --------------------------------------------------------------------------
st.set_page_config(page_title="피부 분석 AI 파이프라인", layout="wide")
st.title("🩺 AiDER: 멀티모달 피부 병변 분석 시스템")

col1, col2 = st.columns([1, 2]) # 오른쪽 결과창을 더 넓게 배치

with col1:
    st.header("사용자 입력")
    
    st.subheader("📸 피부 사진 업로드")
    uploaded_file = st.file_uploader("스마트폰 갤러리/PC 사진 선택", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 원본 사진", use_container_width=True)

    st.subheader("📝 생활습관 문진표")
    skip_survey = st.checkbox("문진표 건너뛰기 (기본값 진행)")
    
    with st.form("survey_form"):
        age = st.number_input("나이", min_value=10, max_value=100, value=24, disabled=skip_survey)
        sex_input = st.radio("성별", ["여성", "남성"], disabled=skip_survey)
        sex = 0 if sex_input == "여성" else 1
        
        lesion_site = st.selectbox("병변 부위", ["얼굴", "목", "가슴", "등", "복합"], disabled=skip_survey)
        duration = st.slider("증상 지속 기간 (주)", 1, 52, 4, disabled=skip_survey)
        pain = st.slider("통증/가려움 (0~10)", 0, 10, 2, disabled=skip_survey)
        
        c_a, c_b = st.columns(2)
        with c_a:
            hormone = st.radio("최근 호르몬 변화", ["아니오", "예"], disabled=skip_survey)
            cosmetics = st.radio("최근 화장품 변경", ["아니오", "예"], disabled=skip_survey)
        with c_b:
            alcohol = st.radio("최근 3일 내 음주", ["아니오", "예"], disabled=skip_survey)
            shaving = st.radio("최근 면도 여부", ["아니오", "예"], disabled=skip_survey)
            
        submitted = st.form_submit_button("전체 파이프라인 실행")

with col2:
    st.header("분석 리포트")
    
    if submitted:
        if uploaded_file is None:
            st.error("사진을 먼저 업로드해주세요!")
        else:
            with st.spinner("AI가 분석 중 입니다..."):
                temp_path = "temp_upload.jpg"
                image.save(temp_path)
                
                # --- 2단계: 품질 검사 ---
                is_ok, qc_info = qc.check_quality(temp_path)
                if not is_ok:
                    st.error("🚨 사진 품질이 떨어집니다. 정확한 분석을 위해 다시 촬영해주세요.")
                    st.json(qc_info)
                else:
                    st.success("✅ 분석을 시작합니다.")
                    
                    # --- 3단계: YOLO 병변 검출 ---
                    from ultralytics import YOLO
                    try:
                        weights_path = "runs/detect/acne04_yolov8s/weights/best.pt" 
                        if not os.path.exists(weights_path):
                            weights_path = "yolov8n.pt" # 임시 모델
                        model = YOLO(weights_path)
                        detections = det.detect_lesions(model, temp_path)
                    except Exception as e:
                        st.error(f"YOLO 에러: {e}")
                        detections = []

                    if not detections:
                        st.warning("🔍 검출된 병변이 없습니다.")
                    else:
                        st.markdown("### 🔍 STEP 3: 병변 영역 검출")
                        img_bgr = cv2.imread(temp_path)
                        for d in detections:
                            x1, y1, x2, y2 = d['bbox']
                            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), width=300)

                        # --- 4단계: ROI 추출 ---
                        st.markdown("### ✂️ STEP 4: 병변 위치 파악")
                        rois = det.extract_rois(temp_path, detections)
                        
                        # 시각화를 위해 원본에서 직접 자른 이미지 표시 (상위 5개만)
                        roi_cols = st.columns(min(5, len(rois)))
                        for idx, r in enumerate(rois[:5]):
                            x1, y1, x2, y2 = r['bbox']
                            crop_img = cv2.imread(temp_path)[y1:y2, x1:x2]
                            with roi_cols[idx]:
                                st.image(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB), caption=f"ROI {idx+1}")

                        # --- 5단계: CNN 기반 피부질환 분류 ---
                        st.markdown("### 🧠 STEP 5: 질환 분류 중... ")
                        
                        # 분류 모델 로드 (학습된 가중치가 없으면 랜덤 초기화 모델 사용)
                        clf_model = cls.build_classifier(pretrained=False) 
                        if os.path.exists("best_cnn.pth"):
                            clf_model.load_state_dict(torch.load("best_cnn.pth", map_location="cpu"))
                            clf_model.eval()
                        
                        roi_results = []
                        table_data = []
                        for idx, r in enumerate(rois):
                            probs = cls.classify_roi(clf_model, r["roi"])
                            roi_results.append({"probs": probs, "conf": r["conf"], "area": r["area"]})
                            
                            row_dict = {"ROI ID": idx + 1}
                            for i, class_name in enumerate(cls.CLASSES):
                                row_dict[class_name] = round(float(probs[i]), 2)
                            table_data.append(row_dict)
                            
                        st.dataframe(pd.DataFrame(table_data), hide_index=True)

                        # --- 6단계: 다중 ROI 결과 통합 ---
                        st.markdown("### 📊 STEP 6: 다중 ROI 가중 평균 통합")
                        final_probs, pred_class = cls.aggregate_rois(roi_results)
                        
                        df_probs = pd.DataFrame({
                            "질환": cls.CLASSES,
                            "이미지 단위 확률": [round(float(p), 2) for p in final_probs]
                        }).set_index("질환")
                        st.bar_chart(df_probs)

                        # --- 7, 8, 11단계: 생활습관 융합 및 최종 출력 ---
                        st.markdown("---")
                        st.markdown("### 🏁 분석 결과 입니다.")
                        
                        # 05_lifestyle_fusion 로직 모사 (학습된 XGBoost가 없으므로 가상의 룰베이스 적용)
                        # 실제로는 여기서 fusion_model.predict() 가 들어갑니다.
                        import xgboost as xgb
                        fusion_model = xgb.XGBClassifier()
                        if os.path.exists("best_fusion_model.json"):
                            fusion_model.load_model("best_fusion_model.json")

                            # 실제 예측 로직 (feature_vector를 만들어서 넣음)
                            if os.path.exists("best_fusion_model.json"):
                            # [진짜 모델이 있을 때] XGBoost로 예측
                                fusion_model.load_model("best_fusion_model.json")
                                site_mapping = {"얼굴": 0, "목": 1, "가슴": 2, "등": 3, "복합": 4}
                                lesion_site_encoded = site_mapping.get(lesion_site, 0)
                                # XGBoost가 학습했던 딱 13개의 변수만 골라서 전달!
                                feature_row = {
                                    "age": age, 
                                    "sex": sex, 
                                    "lesion_site": lesion_site_encoded, 
                                    "duration_weeks": duration,
                                    "pain_level": pain, 
                                    "cosmetic_change": 1 if cosmetics == "예" else 0,
                                    "sleep_hours": 7.0,   # UI에 없는 값은 적절한 기본값 처리 혹은 UI 추가
                                    "stress_level": 5,   # UI에 없는 값은 적절한 기본값 처리 혹은 UI 추가
                                    "diet_score": 5,     # UI에 없는 값은 적절한 기본값 처리 혹은 UI 추가
                                    "fever": 1 if pain > 7 else 0, # 예시용 매핑
                                    "hormone_change": 1 if hormone == "예" else 0,
                                    "alcohol_3days": 1 if alcohol == "예" else 0,
                                    "shaving": 1 if shaving == "예" else 0,
                                    
                                    # 이미지 분석 결과 탭
                                    "img_prob_acne": final_probs[cls.CLASSES.index("여드름")] if "여드름" in cls.CLASSES else 0,
                                    "n_lesions": len(rois), 
                                    "total_lesion_area": sum(r["area"] for r in rois)
                                }
                                input_df = pd.DataFrame([feature_row])
                                severity_grade = int(fusion_model.predict(input_df)[0])
                        else:
                            img_score = final_probs[cls.CLASSES.index("여드름")] if "여드름" in cls.CLASSES else 0
                            penalty = 0
                            if not skip_survey:
                                if alcohol == "예": penalty += 1
                                if hormone == "예": penalty += 1
                                if pain > 5: penalty += 1
                                
                            severity_grade = min(3, int((len(rois) / 10) + (penalty * 0.5)))
                        
                        # 다이어그램과 유사한 최종 UI 구성
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.info(f"**최종 진단:** {pred_class} 가능성 높음")
                            st.metric(label="예측 신뢰도", value=f"{int(max(final_probs)*100)}%")
                            st.write(f"**Hayashi 중증도 등급:** {severity_grade} 등급")
                            
                        with res_col2:
                            st.write("**추천 성분 (Rule-based)**")
                            if "여드름" in pred_class or severity_grade > 0:
                                st.success("✔️ Benzoyl Peroxide\n✔️ Salicylic Acid\n✔️ 저자극 보습제")
                            else:
                                st.success("✔️ 저자극 보습제\n✔️ 자외선 차단제")
                                
                            if severity_grade >= 2 or pain > 7:
                                st.error("🚨 위험 신호 있음: 병원 진료 권고")
                            else:
                                st.success("🟢 위험 신호 없음: 일반의약품 관리 가능")

                if os.path.exists("temp_upload.jpg"):
                    os.remove("temp_upload.jpg")
