"""
=============================================================================
03_yolo_detect_and_roi.py
파이프라인 3~4단계: YOLOv8 병변 검출 학습/추론 + ROI 추출
=============================================================================

3단계: ACNE04로 YOLOv8을 학습하여 'lesion' 바운딩 박스를 검출
4단계: 검출된 박스에 margin을 더해 ROI를 잘라내고 전처리

필요 패키지:
    pip install ultralytics opencv-python

이 파일이 이 프로젝트의 '반드시 실행하고 싶은' 핵심(3번)입니다.
=============================================================================
"""

import cv2
import numpy as np
from pathlib import Path


# --------------------------------------------------------------------------
# 3-A. 학습 (한 번만 실행)
# --------------------------------------------------------------------------
def train_yolo(data_yaml="./acne04_yolo/data.yaml", epochs=100, imgsz=640):
    """
    ACNE04 YOLO 포맷 데이터로 YOLOv8 검출 모델 학습.
    여드름 병변은 작은 객체이므로 imgsz를 크게(640~1024) 두는 편이 유리.
    """
    from ultralytics import YOLO

    # n/s/m/l/x 중 데이터 양과 GPU에 맞게 선택. 시작은 yolov8s 권장.
    model = YOLO("yolov8s.pt")  # 사전학습 가중치에서 전이학습
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        patience=20,           # 조기 종료
        name="acne04_yolov8s",
        # 작은 객체 검출 도움이 되는 증강
        mosaic=1.0,
        scale=0.5,
        fliplr=0.5,
    )
    return results


# --------------------------------------------------------------------------
# 3-B. 추론
# --------------------------------------------------------------------------
def detect_lesions(model, image_path, conf=0.25):
    """
    학습된 모델로 병변 검출.
    반환: [{'bbox': (x1,y1,x2,y2), 'conf': float}, ...]
    (다이어그램 3번의 'YOLO 출력 정보' 표에 해당)
    """
    res = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    detections = []
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        detections.append({
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "conf": float(box.conf[0].cpu().numpy()),
        })
    return detections


# --------------------------------------------------------------------------
# 4. ROI 추출 + 전처리
# --------------------------------------------------------------------------
def preprocess_roi(roi, size=224):
    """
    다이어그램 4번 전처리 과정 재현:
    Resize -> Color Normalization -> Contrast Enhancement -> Noise Reduction
    반환: float32 정규화된 (size, size, 3) 배열 (CNN 입력용)
    """
    # 1) Resize
    roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)

    # 2) Contrast Enhancement (CLAHE, LAB의 L채널에 적용)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    roi = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # 3) Noise Reduction
    roi = cv2.bilateralFilter(roi, d=5, sigmaColor=50, sigmaSpace=50)

    # 4) Color Normalization (0~1 스케일 + ImageNet 표준화)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    roi = (roi - mean) / std
    return roi


def extract_rois(image_path, detections, margin=0.15, size=224):
    """
    검출 박스마다 margin을 더해 ROI를 잘라내고 전처리.
    margin: 박스 크기 대비 여유 비율 (다이어그램의 'Margin 포함')
    반환: [{'roi': 전처리배열, 'bbox': (..), 'conf': float, 'area': int}, ...]
    """
    img = cv2.imread(str(image_path))
    H, W = img.shape[:2]
    rois = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bw, bh = x2 - x1, y2 - y1
        # margin 적용 + 경계 클리핑
        mx, my = int(bw * margin), int(bh * margin)
        xa, ya = max(0, x1 - mx), max(0, y1 - my)
        xb, yb = min(W, x2 + mx), min(H, y2 + my)

        crop = img[ya:yb, xa:xb]
        if crop.size == 0:
            continue
        roi = preprocess_roi(crop, size=size)
        rois.append({
            "roi": roi,
            "bbox": (xa, ya, xb, yb),
            "conf": det["conf"],
            "area": (xb - xa) * (yb - ya),
        })
    return rois


# --------------------------------------------------------------------------
# 데모 실행
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from ultralytics import YOLO

    # 학습이 끝났다고 가정하고 best.pt 로드
    weights = "runs/detect/acne04_yolov8s/weights/best.pt"
    if Path(weights).exists():
        model = YOLO(weights)
        test_img = "test.jpg"
        dets = detect_lesions(model, test_img)
        print(f"검출된 병변 수: {len(dets)}")
        for i, d in enumerate(dets):
            print(f"  [{i}] bbox={d['bbox']} conf={d['conf']:.2f}")
        rois = extract_rois(test_img, dets)
        print(f"추출된 ROI 수: {len(rois)}")
    else:
        print("먼저 train_yolo()로 모델을 학습하세요.")
