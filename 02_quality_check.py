"""
=============================================================================
02_quality_check.py
파이프라인 2단계: 입력 이미지 품질 검사
=============================================================================

다이어그램 2번 단계에 대응합니다.
- 흐림 정도 (blur)        : 라플라시안 분산으로 측정
- 밝기/대비 (brightness)  : 그레이스케일 평균/표준편차
- 피부 영역 포함 여부      : HSV 피부색 마스크 비율
- (선택) 얼굴 검출         : OpenCV Haar Cascade

기준 미달 시 '재촬영 요청' 신호(False)를 반환합니다.
=============================================================================
"""

import cv2
import numpy as np


# 임계값들 — 데이터에 맞게 튜닝하세요
THRESH = {
    "blur_min": 10,      # 라플라시안 분산이 이보다 낮으면 흐림
    "bright_min": 40,       # 너무 어두움
    "bright_max": 220,      # 너무 밝음/과노출
    "skin_ratio_min": 0.15, # 피부 픽셀 비율 최소
}


def variance_of_laplacian(gray):
    """초점/흐림 측정: 라플라시안 분산이 클수록 선명."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def skin_ratio(bgr):
    """HSV 기반 단순 피부색 마스크 비율 계산."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 일반적인 피부색 HSV 범위 (조명에 따라 조정 필요)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 170, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return float(mask.mean()) / 255.0


def check_quality(image_path, detect_face=False):
    """
    이미지 품질을 검사하고 (통과 여부, 상세 지표 dict) 반환.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return False, {"error": "이미지를 읽을 수 없음"}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    blur = variance_of_laplacian(gray)
    brightness = float(gray.mean())
    skin = skin_ratio(bgr)

    checks = {
        "blur_ok": blur >= THRESH["blur_min"],
        "bright_ok": THRESH["bright_min"] <= brightness <= THRESH["bright_max"],
        "skin_ok": skin >= THRESH["skin_ratio_min"],
    }

    if detect_face:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, 1.1, 4)
        checks["face_ok"] = len(faces) > 0

    passed = all(checks.values())
    detail = {
        "blur": round(blur, 1),
        "brightness": round(brightness, 1),
        "skin_ratio": round(skin, 3),
        **checks,
        "passed": passed,
        "action": "OK" if passed else "재촬영 요청",
    }
    return passed, detail


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    ok, info = check_quality(path, detect_face=False)
    print(info)
