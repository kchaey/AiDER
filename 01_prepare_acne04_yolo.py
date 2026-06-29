"""
=============================================================================
01_prepare_acne04_yolo.py
ACNE04 데이터셋을 YOLO 학습 포맷으로 변환하는 스크립트 (파이프라인 1~3단계 준비)
=============================================================================

[배경]
ACNE04 데이터셋(GitHub: xpwu95/LDL)은 다음과 같은 구조를 가집니다.

    ACNE04/
    ├── Classification/          # 중증도 분류용 (Hayashi 기준 0~3 등급)
    │   ├── JPEGImages/          # 원본 얼굴 이미지 (*.jpg)
    │   └── NNEW_trainval_*.txt  # "파일명 등급" 형태의 라벨
    └── Detection/               # 병변 검출용
        ├── JPEGImages/
        ├── Annotations/         # PASCAL VOC XML (바운딩 박스)
        └── ...

이 스크립트는 Detection/Annotations 의 VOC XML을
YOLO 포맷(.txt: class cx cy w h, 0~1 정규화)으로 변환합니다.
클래스는 'lesion' 하나(단일 클래스 검출)로 둡니다.
=============================================================================
"""

import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# --------------------------------------------------------------------------
# 경로 설정 — 본인 환경에 맞게 수정
# --------------------------------------------------------------------------
ACNE04_ROOT = Path("./ACNE04")                       # 압축 푼 ACNE04 루트
VOC_IMG_DIR = ACNE04_ROOT / "Classification" / "JPEGImages"
VOC_ANN_DIR = ACNE04_ROOT / "Detection" / "Annotations"

OUT_ROOT = Path("./acne04_yolo")                     # YOLO 학습용 출력 폴더
TRAIN_RATIO = 0.8                                    # train/val 분할 비율
CLASS_NAMES = ["lesion"]                             # 단일 클래스 검출


def voc_to_yolo_bbox(size, box):
    """
    VOC 절대좌표 (xmin, ymin, xmax, ymax) -> YOLO 정규화 (cx, cy, w, h)
    size: (이미지 너비 W, 이미지 높이 H)
    """
    W, H = size
    xmin, ymin, xmax, ymax = box
    # 중심 좌표와 너비/높이를 0~1로 정규화
    cx = ((xmin + xmax) / 2.0) / W
    cy = ((ymin + ymax) / 2.0) / H
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return cx, cy, w, h


def convert_one_xml(xml_path):
    """VOC XML 한 개를 읽어 YOLO 라벨 문자열 리스트로 변환."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        # ACNE04는 모든 객체가 여드름 병변이므로 class_id = 0 으로 통일
        cls_id = 0
        bnd = obj.find("bndbox")
        box = (
            float(bnd.find("xmin").text),
            float(bnd.find("ymin").text),
            float(bnd.find("xmax").text),
            float(bnd.find("ymax").text),
        )
        cx, cy, w, h = voc_to_yolo_bbox((W, H), box)
        # 경계 밖 좌표 클리핑 (간혹 주석이 이미지 범위를 살짝 넘는 경우 방지)
        cx, cy, w, h = [min(max(v, 0.0), 1.0) for v in (cx, cy, w, h)]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def main():
    random.seed(42)  # 재현성 확보

    # 출력 폴더 구조 생성: images/{train,val}, labels/{train,val}
    for split in ["train", "val"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    xml_files = sorted(glob.glob(str(VOC_ANN_DIR / "*.xml")))
    if not xml_files:
        raise FileNotFoundError(
            f"XML 주석을 찾지 못했습니다: {VOC_ANN_DIR}\n"
            "ACNE04 Detection/Annotations 경로를 확인하세요."
        )

    random.shuffle(xml_files)
    n_train = int(len(xml_files) * TRAIN_RATIO)
    splits = {"train": xml_files[:n_train], "val": xml_files[n_train:]}

    for split, files in splits.items():
        for xml_path in files:
            stem = Path(xml_path).stem
            # 대응하는 이미지 찾기 (.jpg 가정, 필요 시 확장자 확장)
            img_src = VOC_IMG_DIR / f"{stem}.jpg"
            if not img_src.exists():
                # 일부 데이터는 대문자 확장자 등 변형이 있을 수 있음
                cand = list(VOC_IMG_DIR.glob(f"{stem}.*"))
                if not cand:
                    print(f"[경고] 이미지 없음, 건너뜀: {stem}")
                    continue
                img_src = cand[0]

            # 라벨 변환
            lines = convert_one_xml(xml_path)

            # 이미지 복사 + 라벨 저장
            shutil.copy(img_src, OUT_ROOT / "images" / split / img_src.name)
            label_out = OUT_ROOT / "labels" / split / f"{stem}.txt"
            label_out.write_text("\n".join(lines))

        print(f"[{split}] {len(files)}개 처리 완료")

    # data.yaml 생성 (YOLO 학습 시 사용)
    yaml_text = (
        f"path: {OUT_ROOT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (OUT_ROOT / "data.yaml").write_text(yaml_text)
    print(f"\n완료! data.yaml 생성됨 -> {OUT_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
