"""
=============================================================================
04_cnn_classify_and_aggregate.py
파이프라인 5~6단계: CNN(EfficientNet-B0) ROI 분류 + 다중 ROI 통합
=============================================================================

5단계: 각 ROI를 EfficientNet-B0로 피부질환 분류 (여드름/모낭염/접촉성피부염/습진 등)
6단계: 여러 ROI의 확률을 (confidence x area) 가중 평균으로 통합해
       이미지 단위 최종 예측 확률 산출

학습 데이터: DermNet 또는 10-class Kaggle 데이터셋(분류 라벨)
필요 패키지: pip install torch torchvision
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


# 분류할 피부질환 클래스 (데이터셋에 맞게 수정)
CLASSES = ["여드름", "모낭염", "접촉성피부염", "습진/아토피", "기타"]


# --------------------------------------------------------------------------
# 5-A. 모델 정의
# --------------------------------------------------------------------------
def build_classifier(num_classes=len(CLASSES), pretrained=True):
    """EfficientNet-B0 기반 분류기. 마지막 분류층만 클래스 수에 맞게 교체."""
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    )
    in_feat = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feat, num_classes)
    return model


# --------------------------------------------------------------------------
# 5-B. 추론 (ROI 1개 -> 클래스별 확률)
# --------------------------------------------------------------------------
@torch.no_grad()
def classify_roi(model, roi_array, device="cpu"):
    """
    roi_array: 03단계 preprocess_roi 출력 (H,W,3) 이미 표준화됨.
    반환: numpy 확률벡터 (num_classes,)
    """
    model.eval()
    # (H,W,C) -> (1,C,H,W)
    x = torch.from_numpy(roi_array.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


# --------------------------------------------------------------------------
# 6. 다중 ROI 통합 (가중 평균)
# --------------------------------------------------------------------------
def aggregate_rois(roi_results):
    """
    다이어그램 6번 수식 구현:
        최종확률 = Σ(w_i * p_i) / Σ(w_i),   w_i = confidence_i x area_i

    roi_results: [{'probs': np.array, 'conf': float, 'area': int}, ...]
    반환: (최종 확률벡터, 최종 예측 클래스명)
    """
    if not roi_results:
        return None, None

    weights = np.array([r["conf"] * r["area"] for r in roi_results], dtype=np.float64)
    probs = np.stack([r["probs"] for r in roi_results])  # (N, C)

    if weights.sum() == 0:
        final = probs.mean(axis=0)
    else:
        final = (weights[:, None] * probs).sum(axis=0) / weights.sum()

    pred_idx = int(final.argmax())
    return final, CLASSES[pred_idx]


# --------------------------------------------------------------------------
# 5~6 통합 실행: 이미지 한 장 -> 질환 확률
# --------------------------------------------------------------------------
def predict_image_disease(clf_model, rois, device="cpu"):
    """
    rois: 03단계 extract_rois 출력 리스트.
    반환: dict (클래스별 최종 확률 + 예측 클래스)
    """
    roi_results = []
    for r in rois:
        probs = classify_roi(clf_model, r["roi"], device=device)
        roi_results.append({"probs": probs, "conf": r["conf"], "area": r["area"]})

    final, pred = aggregate_rois(roi_results)
    if final is None:
        return {"prediction": None, "probs": None}

    return {
        "prediction": pred,
        "probs": {c: round(float(p), 4) for c, p in zip(CLASSES, final)},
        "n_rois": len(rois),
    }



# --------------------------------------------------------------------------
# 5-C. CNN 모델 학습 루프 (추가된 기능)
# --------------------------------------------------------------------------
def train_cnn_model(data_dir, epochs=10, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    폴더별로 정리된 이미지(data_dir/여드름, data_dir/모낭염 등)를 불러와 학습하고 가중치를 저장합니다.
    """
    import torch.optim as optim
    from torchvision import datasets
    from torch.utils.data import DataLoader

    print(f"디바이스 [{device}] 로 학습을 시작합니다...")

    # 이미지 증강 및 전처리 (03단계의 전처리와 유사하게)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 데이터셋 로드
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = build_classifier(num_classes=len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # 학습 완료 후 저장
    torch.save(model.state_dict(), "best_cnn.pth")
    print("✅ CNN 가중치가 'best_cnn.pth'로 저장되었습니다!")
    return model

# --------------------------------------------------------------------------
# 실행 블록
# --------------------------------------------------------------------------


if __name__ == "__main__":
    # 데모: 랜덤 ROI로 파이프라인 동작 확인 (실제로는 03단계 출력 사용)
    model = build_classifier()
    fake_rois = [
        {"roi": np.random.randn(224, 224, 3).astype(np.float32),
         "conf": 0.9, "area": 1500},
        {"roi": np.random.randn(224, 224, 3).astype(np.float32),
         "conf": 0.7, "area": 900},
    ]
    out = predict_image_disease(model, fake_rois)
    print(out)
    train_cnn_model(data_dir="./dermnet_dataset/train", epochs=15)
    pass
