# 실험 결과: ResNet50 & ViT-S16

이 문서는 **ResNet50**과 **ViT-S16** 모델을 사전 학습 여부 및 다양한 데이터 증강 기법과 함께 실험한 결과를 정리한 문서입니다.

## 📌 실험 환경
- **모델**: ResNet50, ViT-S16
- **사전 학습 (Pretrained)**: `True` (사전 학습 사용), `False` (사전 학습 없음)
- **데이터 증강 (Augmentation)**:
  - `None`: 증강 없음
  - `GaussianBlur`: 가우시안 블러 적용
  - `RandomErasing`: 이미지 일부 영역을 랜덤하게 제거

## 📊 Results Summary

| Model      | Pretrained | Augmentation   | Accuracy (%) | Loss  |
|------------|------------|---------------|-------------|------|
| ResNet50   | False      | None          | 63.50       | 1.8066 |
| ResNet50   | False      | GaussianBlur  | 64.84       | 1.6467 |
| ResNet50   | False      | RandomErasing | 66.46       | 1.6664 |
| ResNet50   | True       | None          | 70.69       | 1.1846 |
| ResNet50   | True       | GaussianBlur  | 68.06       | 1.4880 |
| ResNet50   | True       | RandomErasing | 71.06       | 1.2148 |
| ViT-S16    | False      | None          | 23.08       | 2.1240 |
| ViT-S16    | False      | GaussianBlur  | 22.04       | 2.1962 |
| ViT-S16    | False      | RandomErasing | 20.41       | 2.2806 |
| ViT-S16    | True       | None          | 49.65       | 1.5354 |
| ViT-S16    | True       | GaussianBlur  | 47.20       | 1.4970 |
| ViT-S16    | True       | RandomErasing | 44.29       | 1.5419 |