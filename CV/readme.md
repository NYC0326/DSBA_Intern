Vision Model Training with Augmentations

📌 Overview

본 프로젝트는 ResNet50 및 ViT-S/16 모델을 활용하여 주어진 데이터세트를 학습하고,다양한 데이터 증가 기반(Augmentation) 을 적용하여 모델의 성능을 비교하는 시험을 진행합니다.

모델: ResNet50, ViT-S/16

시험 조건:

Pretrained vs. Non-Pretrained 비교

다양한 Augmentation 기반 적용 (GaussianBlur, ColorJitter 등)

평가 방식: Accuracy & Loss 차지

결과 저장: logs/ 디렉토리에 학습 로그 및 평가 결과 저장

📂 Directory Structure

📁 project_root/
│️─📁 data/                     # 데이터세트 디렉토리 (train/test 데이터)
│️─📁 logs/                     # 학습 결과 저장 (모델별, Augmentation 별)
│️── 📁 ResNet50/              
│   │   📁 False/            # Pretrained 사용 X
│   │   📁 True/             # Pretrained 사용 O
│️── 📁 ViT-S/16/
│️─📁 models/                   # ResNet50, ViT-S/16 모델 정의
│️─📁 dataset/                  # 데이터 로더 (DatasetLoader)
│️─📁 train.py                   # 모델 학습 코드
│️─📁 evaluate.py                # 모델 평가 코드
│️─📁 experiment.py              # 시험 실행 및 결과 저장
│️─📄 README.md                  # 프로젝트 개요 및 실행 방법

🔧 Installation & Requirements

pip install torch torchvision timm tqdm matplotlib pandas

🚀 How to Run

1️⃣ 데이터세트 준비

data/ 폴더에 train_data.npy, train_target.npy, test_data.npy, test_target.npy 파일을 준비합니다.

2️⃣ 학습 실행

python experiment.py

학습이 시작되면 logs/ 폴더에 학습 결과가 자동 저장됩니다.

Augmentation별로 저장되며, 예제:

logs/
├── ResNet50/
│   ├── False/
│   │   ├── GaussianBlur/
│   │   │   ├── training_log.json
│   │   │   ├── training_results.csv
├── ViT-S/16/
│   ├── True/
│   │   ├── GaussianBlur/
│   │   ├── ColorJitter/

3️⃣ 시험 결과 확인

cat logs/ResNet50/True/GaussianBlur/training_results.csv

logs/experiment_results.csv 에 모든 시험 결과가 정보됩니다.

logs/model_comparison.png 에 모델별 성능 비교 그래프 저장됩니다.

📊 Experiment Results

(Results Table Here)

✨ Future Work

더 다양한 Augmentation 기반 추가 시험 (e.g., RandomRotation, Cutout)

더 크는 이미지 데이터세트로 확장 (32x32 보다 크게)

Gradient-based Explainability 적용 (Vanilla Gradients 등)

📌 References

DSBA Pretraining - Vision Experiments

MemSeg: Memory-Augmented Segmentation

