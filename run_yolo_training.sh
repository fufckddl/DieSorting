#!/bin/bash
# YOLO 학습 자동 실행 스크립트

cd "$(dirname "$0")"
source venv/bin/activate

echo "YOLO 학습 파이프라인 실행"
echo "=" * 60

# Step 1: 라벨링 상태 확인
echo "Step 1: 라벨링 상태 확인 중..."
python utils/check_labeling_status.py

read -p "라벨링이 완료되었나요? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "라벨링을 먼저 완료해주세요."
    echo "LabelImg 실행: ./run_labeling.sh"
    exit 1
fi

# Step 2: YOLO 데이터셋 구조로 변환
echo ""
echo "Step 2: YOLO 데이터셋 구조로 변환 중..."
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \
    --output-dir yolo_dataset/

if [ $? -ne 0 ]; then
    echo "❌ 데이터셋 변환 실패"
    exit 1
fi

# Step 3: YOLO 학습
echo ""
echo "Step 3: YOLO 모델 학습 시작..."
echo "학습 시간이 오래 걸릴 수 있습니다."
echo ""

python train_yolo.py \
    --data yolo_dataset/dataset.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 640

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 학습 완료!"
    echo "weights/yolo_detector.pt 파일이 생성되었습니다."
    echo ""
    echo "다음 단계:"
    echo "python main.py  # GUI 실행하여 결과 확인"
else
    echo "❌ 학습 실패"
    exit 1
fi


