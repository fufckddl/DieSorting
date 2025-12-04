# 올바른 학습 순서

## ⚠️ 잘못된 순서

```
1. 현재 학습 완료 (잘못된 라벨로 학습)
2. 자동 검출 실행 ❌
```

**문제점:**
- 잘못 학습된 모델은 실제 불량 위치를 검출하지 못함
- 자동 검출도 부정확함

---

## ✅ 올바른 순서

### 옵션 1: 현재 학습 중단 후 재시작 (권장)

```
1. 현재 학습 중단
2. 자동 검출로 실제 불량 위치 라벨 생성 (사전학습 YOLO 사용)
3. 생성된 라벨로 재학습
4. 학습 완료 후 사용
```

### 옵션 2: 현재 학습 완료 후 수동 라벨링

```
1. 현재 학습 완료 (잘못된 모델이지만 일단 완료)
2. LabelImg로 실제 불량 위치 수동 라벨링
3. 재학습
```

---

## 🚀 권장 실행 순서 (옵션 1)

### Step 1: 현재 학습 중단

```bash
# 학습 프로세스 확인
ps aux | grep train_yolo

# 학습 중단 (프로세스 ID로)
kill [프로세스_ID]
```

### Step 2: 자동 검출로 실제 불량 위치 라벨 생성

**중요:** 현재 사전학습 YOLO 모델을 사용 (아직 잘못 학습된 모델 아님)

```bash
python utils/smart_labeling_helper.py --dataset-dir dataset/
```

**이 시점에서 사용되는 모델:**
- `weights/yolo_detector.pt`가 없으면
- 사전학습 모델 (`yolov8n.pt`) 사용
- 일반 객체 검출 모델이지만, 불량 위치를 어느 정도 검출 가능

### Step 3: 생성된 라벨 확인

```bash
python utils/check_labeling_status.py
```

### Step 4: YOLO 데이터셋 재생성

```bash
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \
    --output-dir yolo_dataset/
```

### Step 5: 재학습

```bash
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

---

## 📊 순서 비교

| 순서 | 방법 | 결과 |
|------|------|------|
| ❌ 잘못됨 | 학습 완료 → 자동 검출 | 잘못된 모델로 검출, 부정확 |
| ✅ 올바름 | 자동 검출 → 재학습 | 실제 위치 학습, 정확함 |

---

## 💡 핵심 포인트

**자동 검출 도구가 사용하는 모델:**
- `weights/yolo_detector.pt`가 있으면: 학습된 모델 사용
- 없으면: 사전학습 모델 (`yolov8n.pt`) 사용

**따라서:**
- ✅ 학습 전에 실행: 사전학습 모델 사용 → 어느 정도 검출 가능
- ❌ 잘못 학습된 모델로 실행: 부정확한 검출

---

## 🎯 결론

**현재 학습을 중단하고 다음 순서로 진행하는 것을 권장합니다:**

1. 학습 중단
2. 자동 검출 실행 (사전학습 모델 사용)
3. 생성된 라벨로 재학습

이렇게 하면 실제 불량 위치를 학습하여 정확한 박스 표시가 가능합니다.


