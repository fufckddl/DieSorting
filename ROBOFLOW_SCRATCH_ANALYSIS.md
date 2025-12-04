# Roboflow Scratch 데이터셋 분석 결과

## 📊 데이터셋 구조

```
Scratch.v3i.yolov8/
  ├── train/
  │   ├── images/ (35개)
  │   └── labels/ (35개)
  ├── valid/
  │   ├── images/ (5개)
  │   └── labels/ (5개)
  ├── test/
  │   ├── images/ (4개)
  │   └── labels/ (4개)
  ├── data.yaml
  └── README.roboflow.txt
```

**총 44개 이미지** (Train: 35, Valid: 5, Test: 4)

---

## ⚠️ 발견된 문제점

### 1. 라벨 파일 형식 불일치

일부 라벨 파일이 **YOLO detection 형식(5개 값)**이 아니라 **YOLO segmentation 형식(다수 좌표)**으로 되어 있습니다.

**올바른 형식 (YOLO detection):**
```
0 0.5 0.5 0.2 0.3
```
- `0`: 클래스 ID
- `0.5 0.5`: 중심 좌표
- `0.2 0.3`: 너비, 높이

**문제 형식 (YOLO segmentation):**
```
0 0.78125 0.125 0.8421875 0.125 0.8421875 0.1859375 ...
```
- `0`: 클래스 ID
- 이후: 다수의 (x, y) 좌표 쌍 (폴리곤 좌표)

**영향:**
- YOLOv8-det 모델은 detection 형식만 지원
- Segmentation 형식은 변환 필요

---

### 2. 클래스 이름 불일치

- **Roboflow**: `'scratch'` (data.yaml)
- **현재 프로젝트**: `'defect'` (단일 클래스)

**해결:**
- 클래스 이름은 학습에는 영향 없음 (클래스 ID 0으로 통일)
- 하지만 일관성을 위해 `'defect'`로 변경 권장

---

### 3. 이미지 파일명 변경

Roboflow가 파일명을 변경:
- 원본: `801037.jpg`
- Roboflow: `801037_jpg.rf.4c883178659e6f163898de1501855ab1.jpg`

**영향:**
- 현재 `dataset/Scratch/` 폴더의 이미지와 매칭이 어려울 수 있음

---

## ✅ 해결 방법

### 방법 1: Segmentation → Detection 변환 (추천)

Segmentation 좌표를 bounding box로 변환:

```python
# utils/convert_segmentation_to_detection.py 생성 필요
```

**변환 로직:**
1. Segmentation 좌표에서 최소/최대 x, y 찾기
2. Bounding box 계산: `(center_x, center_y, width, height)`

### 방법 2: YOLOv8-seg 모델 사용

Segmentation 모델로 변경 (더 정확하지만 학습 시간 증가)

### 방법 3: Detection 형식 라벨만 사용

Segmentation 형식 라벨을 제외하고 Detection 형식만 사용

---

## 🎯 현재 상태

### ✅ 바로 사용 가능한 항목

1. **YOLO detection 형식 라벨 파일들** (일부)
2. **Train/Val/Test 분할 완료**
3. **이미지-라벨 매칭 완료**

### ❌ 수정이 필요한 항목

1. **Segmentation 형식 라벨 파일** → Detection 형식으로 변환
2. **data.yaml 클래스 이름** → `'defect'`로 변경 (선택적)
3. **이미지 파일명** → 원본과 매칭 (선택적)

---

## 🚀 권장 작업 순서

### Step 1: Segmentation → Detection 변환 스크립트 생성

```bash
python utils/convert_segmentation_to_detection.py \
    --input-dir dataset/Scratch.v3i.yolov8 \
    --output-dir dataset/Scratch.v3i.yolov8_fixed
```

### Step 2: 변환된 데이터셋 검증

```bash
python utils/import_roboflow_labels.py \
    dataset/Scratch.v3i.yolov8_fixed \
    --validate-only
```

### Step 3: 학습 실행

```bash
# 방법 A: Roboflow 데이터셋 그대로 사용
python train_yolo.py \
    --data dataset/Scratch.v3i.yolov8_fixed/data.yaml \
    --epochs 100

# 방법 B: 전체 프로젝트와 통합
python utils/import_roboflow_labels.py \
    dataset/Scratch.v3i.yolov8_fixed \
    --target-dir dataset/Scratch
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \
    --output-dir yolo_dataset/
python train_yolo.py \
    --data yolo_dataset/dataset.yaml \
    --epochs 100
```

---

## 📋 다음 단계

1. **Segmentation → Detection 변환 스크립트 생성** (필수)
2. **변환 후 검증**
3. **학습 실행**
4. **GUI에서 테스트**

---

## 💡 참고

- **Roboflow에서 재다운로드**: Export 시 "YOLOv8 Detection" 형식 선택하면 문제 해결 가능
- **현재 데이터셋 사용**: 변환 스크립트로 해결 가능


