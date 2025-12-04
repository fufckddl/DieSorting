# 빠른 학습 시작 가이드

현재 `dataset/` 폴더에 패턴별로 분류된 이미지가 있습니다.

## 🎯 학습 목표

1. **패턴 분류 모델 학습**: 패턴별 이미지를 직접 학습
2. **YOLO 검출 모델 학습**: 불량 부분을 박스로 검출하기 위한 학습

---

## ✅ 방법 1: 패턴 분류 모델 학습 (즉시 가능, 권장)

패턴별로 분류된 이미지를 직접 사용하여 학습합니다.

### 실행 명령어

```bash
# 현재 데이터셋 구조 그대로 사용
python train_pattern_classifier.py --data-dir dataset/ --epochs 100
```

### 데이터 구조

현재 `dataset/` 구조 그대로 사용됩니다:
- `Local/`, `Center/` → `local` 클래스
- `Edge Local/` → `edge-local` 클래스
- `Donut/` → `donut` 클래스
- `Edge Ring/` → `edge-ring` 클래스
- `Scratch/` → `scratch` 클래스
- `near full/` → `near-full` 클래스
- `none/` → `none` 클래스

### 결과

- `weights/pattern_classifier.pt`에 학습된 모델 저장
- 패턴 분류 정확도 향상 (신뢰도 15% → 70-90% 예상)

---

## ⚠️ 방법 2: YOLO 검출 모델 학습 (bbox 라벨링 필요)

불량 부분을 박스로 표시하려면 YOLO 검출 모델 학습이 필요합니다.

### 현재 제약사항

**❌ bbox 라벨링이 필요합니다!**

현재 데이터는 패턴별로만 분류되어 있고, 각 불량의 위치(bbox) 정보가 없습니다.

### 옵션 A: 간단한 방법 (정확도 낮음)

각 이미지 전체를 하나의 불량으로 간주하고 학습:

```bash
# 1. YOLO 데이터셋 준비 (이미지 전체를 불량으로 간주)
python utils/prepare_yolo_dataset.py --source-dir dataset/ --output-dir yolo_dataset/

# 2. YOLO 학습
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

**단점:**
- 정확도가 낮음 (이미지 전체를 불량으로 간주)
- 실제 불량 위치를 정확히 검출하지 못함

### 옵션 B: 정확한 방법 (추천)

각 불량의 실제 bbox 좌표를 라벨링:

1. **라벨링 도구 사용** (예: labelImg, Roboflow)
   - 각 이미지에서 불량 부분을 박스로 표시
   - YOLO 형식으로 저장 (`.txt` 파일)

2. **데이터셋 구조 준비**
   ```
   yolo_dataset/
     train/
       images/
       labels/  # .txt 파일 (bbox 좌표)
     val/
       images/
       labels/
   ```

3. **학습 실행**
   ```bash
   python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
   ```

---

## 💡 추천 학습 전략

### Phase 1: 패턴 분류 모델 먼저 학습 (즉시 실행 가능)

```bash
python train_pattern_classifier.py --data-dir dataset/ --epochs 100
```

**이유:**
- 현재 데이터로 바로 학습 가능
- 효과가 즉시 나타남
- bbox 라벨링 불필요

### Phase 2: YOLO 검출 모델 학습 (불량 박스 표시용)

불량 부분을 박스로 표시하려면 YOLO 검출 모델 학습이 필요합니다.

#### Step 1: 라벨링 도구로 bbox 좌표 수집

**추천 라벨링 도구:**
1. **LabelImg** (가장 간단, 추천)
   ```bash
   # 설치
   pip install labelImg
   
   # 실행
   labelImg
   ```
   - 사용법:
     1. Open Dir: `dataset/` 폴더 열기
     2. 각 패턴 폴더에서 이미지 열기
     3. Create RectBox로 불량 부분 박스 그리기
     4. 클래스명: `defect` 입력
     5. Save: YOLO 형식으로 저장
     6. Next Image로 다음 이미지 진행

2. **Roboflow** (온라인, 협업 가능)
   - https://roboflow.com
   - 웹 브라우저에서 사용 가능
   - YOLO 형식 자동 지원

3. **CVAT** (고급 기능)
   - https://github.com/openvinotoolkit/cvat
   - 더 많은 기능 제공

**라벨링 결과:**
- 각 이미지마다 `.txt` 파일 생성
- YOLO 형식: `class_id center_x center_y width height` (정규화된 좌표)

#### Step 2: YOLO 데이터셋 준비

**방법 A: LabelImg 사용 시 (XML → YOLO 변환)**

```bash
# 1. LabelImg에서 XML 형식으로 저장한 경우 변환
python utils/convert_labels_to_yolo.py convert \
    --image-dir dataset/ \
    --xml-dir dataset/annotations/ \
    --output-dir dataset/labels/

# 2. YOLO 데이터셋 구조로 정리
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/labels/ \
    --output-dir yolo_dataset/
```

**방법 B: 이미 LabelImg에서 YOLO 형식으로 저장한 경우**

```bash
# LabelImg에서 직접 YOLO 형식으로 저장했다면 바로 정리
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/labels/ \  # LabelImg가 저장한 위치
    --output-dir yolo_dataset/
```

#### Step 3: YOLO 학습

```bash
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

**학습 결과:**
- `weights/yolo_detector.pt` 생성
- GUI 애플리케이션에서 자동으로 사용
- 불량 부분이 박스로 정확하게 표시됨

---

#### ⚠️ 참고: 간단한 방법 (정확도 낮음)

bbox 라벨링이 어려운 경우, 각 이미지 전체를 하나의 불량으로 간주:

```bash
# 이미지 전체를 불량으로 간주 (정확도 낮음)
python utils/prepare_yolo_dataset.py --source-dir dataset/ --output-dir yolo_dataset/
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

**주의:** 이 방법은 정확도가 낮습니다. 실제 불량 위치를 정확히 검출하지 못할 수 있습니다.

---

## 📊 학습 결과 확인

### 패턴 분류 모델 학습 후

- `weights/pattern_classifier.pt` 생성됨
- GUI 애플리케이션에서 자동으로 사용
- 패턴 분류 신뢰도 향상 확인

### YOLO 검출 모델 학습 후

- `weights/yolo_detector.pt` 생성됨
- GUI 애플리케이션에서 자동으로 사용
- 불량 부분이 박스로 표시됨

---

## ⚡ 빠른 시작 (패턴 분류만)

```bash
# 1. 패턴 분류 모델 학습
python train_pattern_classifier.py --data-dir dataset/ --epochs 100

# 2. GUI 실행하여 결과 확인
python main.py
```

---

## 📝 참고사항

- **패턴 분류 모델**: 현재 데이터로 바로 학습 가능 ✅
- **YOLO 검출 모델**: bbox 라벨링 필요하거나, 간단한 방법 사용 가능 ⚠️
- 두 모델 모두 학습하면 전체 파이프라인 성능 향상

