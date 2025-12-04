# Donut 모델 테스트 가이드

## 🤔 Donut만 따로 학습해야 하나요?

**아니요!** 현재 프로젝트 구조에서는 **모든 패턴을 하나의 YOLO 모델**로 학습하는 것이 맞습니다.

---

## 📊 현재 프로젝트 구조

### YOLO 검출 모델의 역할

```
[모든 패턴 이미지]
  ↓
[YOLO 검출 모델] ← 하나의 모델
  ↓
[불량 위치 검출 (클래스: "defect")]
```

**핵심:**
- ✅ **단일 YOLO 모델**이 모든 패턴의 불량을 검출
- ✅ 클래스는 **"defect"** 하나만 사용
- ✅ 패턴 분류는 별도 모델에서 수행

---

## 🎯 올바른 학습 방법

### 방법 1: 전체 데이터셋 통합 학습 (권장)

```bash
# 1. 모든 패턴 데이터셋 준비
# - dataset/Donut/ (라벨링 완료)
# - dataset/Scratch/ (라벨링 완료)
# - dataset/Local/ (라벨링 필요)
# - dataset/Edge Local/ (라벨링 필요)
# - 등등...

# 2. YOLO 데이터셋 생성
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \
    --output-dir yolo_dataset/

# 3. 통합 학습
python train_yolo.py \
    --data yolo_dataset/dataset.yaml \
    --epochs 100
```

**장점:**
- ✅ 모든 패턴에서 불량 검출 가능
- ✅ 모델이 일반화됨
- ✅ 실제 사용 환경과 유사

---

### 방법 2: Donut만 테스트 학습 (개발/테스트용)

만약 Donut 데이터만으로 **빠른 테스트**를 원한다면:

```bash
# Donut 데이터만으로 학습 (테스트용)
python train_yolo.py \
    --data dataset/DOUNT-TEST.v2i.yolov8/data.yaml \
    --epochs 50 \
    --batch 16
```

**용도:**
- ⚠️ 개발/테스트 목적
- ⚠️ Donut 이미지만 검출 가능
- ⚠️ 다른 패턴은 검출 불가

---

## 🚀 현재 상황에서 Donut 테스트 방법

### Option A: 기존 모델로 Donut 이미지 테스트

이미 학습된 모델이 있다면:

```bash
# GUI 실행
python main.py

# Donut 이미지 선택 → 분석 시작
```

**장점:** 별도 학습 불필요

---

### Option B: Donut 데이터로 빠른 학습 테스트

새로운 학습을 원한다면:

```bash
# Donut 데이터만으로 빠른 학습 (50 epochs)
python train_yolo.py \
    --data dataset/DOUNT-TEST.v2i.yolov8/data.yaml \
    --epochs 50 \
    --batch 16

# 학습 완료 후 GUI 테스트
python main.py
```

---

## 💡 추천 워크플로우

### 1단계: 현재 Donut 데이터로 테스트 학습 (빠른 검증)

```bash
# Donut만 학습 (약 30분~1시간)
python train_yolo.py \
    --data dataset/DOUNT-TEST.v2i.yolov8/data.yaml \
    --epochs 50 \
    --batch 16
```

**목적:** 
- 라벨링이 올바른지 확인
- 학습 파이프라인 테스트
- Donut 검출 결과 확인

### 2단계: 다른 패턴 데이터 준비

```bash
# Scratch 데이터 변환 (이미 완료)
# - dataset/Scratch.v3i.yolov8_fixed/

# 나머지 패턴 라벨링 진행
# - Local, Edge Local, Edge Ring 등
```

### 3단계: 전체 데이터셋 통합 학습

```bash
# 모든 패턴 통합
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \
    --output-dir yolo_dataset/

# 전체 학습 (약 2-3시간)
python train_yolo.py \
    --data yolo_dataset/dataset.yaml \
    --epochs 100 \
    --batch 16
```

---

## 📋 결론

| 방법 | 학습 필요? | 검출 범위 | 용도 |
|------|-----------|----------|------|
| **기존 모델 사용** | ❌ 불필요 | 모든 패턴 | 빠른 테스트 |
| **Donut만 학습** | ✅ 필요 | Donut만 | 개발/테스트 |
| **전체 학습** | ✅ 필요 | 모든 패턴 | 실제 사용 |

---

## ✅ 권장사항

**현재 상황:**
1. **Donut 데이터만으로 빠른 테스트 학습** (50 epochs)
   - 라벨링 확인
   - 학습 파이프라인 검증
   - Donut 검출 결과 확인

2. **다른 패턴 데이터 준비 후 통합 학습**
   - 실제 사용 환경과 동일
   - 모든 패턴 검출 가능

---

## 🚀 지금 바로 테스트하기

**Donut만 빠르게 학습:**

```bash
cd "/Users/dlckdfuf/Desktop/반도체 졸업프로젝트"
source venv/bin/activate

python train_yolo.py \
    --data dataset/DOUNT-TEST.v2i.yolov8/data.yaml \
    --epochs 50 \
    --batch 16
```

**학습 완료 후:**
```bash
python main.py
# GUI에서 Donut 이미지 테스트
```

---

**요약:** Donut만 따로 학습할 필요는 없지만, **테스트 목적으로 빠르게 학습**해볼 수 있습니다. 최종적으로는 **모든 패턴을 통합하여 하나의 모델**로 학습하는 것이 맞습니다!


