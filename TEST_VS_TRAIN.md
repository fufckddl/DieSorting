# 학습 vs 테스트 구분 가이드

## 🤔 명령어 구분

### ✅ 학습 명령어 (모델 훈련)

```bash
python train_yolo.py --data dataset/DOUNT-TEST.v2i.yolov8/data.yaml --epochs 50 --batch 16
```

**이 명령어는:**
- ✅ YOLO 모델을 **학습(훈련)**합니다
- ✅ 데이터셋을 보고 모델이 패턴을 학습합니다
- ✅ 시간이 오래 걸립니다 (30분~수시간)
- ✅ 학습 완료 후 `weights/yolo_detector.pt` 파일이 생성됩니다

---

### ✅ 테스트 명령어 (GUI로 결과 확인)

```bash
python main.py
```

**이 명령어는:**
- ✅ 학습된 모델로 **이미지 분석**을 수행합니다
- ✅ GUI 창이 열리고 이미지를 선택하여 테스트할 수 있습니다
- ✅ 학습된 모델이 불량을 검출하고 박스를 표시합니다

---

## 📋 현재 상황에서 Donut 테스트하는 방법

### 시나리오 1: 이미 학습된 모델이 있는 경우

```bash
# 1. GUI 실행 (학습 불필요)
python main.py

# 2. GUI에서 Donut 이미지 선택
# 3. "분석 시작" 버튼 클릭
# 4. 결과 확인 (박스 표시 확인)
```

**학습 명령어 불필요!** ✅

---

### 시나리오 2: 아직 학습된 모델이 없는 경우

```bash
# 1단계: 먼저 학습 필요 (필수!)
python train_yolo.py \
    --data dataset/DOUNT-TEST.v2i.yolov8/data.yaml \
    --epochs 50 \
    --batch 16

# 학습 완료까지 대기 (30분~1시간)

# 2단계: 학습 완료 후 GUI로 테스트
python main.py

# 3. GUI에서 Donut 이미지 선택
# 4. "분석 시작" 버튼 클릭
# 5. 결과 확인
```

**학습 먼저 필요!** ⚠️

---

## 🎯 요약

| 작업 | 명령어 | 시간 | 목적 |
|------|--------|------|------|
| **학습 (Train)** | `python train_yolo.py ...` | 30분~수시간 | 모델 훈련 |
| **테스트 (Test)** | `python main.py` | 즉시 | 결과 확인 |

---

## 💡 질문에 대한 답변

**Q: "이 명령어는 학습하는 파이썬 파일을 실행하는 아니야?"**

**A: 네, 맞습니다!** 

- `python train_yolo.py` = **학습 실행**
- `python main.py` = **테스트 실행 (GUI)**

**"Donut 모델 테스트"를 하려면:**

1. **학습된 모델이 이미 있으면:**
   - ❌ 학습 명령어 불필요
   - ✅ `python main.py`만 실행하면 됨

2. **학습된 모델이 없으면:**
   - ✅ 먼저 학습 필요 (`python train_yolo.py`)
   - ✅ 학습 완료 후 `python main.py`로 테스트

---

## ✅ 현재 추천 방법

**옵션 1: 기존 모델 확인 (빠름)**
```bash
# 학습된 모델 있는지 확인
ls -lh weights/yolo_detector.pt

# 있으면 바로 테스트
python main.py
```

**옵션 2: Donut 데이터로 새로 학습 (시간 소요)**
```bash
# 학습 실행
python train_yolo.py \
    --data dataset/DOUNT-TEST.v2i.yolov8/data.yaml \
    --epochs 50 \
    --batch 16

# 학습 완료 후 테스트
python main.py
```

---

**결론:** 네, `train_yolo.py`는 학습 파일입니다! 테스트는 `main.py`로 하면 됩니다. 🎯


