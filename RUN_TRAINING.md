# 라벨링 및 YOLO 학습 실행 가이드

## 현재 상태
- ✅ LabelImg 설치 완료
- ✅ 데이터셋 준비 완료 (802개 이미지)
- ⚠️ 라벨링 대기 중 (0개 완료)

---

## ⚡ 빠른 시작: 자동 검출로 시간 절약!

**800개를 일일이 라벨링하지 마세요!** 자동 검출 후 검증만 하면 됩니다:

```bash
# 1. YOLO로 자동 검출하여 라벨 파일 생성
python utils/smart_labeling_helper.py --dataset-dir dataset/

# 2. 생성된 라벨 확인 및 검증 (수정만 진행)
# LabelImg로 열어서 검출 안 된 것만 추가

# 3. 학습
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

**시간 절약:** 수동 라벨링 13-27시간 → 자동 검출 + 검증 2-4시간 (80% 절약!)

자세한 내용: `SMART_LABELING_GUIDE.md` 참고

---

## 🚀 단계별 실행 방법

### Step 0: 자동 검출로 라벨 생성 (권장)

```bash
# 모든 패턴 폴더에서 자동 검출
python utils/smart_labeling_helper.py --dataset-dir dataset/
```

이 명령으로 대부분의 이미지에 대해 라벨이 자동 생성됩니다!

### Step 1: 라벨링 진행 상황 확인

```bash
python utils/check_labeling_status.py
```

현재 상태: **802개 이미지 중 0개 라벨링 완료**

---

### Step 2: LabelImg 실행 및 라벨링

**방법 1: 스크립트 실행 (권장)**
```bash
./run_labeling.sh
```

**방법 2: 직접 실행**
```bash
source venv/bin/activate
labelImg
```

**LabelImg 사용 방법:**
1. `Open Dir` 버튼 클릭 → `dataset/Donut/` 폴더 선택
2. 좌측 상단 **"PascalVOC"** 버튼 클릭 → **"YOLO"** 선택
3. `View` → `Auto Save mode` 체크
4. 각 이미지에서:
   - `W` 키: 박스 그리기 시작
   - 마우스로 불량 부분 드래그하여 박스 생성
   - 클래스명 입력: `defect`
   - 자동 저장됨
5. `D` 키로 다음 이미지 진행
6. 모든 패턴 폴더 반복

**팁:**
- 각 패턴 폴더별로 작업하면 효율적
- 일관된 라벨링 기준 유지
- 작은 불량도 놓치지 않도록 주의

---

### Step 3: 라벨링 진행 상황 모니터링

라벨링 중간중간 진행 상황 확인:
```bash
python utils/check_labeling_status.py
```

---

### Step 4: 라벨링 완료 후 YOLO 학습 (자동화 스크립트)

라벨링이 완료되면:
```bash
./run_yolo_training.sh
```

이 스크립트가 자동으로:
1. 라벨링 상태 확인
2. YOLO 데이터셋 구조로 변환
3. YOLO 모델 학습 실행

---

### Step 4 (수동 실행): YOLO 데이터셋 구조로 변환

라벨링 완료 후:
```bash
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \
    --output-dir yolo_dataset/
```

---

### Step 5 (수동 실행): YOLO 학습

```bash
python train_yolo.py \
    --data yolo_dataset/dataset.yaml \
    --epochs 100 \
    --batch 16
```

---

### Step 6: 결과 확인

학습 완료 후:
```bash
python main.py
```

GUI에서:
1. 이미지 선택
2. "분석 시작" 클릭
3. **결과 이미지에서 불량 박스 확인** ✅

---

## ⚡ 빠른 명령어 요약

```bash
# 1. 라벨링 상태 확인
python utils/check_labeling_status.py

# 2. LabelImg 실행
./run_labeling.sh
# 또는: labelImg

# 3. 라벨링 완료 후 자동 학습
./run_yolo_training.sh

# 4. GUI 실행하여 결과 확인
python main.py
```

---

## 📊 예상 소요 시간

- **라벨링**: 이미지당 1-2분 × 802개 = 약 **13-27시간**
- **YOLO 학습**: GPU 사용 시 약 **1-2시간**, CPU 사용 시 **5-10시간**

**팁:** 샘플 데이터로 먼저 테스트
- 각 패턴별로 10-20개씩만 라벨링하여 학습 테스트
- 학습이 잘 되면 나머지 진행

