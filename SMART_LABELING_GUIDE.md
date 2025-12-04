# 스마트 라벨링 가이드 - 800개 효율적으로 처리하기

## 😰 문제: 800개 이미지를 일일이 라벨링?

**시간 계산:**
- 이미지당 1-2분 × 800개 = **13-27시간** 
- 매우 비효율적!

---

## 💡 해결책: 반자동 라벨링 (자동 검출 + 검증)

### 방법 1: 자동 검출 후 검증 (가장 효율적 ⭐⭐⭐)

**아이디어:**
1. 현재 YOLO 모델로 자동 검출
2. 라벨 파일 자동 생성
3. 사용자는 **검증 및 수정만** 진행

**시간 절약:**
- 자동 검출: 몇 분
- 검증/수정: 이미지당 10-30초
- **예상 시간: 2-4시간** (80% 시간 절약!)

#### 실행 방법:

```bash
# 1. 모든 패턴 폴더에서 자동 검출
python utils/smart_labeling_helper.py --dataset-dir dataset/

# 또는 특정 패턴만
python utils/smart_labeling_helper.py --dataset-dir dataset/ --pattern Donut
```

**결과:**
- 각 이미지와 같은 위치에 `.txt` 파일 자동 생성
- 신뢰도가 높은 검출 결과만 저장

**다음 단계:**
1. LabelImg로 열어서 확인
2. **검증 및 수정만** 진행:
   - 검출되지 않은 불량 추가
   - 잘못된 박스 삭제/수정
   - 이미 검출된 것들은 그대로 유지

---

### 방법 2: 샘플링 + 학습 테스트 (단계적 접근 ⭐⭐)

**전략:**
1. 각 패턴별로 **10-20개씩만** 먼저 라벨링
2. 학습하여 성능 확인
3. 문제없으면 나머지 진행 또는 자동 검출 활용

#### 실행:

```bash
# 1. 샘플 데이터만 수동 라벨링 (약 100-150개)
# 2. 학습 테스트
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 50

# 3. 학습된 모델로 나머지 자동 검출
python utils/smart_labeling_helper.py --dataset-dir dataset/
```

---

### 방법 3: 자동 검출 → 패턴별 분류 활용 (현실적 ⭐⭐)

**전략:**
1. 현재 YOLO로 자동 검출 (정확도 낮아도 OK)
2. Wafer-map 생성
3. 이미 패턴별로 분류된 데이터 활용
4. 패턴 분류 모델 학습 (YOLO 학습 없이도 가능)

#### 실행:

```bash
# 1. 자동 검출로 라벨 생성 (부정확해도 OK)
python utils/smart_labeling_helper.py --dataset-dir dataset/ --min-confidence 0.2

# 2. 패턴 분류 모델만 학습 (이미 패턴별로 분류되어 있음)
python train_pattern_classifier.py --data-dir dataset/ --epochs 100

# 3. YOLO 학습은 나중에 시간 있을 때 수동 검증 후 진행
```

---

## 🚀 추천 워크플로우

### Phase 1: 빠른 시작 (패턴 분류만)

```bash
# 패턴 분류 모델만 학습 (라벨링 불필요!)
python train_pattern_classifier.py --data-dir dataset/ --epochs 100
```

**이유:**
- 현재 데이터로 바로 학습 가능
- 효과 즉시 확인
- 패턴 분류 성능 크게 향상

---

### Phase 2: YOLO 반자동 라벨링 (시간 절약)

```bash
# 1. 자동 검출로 라벨 생성
python utils/smart_labeling_helper.py --dataset-dir dataset/

# 2. 진행 상황 확인
python utils/check_labeling_status.py

# 3. LabelImg로 검증 (수정만)
#    - 검출 안 된 것만 추가
#    - 잘못된 것만 삭제

# 4. YOLO 학습
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

---

## ⏱️ 예상 시간 비교

| 방법 | 소요 시간 | 정확도 |
|------|----------|--------|
| **완전 수동** | 13-27시간 | 높음 |
| **자동 검출 + 검증** | 2-4시간 | 높음 |
| **샘플링 + 자동** | 3-5시간 | 중간-높음 |
| **패턴 분류만** | 즉시 가능 | 중간 |

---

## 💡 실용적인 접근 방법

### 옵션 A: 패턴 분류 먼저 (즉시 가능, 권장)

```bash
# 라벨링 없이 바로 시작!
python train_pattern_classifier.py --data-dir dataset/ --epochs 100
```

**장점:**
- 즉시 시작 가능
- 패턴 분류 정확도 향상
- YOLO는 나중에

---

### 옵션 B: 반자동 라벨링 (시간 절약)

```bash
# 1. 자동 검출
python utils/smart_labeling_helper.py --dataset-dir dataset/

# 2. LabelImg로 검증 (검출 안 된 것만 추가)
# 예상: 전체의 20-30%만 수정 필요

# 3. 학습
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

---

### 옵션 C: 샘플링 (단계적)

```bash
# 각 패턴별로 10-20개만 먼저 라벨링
# → 학습 테스트
# → 문제없으면 자동 검출로 나머지 처리
```

---

## 🎯 최종 권장사항

1. **먼저 패턴 분류 모델 학습** (라벨링 없이)
   - 효과 즉시 확인
   - 전체 파이프라인 개선

2. **필요하면 YOLO 반자동 라벨링**
   - 자동 검출 스크립트 사용
   - 검증만 진행

3. **완전 수동 라벨링은 최후의 수단**
   - 정확도가 매우 중요한 경우만
   - 또는 자동 검출 결과가 너무 나쁜 경우만

---

## 📊 자동 검출 품질 확인

```bash
# 자동 검출 후 품질 확인
python utils/check_labeling_status.py

# 샘플 이미지로 검출 결과 확인
# (GUI 애플리케이션에서 확인 가능)
```

결과가 나쁘면:
- 신뢰도 임계값 조정
- 수동 검증 비율 증가
- 샘플링으로 재시도


