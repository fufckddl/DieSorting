# 라벨링 가이드

불량 부분을 박스로 표시하기 위한 라벨링 작업 가이드입니다.

## 라벨링 도구 추천

### 1. LabelImg (가장 간단, 추천 ⭐)

**설치:**
```bash
pip install labelImg
```

**실행:**
```bash
labelImg
```

**사용 방법:**

1. **Open Dir**: `dataset/` 폴더 또는 각 패턴 폴더 열기
   - 예: `dataset/Donut/` 폴더 열기

2. **YOLO 형식 선택**:
   - 메뉴: View → Auto Save mode 체크
   - 메뉴: View → Display Labels 체크
   - **중요**: 좌측 상단 "PascalVOC" 버튼 클릭 → "YOLO"로 변경

3. **라벨링 작업**:
   - `W` 키: 박스 그리기 시작
   - 마우스로 불량 부분 드래그하여 박스 생성
   - **클래스명 입력: `defect`** ⚠️ 중요!
     - donut, local, edge-ring 등 **모든 패턴 폴더에서 동일하게 "defect" 사용**
     - YOLO는 불량 위치만 검출하고, 패턴 분류는 별도 모델에서 수행
     - 자세한 설명: `LABELING_EXPLANATION.md` 참고
   - `Ctrl + S`: 저장 (Auto Save 모드면 자동 저장)

4. **다음 이미지**:
   - `D` 키: 다음 이미지
   - `A` 키: 이전 이미지

5. **라벨 파일 위치**:
   - LabelImg는 이미지와 같은 디렉터리에 `.txt` 파일 생성
   - 또는 설정에서 별도 디렉터리 지정 가능

**라벨 파일 형식 (YOLO):**
```
0 0.5 0.5 0.1 0.1
```
- `0`: 클래스 ID
- `0.5 0.5`: 중심 좌표 (정규화: 0~1)
- `0.1 0.1`: 너비, 높이 (정규화: 0~1)

---

### 2. Roboflow (온라인, 협업 가능)

**장점:**
- 웹 브라우저에서 바로 사용
- 팀 협업 기능
- 데이터 증강 자동 제공
- YOLO 형식 자동 지원

**사용 방법:**
1. https://roboflow.com 접속
2. 새 프로젝트 생성
3. 이미지 업로드
4. 웹 인터페이스에서 박스 그리기
5. YOLO 형식으로 다운로드

---

### 3. CVAT (고급 기능)

**설치:**
```bash
# Docker로 실행
docker run -d \
  --name cvat \
  -p 8080:8080 \
  openvino/cvat_server
```

**장점:**
- 더 많은 라벨링 기능
- 팀 협업
- 검증 워크플로우

---

## 라벨링 작업 순서

### 1단계: 라벨링 도구 실행

```bash
# LabelImg 실행
labelImg
```

### 2단계: 각 패턴 폴더별로 라벨링

1. `dataset/Donut/` 폴더 열기
2. 각 이미지에서 불량 부분 박스 그리기
3. 모든 이미지 라벨링 완료

4. `dataset/Local/` 폴더 열기
5. 반복...

**팁:**
- 패턴별로 나눠서 작업하면 효율적
- 일관된 라벨링 기준 유지
- 작은 불량도 놓치지 않도록 주의

### 3단계: 라벨 파일 확인

각 이미지와 같은 위치에 `.txt` 파일이 생성되었는지 확인:

```
dataset/
  Donut/
    679360.jpg
    679360.txt  ← 라벨 파일
    679825.jpg
    679825.txt
    ...
```

### 4단계: YOLO 데이터셋 구조로 변환

```bash
# LabelImg에서 YOLO 형식으로 저장했다면
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \  # 이미지와 같은 디렉터리
    --output-dir yolo_dataset/
```

또는 LabelImg에서 별도 디렉터리에 저장했다면:

```bash
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/labels/ \
    --output-dir yolo_dataset/
```

---

## 라벨링 품질 체크리스트

- [ ] 모든 이미지에 라벨이 생성되었는가?
- [ ] 각 불량 부분이 정확히 박스로 표시되었는가?
- [ ] 라벨 파일 형식이 YOLO 형식인가?
- [ ] 클래스명이 일관되게 입력되었는가?

---

## 라벨링 시간 절약 팁

1. **패턴별 우선순위**: 중요한 패턴부터 라벨링
2. **샘플링**: 각 패턴별로 일부만 라벨링하여 먼저 학습 테스트
3. **자동화 도구**: 이미 라벨링된 데이터가 있다면 활용
4. **협업**: 여러 사람이 나눠서 작업

---

## 다음 단계

라벨링 완료 후:

```bash
# YOLO 데이터셋 준비
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \
    --output-dir yolo_dataset/

# YOLO 학습
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

