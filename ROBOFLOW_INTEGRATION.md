# Roboflow 라벨 파일 통합 가이드

## ✅ Roboflow에서 받은 라벨 파일 사용 가능!

Roboflow에서 다운로드한 YOLO 형식 라벨 파일을 바로 사용할 수 있습니다.

---

## 📥 Roboflow에서 다운로드 받는 방법

1. Roboflow 웹사이트에서 프로젝트 열기
2. "Export" → "YOLO v5 Pytorch" 또는 "YOLO" 형식 선택
3. 다운로드 받은 압축 파일 해제

**다운로드 받은 구조:**
```
roboflow_dataset/
  train/
    images/
      001.jpg
      002.jpg
      ...
    labels/
      001.txt
      002.txt
      ...
  val/
    images/
    labels/
  test/ (선택적)
    images/
    labels/
  data.yaml
```

---

## 🔄 Roboflow 라벨 파일 통합 방법

### 방법 1: Roboflow 다운로드 파일 그대로 사용

```bash
# Roboflow에서 다운로드한 데이터셋을 yolo_dataset으로 사용
python train_yolo.py --data roboflow_dataset/data.yaml --epochs 100
```

**가중치 저장 위치:**
- 학습 완료 후 `weights/yolo_detector.pt`에 자동 저장됨

---

### 방법 2: 현재 dataset/ 폴더와 통합

Roboflow에서 라벨링한 파일을 `dataset/` 폴더에 복사:

```bash
# 예: Roboflow에서 받은 라벨 파일들을 dataset/Donut/ 폴더에 복사
# 각 이미지와 같은 이름의 .txt 파일이 있어야 함

dataset/Donut/
  679360.jpg
  679360.txt  ← Roboflow에서 받은 라벨
  679825.jpg
  679825.txt  ← Roboflow에서 받은 라벨
  ...
```

그 다음:
```bash
# YOLO 데이터셋 구조로 변환
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \
    --output-dir yolo_dataset/

# 학습
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

---

## 🎯 Roboflow 라벨 파일 형식 확인

Roboflow에서 받은 `.txt` 파일 형식:
```
0 0.5 0.5 0.2 0.3
0 0.3 0.7 0.1 0.15
```

- `0`: 클래스 ID (defect)
- `0.5 0.5`: 중심 좌표 (정규화: 0~1)
- `0.2 0.3`: 너비, 높이 (정규화: 0~1)

**이 형식은 현재 프로젝트와 100% 호환됩니다!**

---

## ✅ 학습 완료 후 박스 표시

학습이 완료되면:
1. `weights/yolo_detector.pt` 자동 생성
2. GUI에서 이미지 분석 시 **박스 테두리가 자동으로 표시됨**
3. 각 불량 위치에 정확한 박스 표시

---

## 📋 Roboflow 사용 워크플로우

### Step 1: Roboflow에서 라벨링

1. Roboflow 웹사이트 접속
2. 프로젝트 생성 또는 기존 프로젝트 사용
3. `dataset/` 폴더의 이미지 업로드
4. 웹 인터페이스에서 박스 그리기
5. YOLO 형식으로 Export

### Step 2: 다운로드 받은 파일 확인

```bash
# Roboflow에서 다운로드 받은 압축 파일 해제
unzip roboflow_dataset.zip

# 라벨 파일 확인
head roboflow_dataset/train/labels/001.txt
```

### Step 3: 학습

**옵션 A: Roboflow 다운로드 파일 그대로 사용**
```bash
python train_yolo.py --data roboflow_dataset/data.yaml --epochs 100
```

**옵션 B: dataset/ 폴더와 통합**
```bash
# 라벨 파일들을 dataset/ 폴더에 복사 후
python utils/convert_labels_to_yolo.py organize \
    --image-dir dataset/ \
    --labels-dir dataset/ \
    --output-dir yolo_dataset/
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

### Step 4: 결과 확인

```bash
python main.py
```

GUI에서 이미지를 분석하면 **박스 테두리가 정확하게 표시됩니다!**

---

## 💡 장점

1. **웹 브라우저에서 바로 사용** - 설치 불필요
2. **협업 가능** - 팀원들과 함께 라벨링
3. **자동 데이터 증강** - 옵션으로 제공
4. **YOLO 형식 자동 지원** - 호환성 완벽

---

## ⚠️ 주의사항

- Roboflow에서 다운로드할 때 **YOLO 형식**을 선택해야 함
- 클래스명은 `defect` 또는 `0` (클래스 ID)로 통일
- 이미지 파일명과 라벨 파일명이 일치해야 함

---

## 🔍 라벨 파일 검증

Roboflow에서 받은 라벨 파일이 올바른지 확인:

```bash
# 샘플 라벨 파일 확인
cat roboflow_dataset/train/labels/001.txt
```

**올바른 형식:**
```
0 0.5 0.5 0.2 0.3
0 0.3 0.7 0.1 0.15
```

**잘못된 형식 (PascalVOC 등):**
```
<class> <xmin> <ymin> <xmax> <ymax>
```

---

## ✅ 결론

**네, Roboflow에서 받은 라벨 파일로 바로 박스 테두리를 칠 수 있습니다!**

1. Roboflow에서 YOLO 형식으로 Export
2. 다운로드 받은 파일로 학습
3. 학습 완료 후 GUI에서 자동으로 박스 표시됨

Roboflow 라벨 파일을 제공해주시면 바로 통합할 수 있습니다!


