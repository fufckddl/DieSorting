# 모델 학습 현황 문서

## 현재 모델 학습 상태

### ✅ 구현 완료된 부분
- 모델 로딩 구조 (학습된 가중치 또는 사전학습 모델 자동 선택)
- 추론 파이프라인 (학습된 모델 또는 사전학습 모델로 동작)
- GUI 애플리케이션

### ❌ 학습 관련 현황

#### 1. YOLO 검출 모델 (Step A)
- **현재 상태**: ❌ 학습 스크립트 없음
- **사용 중인 모델**: 사전학습 모델 (`yolov8n.pt`)
- **가중치 파일**: `weights/yolo_detector.pt` - 없음
- **문제점**: 
  - 일반 객체 검출용 사전학습 모델 사용 중
  - 반도체 불량 검출에 특화되지 않음
  - 신뢰도 낮음

#### 2. 패턴 분류 모델 (Step C)
- **현재 상태**: ❌ 학습 스크립트 없음, ❌ 학습 안 됨
- **사용 중인 모델**: 
  - PyTorch CNN (`SimplePatternClassifier`) - **초기화만 되고 학습 안 됨**
  - 또는 YOLOv8-cls 사전학습 모델 (일반 이미지 분류용)
- **가중치 파일**: `weights/pattern_classifier.pt` - 없음
- **문제점**:
  - 모델이 학습되지 않아 신뢰도가 매우 낮음 (약 15% = 무작위 수준)
  - 패턴 분류 정확도가 낮음

## 모델 아키텍처

### YOLO 검출 모델
- **기본 모델**: YOLOv8-nano (`yolov8n.pt`)
- **태스크**: Object Detection
- **입력**: 웨이퍼/칩 이미지
- **출력**: 불량 위치 (bbox), 신뢰도, 클래스 ID

### 패턴 분류 모델
- **모델 1**: PyTorch CNN (`SimplePatternClassifier`)
  - 아키텍처: 간단한 CNN (Conv2d → MaxPool → FC)
  - 입력: Wafer-map (128×128, 1채널)
  - 출력: 7개 패턴 클래스 확률
  - 상태: ❌ 학습 안 됨

- **모델 2**: YOLOv8-cls (옵션)
  - 기본 모델: `yolov8n-cls.pt`
  - 입력: Wafer-map 이미지 (RGB, 128×128)
  - 출력: 7개 패턴 클래스 확률
  - 상태: ❌ 일반 이미지 분류용 사전학습 모델 (반도체 패턴에 특화 안 됨)

## 학습이 필요한 이유

### 현재 문제점
1. **YOLO 검출 모델**: 
   - 일반 객체 검출용이므로 반도체 불량을 제대로 검출하지 못함
   - 검출된 불량 개수가 0개인 경우가 많음

2. **패턴 분류 모델**:
   - 학습되지 않은 모델이므로 신뢰도가 매우 낮음 (15% = 무작위 수준)
   - 실제 패턴과 무관하게 분류됨

### 학습 후 예상 결과
- YOLO 검출 모델: 반도체 불량 검출 정확도 향상
- 패턴 분류 모델: 신뢰도 70-90% 이상으로 향상

## 학습 방법

### YOLO 검출 모델 학습

1. **데이터셋 준비**
   - 반도체 불량 이미지 수집
   - YOLO 형식 annotation (YOLO format):
     ```
     class_id center_x center_y width height
     ```
   - `dataset.yaml` 파일 생성

2. **학습 명령어**
   ```bash
   yolo detect train \
       data=dataset.yaml \
       model=yolov8n.pt \
       epochs=100 \
       imgsz=640 \
       batch=16
   ```

3. **가중치 저장**
   - 학습 완료 후 `runs/detect/train/weights/best.pt`를 
   - `weights/yolo_detector.pt`로 복사

### 패턴 분류 모델 학습

1. **데이터셋 준비**
   - 각 패턴 클래스별 wafer-map 이미지 생성
   - `utils/wafer_map.py`를 사용하여 wafer-map 생성
   - 디렉터리 구조:
     ```
     dataset/
       train/
         local/
         edge-local/
         donut/
         edge-ring/
         scratch/
         near-full/
         none/
       val/
         (동일 구조)
     ```

2. **학습 스크립트** (예시)
   ```python
   # train_pattern_classifier.py (작성 필요)
   import torch
   from models.pattern_classifier import SimplePatternClassifier
   # ... 학습 코드 ...
   ```

3. **가중치 저장**
   - 학습 완료 후 `weights/pattern_classifier.pt`에 저장

## 학습 스크립트 작성 필요

현재 프로젝트에는 학습 스크립트가 없습니다. 다음 스크립트를 작성해야 합니다:

1. `train_yolo.py`: YOLO 검출 모델 학습
2. `train_pattern_classifier.py`: 패턴 분류 모델 학습

## 참고 자료

- Ultralytics YOLOv8 문서: https://docs.ultralytics.com/
- YOLOv8 학습 튜토리얼: https://docs.ultralytics.com/modes/train/


