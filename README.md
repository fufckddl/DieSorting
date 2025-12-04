# 반도체 Die Sorting 불량 패턴 분류 시스템

반도체 웨이퍼/칩 이미지에서 불량 패턴을 분류하고 시각화하는 PyQt 기반 데스크톱 애플리케이션입니다.

## 주요 기능

- **4단계 파이프라인 구조**
  1. Step A: YOLOv8 기반 불량 검출 (detection/segmentation)
  2. Step B: 불량 분포 맵 (wafer-map) 생성
  3. Step C: 패턴 분류 (local, edge-local, donut, edge-ring, scratch, near-full, none)
  4. Step D: 불량 영역 시각화

- **PyQt GUI**
  - 2×2 그리드 레이아웃
  - 이미지 선택 및 분석 시작
  - 분석 결과 시각화 (불량 영역 색칠)
  - 상세 불량 결과 테이블
  - 이미지 확대/축소 기능

## 프로젝트 구조

```
반도체 졸업프로젝트/
├── main.py                   # 애플리케이션 엔트리 포인트
├── config.py                 # 설정 파일
├── requirements.txt          # 의존성 목록
├── gui/
│   ├── __init__.py
│   └── main_window.py        # PyQt 메인 윈도우
├── inference/
│   ├── __init__.py
│   └── pipeline.py           # 전체 파이프라인
├── models/
│   ├── __init__.py
│   ├── yolo_detector.py      # YOLOv8 검출 모델
│   └── pattern_classifier.py # 패턴 분류 모델
└── utils/
    ├── __init__.py
    ├── wafer_map.py          # Wafer map 생성 유틸
    └── visualization.py      # 시각화 유틸
```

## 설치 방법

1. **의존성 설치**
```bash
pip install -r requirements.txt
```

2. **모델 가중치 준비** (선택적)
   - `weights/yolo_detector.pt`: YOLO 검출 모델 가중치
   - `weights/pattern_classifier.pt`: 패턴 분류 모델 가중치
   - 가중치 파일이 없으면 사전학습 모델이 자동으로 다운로드됩니다.

## 실행 방법

### GUI 애플리케이션 실행
```bash
python main.py
```

### 사용 방법
1. 메뉴바에서 "파일 > 이미지 열기" 또는 "분석 시작" 버튼 클릭으로 이미지 선택
2. "분석 시작" 버튼 클릭
3. 분석 결과 확인:
   - 우상단: 불량 영역이 색칠된 시각화 이미지
   - 좌하단: 개별 불량 검출 결과 테이블
   - 좌상단: 파일명, 분석 시간, 이미지 크기 정보

## 파이프라인 아키텍처

### 설계 원리
YOLO만으로는 bbox 기반 검출만 가능하므로, donut/edge-ring/near-full 같은 **분포형 패턴**을 직접 설명하기 어렵습니다. 따라서:
- **YOLO**: 개별 불량의 위치를 정확히 검출 (Step A)
- **Wafer-map**: 검출 결과를 공간적 분포 맵으로 변환 (Step B)
- **분류 모델**: 분포 맵을 입력으로 패턴 클래스를 분류 (Step C)
- **시각화**: 패턴별 의미 영역을 색칠하여 설명 가능한 시각화 제공 (Step D)

### 데이터 흐름
```
이미지 선택
    ↓
run_full_pipeline(image_path)
    ↓
[Step A] YOLO 검출 → detections 리스트
    ↓
[Step B] Wafer-map 생성 → 2D 배열 (128×128)
    ↓
[Step C] 패턴 분류 → class_label, confidence
    ↓
[Step D] 시각화 → vis_image (BGR)
    ↓
결과 딕셔너리 반환 → GUI에 표시
```

## 모델 학습 (향후 작업)

### YOLO 검출 모델 학습
```bash
# ultralytics CLI 사용
yolo detect train data=your_dataset.yaml model=yolov8n.pt epochs=100
```

### 패턴 분류 모델 학습
- `utils/wafer_map.py`로 wafer-map 생성
- PyTorch 또는 YOLOv8-cls로 분류 모델 학습
- 가중치를 `weights/pattern_classifier.pt`에 저장

## 설정

`config.py`에서 다음 설정을 변경할 수 있습니다:
- `YOLO_MODEL_SIZE`: YOLO 모델 크기 ("n", "s", "m", "l", "x")
- `GRID_SIZE`: Wafer-map 해상도 (기본: 128×128)
- `CONFIDENCE_THRESHOLD`: YOLO 신뢰도 임계값
- `PATTERN_CLASSES`: 분류할 패턴 클래스 목록

## 라이선스

이 프로젝트는 졸업 프로젝트용으로 개발되었습니다.


