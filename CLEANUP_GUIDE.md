# 프로젝트 정리 가이드

## 📁 라벨링해야 할 데이터셋 폴더

**라벨링 대상:**
- `dataset/Center/` (100개 이미지)
- `dataset/Donut/` (102개 이미지)
- `dataset/Edge Local/` (103개 이미지)
- `dataset/Edge Ring/` (102개 이미지)
- `dataset/Local/` (100개 이미지)
- `dataset/near full/` (95개 이미지)
- `dataset/none/` (100개 이미지)
- `dataset/Scratch/` (100개 이미지)

**총 802개 이미지**를 라벨링해야 합니다.

---

## 🗑️ 삭제 가능한 폴더/파일

### 1. YOLO 학습 결과 폴더 (재생성 예정)
- `yolo_dataset/` - 이전 학습용 데이터셋 (6.5MB)
- `runs/` - YOLO 학습 중 생성된 결과 (17MB)

### 2. 테스트 파일
- `test_qt.py` - Qt 테스트 파일 (더 이상 필요 없음)

### 3. 잘못 위치한 파일 (dataset/ 루트)
- `dataset/641447.txt` - 잘못 위치한 라벨 파일
- `dataset/classes.txt` - 불필요 (YOLO 데이터셋에 자동 생성됨)

---

## ✅ 유지해야 할 폴더/파일

- `dataset/` - 원본 이미지 (라벨링 대상)
- `weights/` - 학습된 모델 (유지)
- `venv/` - 가상 환경
- 모든 `.py` 파일들
- 설정 파일들

---

## 🚀 정리 스크립트


