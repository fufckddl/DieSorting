# 빠른 시작 가이드

## ✅ 데이터셋 준비 완료!

현재 `dataset/` 폴더의 이미지로 YOLO 학습용 데이터셋이 자동 생성되었습니다.

---

## 🚀 바로 학습 시작

```bash
# YOLO 학습 실행
python train_yolo.py --data yolo_dataset/dataset.yaml --epochs 100
```

학습이 완료되면:
- `weights/yolo_detector.pt` 생성
- GUI에서 박스 표시됨 ✅

---

## 📊 현재 상태

- ✅ 데이터셋 준비: 802개 이미지 → yolo_dataset/ 구조 생성
- ✅ Train: 641개, Val: 161개
- ⏳ YOLO 학습 진행 중...

---

## ⚠️ 참고사항

현재는 **이미지 전체를 하나의 불량으로 간주**하여 라벨을 생성했습니다.

**정확도를 높이려면:**
- LabelImg로 실제 불량 위치를 정확히 라벨링
- 또는 자동 검출 도구 사용:
  ```bash
  python utils/smart_labeling_helper.py --dataset-dir dataset/
  ```

**하지만 일단은:**
- 현재 상태로 학습 진행
- 학습 후 박스가 표시되는지 확인
- 필요하면 나중에 정확한 라벨링 추가

---

## 다음 단계

학습 완료 후:
```bash
python main.py
```

GUI에서 이미지를 선택하고 분석하면 박스가 표시됩니다!


