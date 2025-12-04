#!/bin/bash
# 프로젝트 정리 스크립트

cd "$(dirname "$0")"

echo "프로젝트 정리 시작..."
echo ""

# 1. YOLO 학습 결과 폴더 삭제 (재생성 예정)
if [ -d "yolo_dataset" ]; then
    echo "삭제: yolo_dataset/ (재생성 예정)"
    rm -rf yolo_dataset
fi

if [ -d "runs" ]; then
    echo "삭제: runs/ (학습 결과)"
    rm -rf runs
fi

# 2. 테스트 파일 삭제
if [ -f "test_qt.py" ]; then
    echo "삭제: test_qt.py (테스트 파일)"
    rm -f test_qt.py
fi

# 3. dataset 루트의 잘못 위치한 파일들 삭제
if [ -f "dataset/641447.txt" ]; then
    echo "삭제: dataset/641447.txt (잘못 위치한 파일)"
    rm -f dataset/641447.txt
fi

if [ -f "dataset/classes.txt" ]; then
    echo "삭제: dataset/classes.txt (불필요)"
    rm -f dataset/classes.txt
fi

# 4. __pycache__ 정리 (선택적)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "정리: __pycache__ 파일들"

echo ""
echo "✅ 정리 완료!"
echo ""
echo "라벨링 대상 폴더:"
echo "  - dataset/Center/"
echo "  - dataset/Donut/"
echo "  - dataset/Edge Local/"
echo "  - dataset/Edge Ring/"
echo "  - dataset/Local/"
echo "  - dataset/near full/"
echo "  - dataset/none/"
echo "  - dataset/Scratch/"


