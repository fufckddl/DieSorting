#!/bin/bash
# LabelImg 실행 스크립트

cd "$(dirname "$0")"
source venv/bin/activate

# Qt 플러그인 경로 설정 (macOS에서 필요)
export QT_PLUGIN_PATH=$(python -c "from PyQt5.QtCore import QLibraryInfo; import PyQt5; from pathlib import Path; pyqt5_path = Path(PyQt5.__file__).parent; qt5_plugins = pyqt5_path / 'Qt5' / 'plugins'; print(qt5_plugins) if qt5_plugins.exists() else print(QLibraryInfo.location(QLibraryInfo.PluginsPath))" 2>/dev/null)

echo "LabelImg를 실행합니다..."
echo ""
echo "사용 방법:"
echo "1. Open Dir 버튼으로 dataset/ 폴더 선택"
echo "2. 좌측 상단 'PascalVOC' 버튼 클릭 → 'YOLO' 선택"
echo "3. View → Auto Save mode 체크"
echo "4. W 키로 박스 그리기 시작"
echo "5. 각 이미지의 불량 부분을 박스로 표시"
echo "6. 클래스명: defect 입력"
echo "7. D 키로 다음 이미지 진행"
echo ""
echo "라벨링 진행 상황 확인:"
echo "python utils/check_labeling_status.py"
echo ""

labelImg

