"""
반도체 불량 패턴 분류 시스템 - 메인 엔트리 포인트
"""

import sys
import os
from pathlib import Path

# Qt 플러그인 경로 설정 (macOS에서 필요) - QApplication 생성 전에 설정
if sys.platform == 'darwin':
    try:
        from PyQt5.QtCore import QLibraryInfo
        qt_plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        if qt_plugin_path:
            os.environ['QT_PLUGIN_PATH'] = str(qt_plugin_path)
            # 추가 경로 설정
            import PyQt5
            pyqt5_path = Path(PyQt5.__file__).parent
            qt5_plugins = pyqt5_path / 'Qt5' / 'plugins'
            if qt5_plugins.exists():
                os.environ['QT_PLUGIN_PATH'] = str(qt5_plugins)
    except Exception:
        pass

# High DPI 스케일링 설정 (QApplication 생성 전에 설정해야 함)
from PyQt5.QtCore import Qt
os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'

from PyQt5.QtWidgets import QApplication

from gui.main_window import MainWindow


def main():
    """애플리케이션 실행"""
    # High DPI 스케일링 속성은 QApplication 생성 전에 환경 변수로 설정됨
    app = QApplication(sys.argv)
    
    # 추가 속성 설정 (QApplication 생성 후에도 일부 속성은 설정 가능)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # 메인 윈도우 생성 및 표시
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

