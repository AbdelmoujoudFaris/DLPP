import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_main_window_imports_and_constructs():
    from PySide6.QtWidgets import QApplication

    from interfaceshapeai.app.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle().startswith("InterfaceShapeAI")
    window.close()


def test_main_window_has_explainability_tab():
    from PySide6.QtWidgets import QApplication

    from interfaceshapeai.app.main_window import MainWindow
    from interfaceshapeai.app.widgets.explainability_widget import ExplainabilityWidget

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert isinstance(window.explainability_widget, ExplainabilityWidget)
    window.close()
