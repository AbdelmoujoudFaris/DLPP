from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget


class UploadWidget(QWidget):
    """Reusable file-picker: a path field plus a Browse button."""

    file_selected = Signal(str)

    def __init__(self, dialog_filter: str = "Structure files (*.pdb *.ent *.cif *.mmcif)"):
        super().__init__()
        self._dialog_filter = dialog_filter

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("No file selected")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse)

        layout = QHBoxLayout(self)
        layout.addWidget(self.path_edit)
        layout.addWidget(self.browse_button)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select structure file", "", self._dialog_filter)
        if path:
            self.path_edit.setText(path)
            self.file_selected.emit(path)

    def current_path(self) -> str:
        return self.path_edit.text().strip()
