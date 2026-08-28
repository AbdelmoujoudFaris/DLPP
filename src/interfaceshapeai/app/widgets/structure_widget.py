from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from interfaceshapeai.app.widgets.upload_widget import UploadWidget
from interfaceshapeai.structure.chains import list_chain_ids
from interfaceshapeai.structure.parser import StructureParsingError, load_structure


class StructureWidget(QWidget):
    """Tab 1: upload a structure and select the two interacting chains."""

    def __init__(self):
        super().__init__()
        self.structure = None

        self.upload = UploadWidget()
        self.upload.file_selected.connect(self._on_file_selected)

        self.chain_a_combo = QComboBox()
        self.chain_b_combo = QComboBox()
        self.status_label = QLabel("Upload a PDB or mmCIF file to begin.")
        self.status_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Chain A", self.chain_a_combo)
        form.addRow("Chain B", self.chain_b_combo)

        layout = QVBoxLayout(self)
        layout.addWidget(self.upload)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _on_file_selected(self, path: str) -> None:
        try:
            self.structure = load_structure(path)
        except StructureParsingError as exc:
            self.structure = None
            self.status_label.setText(f"Error: {exc}")
            return

        chain_ids = list_chain_ids(self.structure)
        self.chain_a_combo.clear()
        self.chain_b_combo.clear()
        self.chain_a_combo.addItems(chain_ids)
        self.chain_b_combo.addItems(chain_ids)
        if len(chain_ids) > 1:
            self.chain_b_combo.setCurrentIndex(1)
        self.status_label.setText(f"Loaded {len(chain_ids)} chain(s): {', '.join(chain_ids)}")

    def selected_chains(self) -> tuple[str, str] | None:
        if self.structure is None:
            return None
        return self.chain_a_combo.currentText(), self.chain_b_combo.currentText()
