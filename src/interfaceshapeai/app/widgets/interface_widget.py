from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from interfaceshapeai.app.widgets.structure_widget import StructureWidget
from interfaceshapeai.structure.chains import ChainSelectionError, select_chain_pair
from interfaceshapeai.structure.interface import detect_interface


class InterfaceWidget(QWidget):
    """Tab 2: detect and inspect interface residues between the selected chains."""

    def __init__(self, structure_widget: StructureWidget):
        super().__init__()
        self.structure_widget = structure_widget
        self.interface_residues: list = []
        self.chain_a = None
        self.chain_b = None

        self.cutoff_spin = QDoubleSpinBox()
        self.cutoff_spin.setRange(1.0, 20.0)
        self.cutoff_spin.setValue(5.0)
        self.cutoff_spin.setSuffix(" Å")

        self.detect_button = QPushButton("Detect Interface")
        self.detect_button.clicked.connect(self._run_detection)

        self.status_label = QLabel("Select chains in the Structure tab, then detect the interface.")
        self.status_label.setWordWrap(True)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Chain", "Residue #", "Residue", "Min distance (Å)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        form = QFormLayout()
        form.addRow("Distance cutoff", self.cutoff_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.detect_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table)

    def _run_detection(self) -> None:
        chains = self.structure_widget.selected_chains()
        if chains is None:
            self.status_label.setText("No structure loaded yet.")
            return
        chain_a_id, chain_b_id = chains
        try:
            self.chain_a, self.chain_b = select_chain_pair(
                self.structure_widget.structure, chain_a_id, chain_b_id
            )
        except ChainSelectionError as exc:
            self.status_label.setText(f"Error: {exc}")
            return

        self.interface_residues = detect_interface(
            self.chain_a, self.chain_b, distance_cutoff=self.cutoff_spin.value()
        )
        if not self.interface_residues:
            self.status_label.setText(
                f"No interface detected between chains {chain_a_id} and {chain_b_id}. "
                "Try increasing the distance cutoff or selecting another chain pair."
            )
            self.table.setRowCount(0)
            return

        self.status_label.setText(f"Detected {len(self.interface_residues)} interface residues.")
        self.table.setRowCount(len(self.interface_residues))
        for row, residue in enumerate(self.interface_residues):
            self.table.setItem(row, 0, QTableWidgetItem(residue.chain))
            self.table.setItem(row, 1, QTableWidgetItem(str(residue.residue_number)))
            self.table.setItem(row, 2, QTableWidgetItem(residue.residue_name))
            self.table.setItem(row, 3, QTableWidgetItem(f"{residue.min_distance:.2f}"))
