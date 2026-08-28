from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from interfaceshapeai.app.widgets.voxel_widget import VoxelWidget
from interfaceshapeai.explainability.gradcam3d import grad_cam_3d
from interfaceshapeai.explainability.integrated_gradients import integrated_gradients
from interfaceshapeai.explainability.residue_mapping import map_voxel_importance_to_residues
from interfaceshapeai.explainability.saliency import compute_saliency
from interfaceshapeai.inference.predictor import Predictor
from interfaceshapeai.utils.config import Config

_METHODS = {
    "Grad-CAM 3D": "gradcam",
    "Saliency": "saliency",
    "Integrated Gradients": "integrated_gradients",
}


class ExplainabilityWidget(QWidget):
    """Tab 5: explain the (demo-mode, untrained) model's function-class
    prediction via real gradient-based attribution, mapped back onto
    interface residues (section 16). Only supports architecture "cnn3d";
    multi-resolution explainability is on the roadmap.
    """

    def __init__(self, voxel_widget: VoxelWidget, config: Config):
        super().__init__()
        self.voxel_widget = voxel_widget
        self.config = config

        self.method_combo = QComboBox()
        self.method_combo.addItems(list(_METHODS.keys()))

        self.target_combo = QComboBox()
        self.target_combo.addItem("(top prediction)")
        self.target_combo.addItems(config.model.function_classes)

        self.generate_button = QPushButton("Generate Explanation")
        self.generate_button.clicked.connect(self._run_explanation)

        self.status_label = QLabel("Voxelize an interface first, then generate an explanation.")
        self.status_label.setWordWrap(True)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Rank", "Chain", "Residue", "Score", "Depth", "Feature"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        form = QFormLayout()
        form.addRow("Method", self.method_combo)
        form.addRow("Target (function class)", self.target_combo)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.table)

    def _run_explanation(self) -> None:
        voxel = self.voxel_widget.voxel_tensor
        if voxel is None or self.voxel_widget.coords is None:
            self.status_label.setText("No voxel tensor available. Voxelize an interface first.")
            return
        if self.config.model.architecture != "cnn3d":
            self.status_label.setText(
                f"Explainability is implemented for architecture 'cnn3d' only "
                f"(current: '{self.config.model.architecture}')."
            )
            return

        target_text = self.target_combo.currentText()
        target_class = (
            None if target_text == "(top prediction)" else self.config.model.function_classes.index(target_text)
        )

        predictor = Predictor(self.config, in_channels=voxel.shape[0])
        method = _METHODS[self.method_combo.currentText()]
        if method == "saliency":
            importance = compute_saliency(predictor.model, voxel, target="function", target_class=target_class)
        elif method == "gradcam":
            importance = grad_cam_3d(predictor.model, voxel, target="function", target_class=target_class)
        else:
            importance = integrated_gradients(predictor.model, voxel, target="function", target_class=target_class)

        rows = map_voxel_importance_to_residues(
            importance,
            self.voxel_widget.coords,
            self.voxel_widget.residue_ids,
            self.voxel_widget.residue_names,
            self.voxel_widget.burial_depth,
            voxel_size=self.voxel_widget.DEFAULT_VOXEL_SIZE,
            grid_size=voxel.shape[-1],
        )

        prefix = "MODEL STATUS: demo mode (no trained weights) - not scientifically meaningful. " if predictor.demo_mode else ""
        self.status_label.setText(f"{prefix}Showing top {min(len(rows), 20)} residues by {method} importance.")
        self.table.setRowCount(min(len(rows), 20))
        for row_index, row in enumerate(rows[:20]):
            self.table.setItem(row_index, 0, QTableWidgetItem(str(row["rank"])))
            self.table.setItem(row_index, 1, QTableWidgetItem(row["chain"]))
            self.table.setItem(row_index, 2, QTableWidgetItem(f"{row['residue_name']}{row['residue_number']}"))
            self.table.setItem(row_index, 3, QTableWidgetItem(f"{row['score']:.3f}"))
            self.table.setItem(row_index, 4, QTableWidgetItem(f"{row['depth']:.3f}"))
            self.table.setItem(row_index, 5, QTableWidgetItem(row["feature"]))
