from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from interfaceshapeai.app.widgets.voxel_widget import VoxelWidget
from interfaceshapeai.inference.predictor import Predictor
from interfaceshapeai.utils.config import Config


class PredictionWidget(QWidget):
    """Tab 4: run the (demo-mode, untrained) model on the voxelized interface."""

    def __init__(self, voxel_widget: VoxelWidget, config: Config):
        super().__init__()
        self.voxel_widget = voxel_widget
        self.config = config

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self._run_prediction)

        self.warning_label = QLabel(
            "MODEL STATUS: architecture available, no trained weights loaded (demo mode).\n"
            "Predictions are for pipeline testing only and are NOT scientifically meaningful."
        )
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #b45309; font-weight: bold;")

        self.result_label = QLabel("Voxelize an interface first, then run prediction.")
        self.result_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.warning_label)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.result_label)
        layout.addStretch(1)

    def _run_prediction(self) -> None:
        voxel = self.voxel_widget.voxel_tensor
        if voxel is None:
            self.result_label.setText("No voxel tensor available. Voxelize an interface first.")
            return

        predictor = Predictor(self.config, in_channels=voxel.shape[0])
        result = predictor.predict(voxel)

        lines = ["Secondary structure prediction:"]
        for label, prob in result["structure_prediction"].items():
            lines.append(f"  {label}: {prob:.2f}")
        lines.append("Functional class prediction:")
        for label, prob in result["function_prediction"].items():
            lines.append(f"  {label}: {prob:.2f}")
        self.result_label.setText("\n".join(lines))
