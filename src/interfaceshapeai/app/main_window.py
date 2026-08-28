from PySide6.QtWidgets import QLabel, QMainWindow, QStatusBar, QTabWidget

from interfaceshapeai.app.widgets.explainability_widget import ExplainabilityWidget
from interfaceshapeai.app.widgets.interface_widget import InterfaceWidget
from interfaceshapeai.app.widgets.prediction_widget import PredictionWidget
from interfaceshapeai.app.widgets.structure_widget import StructureWidget
from interfaceshapeai.app.widgets.voxel_widget import VoxelWidget
from interfaceshapeai.utils.config import Config, load_config
from interfaceshapeai.utils.device import resolve_device


class MainWindow(QMainWindow):
    def __init__(self, config: Config | None = None):
        super().__init__()
        self.config = config or load_config()

        self.setWindowTitle("InterfaceShapeAI - Protein-Protein Interface Deep Learning")
        self.resize(1000, 700)

        self.structure_widget = StructureWidget()
        self.interface_widget = InterfaceWidget(self.structure_widget)
        self.voxel_widget = VoxelWidget(self.interface_widget)
        self.prediction_widget = PredictionWidget(self.voxel_widget, self.config)
        self.explainability_widget = ExplainabilityWidget(self.voxel_widget, self.config)

        tabs = QTabWidget()
        tabs.addTab(self.structure_widget, "1. Structure")
        tabs.addTab(self.interface_widget, "2. Interface")
        tabs.addTab(self.voxel_widget, "3. Voxelize")
        tabs.addTab(self.prediction_widget, "4. Predict")
        tabs.addTab(self.explainability_widget, "5. Explain")
        self.setCentralWidget(tabs)

        device = resolve_device(self.config.device.type)
        status_bar = QStatusBar()
        status_bar.addWidget(QLabel(f"Status: Ready    Device: {device}"))
        self.setStatusBar(status_bar)
