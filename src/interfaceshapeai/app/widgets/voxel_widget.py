import numpy as np
from PySide6.QtWidgets import (
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from interfaceshapeai.app.widgets.interface_widget import InterfaceWidget
from interfaceshapeai.geometry.burial_depth import geometric_burial_depth
from interfaceshapeai.geometry.voxelization import voxelize_interface
from interfaceshapeai.inference.preprocessing import extract_interface_atoms


class VoxelWidget(QWidget):
    """Tab 3: convert the detected interface into a 3D voxel tensor."""

    DEFAULT_CHANNELS = ["occupancy", "burial_depth", "hydrophobicity", "charge"]
    DEFAULT_VOXEL_SIZE = 1.0  # must match the voxel_size passed to voxelize_interface below

    def __init__(self, interface_widget: InterfaceWidget):
        super().__init__()
        self.interface_widget = interface_widget
        self.voxel_tensor = None
        # Retained (alongside voxel_tensor) so ExplainabilityWidget can map
        # voxel importance back to residues without recomputing extraction.
        self.coords = None
        self.residue_names: list[str] = []
        self.residue_ids: list[tuple[str, int]] = []
        self.burial_depth = None

        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(8, 128)
        self.grid_size_spin.setValue(32)

        self.voxelize_button = QPushButton("Voxelize Interface")
        self.voxelize_button.clicked.connect(self._run_voxelization)

        self.status_label = QLabel("Detect an interface first, then voxelize it.")
        self.status_label.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Grid size", self.grid_size_spin)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.voxelize_button)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _run_voxelization(self) -> None:
        residues = self.interface_widget.interface_residues
        chain_a, chain_b = self.interface_widget.chain_a, self.interface_widget.chain_b
        if not residues or chain_a is None:
            self.status_label.setText("No interface residues available. Run detection first.")
            return

        coords_a, names_a, ids_a = extract_interface_atoms(chain_a, residues)
        coords_b, names_b, ids_b = extract_interface_atoms(chain_b, residues)
        coords = np.concatenate([coords_a, coords_b], axis=0)
        residue_names = names_a + names_b
        residue_ids = ids_a + ids_b

        depth = geometric_burial_depth(coords)
        self.voxel_tensor = voxelize_interface(
            coords,
            residue_names,
            depth,
            grid_size=self.grid_size_spin.value(),
            channels=self.DEFAULT_CHANNELS,
        )
        self.coords = coords
        self.residue_names = residue_names
        self.residue_ids = residue_ids
        self.burial_depth = depth
        occupied = int((self.voxel_tensor[0] > 0).sum().item())
        self.status_label.setText(
            f"Voxel tensor shape: {tuple(self.voxel_tensor.shape)} "
            f"({occupied} occupied voxels, channels={self.DEFAULT_CHANNELS})"
        )
