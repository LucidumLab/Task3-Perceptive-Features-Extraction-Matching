from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QComboBox, QSpinBox, 
    QDoubleSpinBox, QPushButton, QLabel
)
from PyQt5.QtCore import Qt

class NoiseFilterTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        noise_and_filter_layout = QVBoxLayout(self)

        # Noise Frame
        noise_frame = QFrame()
        noise_frame.setObjectName("noise_frame")
        noise_layout = QVBoxLayout(noise_frame)
        noise_layout.setAlignment(Qt.AlignTop)

        # Noise UI Components
        self.noiseType = QComboBox()
        self.noiseType.addItems(["uniform", "gaussian", "salt_pepper"])
        self.noiseType.currentTextChanged.connect(self.update_noise_params_visibility)

        self.noiseIntensity = QSpinBox()
        self.noiseIntensity.setRange(1, 100)
        self.noiseIntensity.setValue(50)

        self.gaussianMean = QDoubleSpinBox()
        self.gaussianMean.setRange(-50, 50)
        self.gaussianMean.setValue(0)

        self.gaussianStd = QDoubleSpinBox()
        self.gaussianStd.setRange(1, 100)
        self.gaussianStd.setValue(25)

        self.saltProb = QDoubleSpinBox()
        self.saltProb.setRange(0.0, 1.0)
        self.saltProb.setSingleStep(0.01)
        self.saltProb.setValue(0.02)

        self.pepperProb = QDoubleSpinBox()
        self.pepperProb.setRange(0.0, 1.0)
        self.pepperProb.setSingleStep(0.01)
        self.pepperProb.setValue(0.02)

        self.btn_noise = QPushButton("Add Noise")
        self.btn_noise.clicked.connect(parent.apply_noise)

        self.noiseParamLayout = QVBoxLayout()
        noise_type_layout = QHBoxLayout()
        noise_type_layout.addWidget(QLabel("Noise Type"))
        noise_type_layout.addWidget(self.noiseType)
        self.noiseParamLayout.addLayout(noise_type_layout)

        self.intensity_label = QLabel("Intensity")
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(self.intensity_label)
        intensity_layout.addWidget(self.noiseIntensity)
        self.noiseParamLayout.addLayout(intensity_layout)

        self.mean_label = QLabel("Mean")
        mean_layout = QHBoxLayout()
        mean_layout.addWidget(self.mean_label)
        mean_layout.addWidget(self.gaussianMean)
        self.noiseParamLayout.addLayout(mean_layout)

        self.std_label = QLabel("Std Dev")
        std_layout = QHBoxLayout()
        std_layout.addWidget(self.std_label)
        std_layout.addWidget(self.gaussianStd)
        self.noiseParamLayout.addLayout(std_layout)

        self.salt_label = QLabel("Salt Prob")
        salt_layout = QHBoxLayout()
        salt_layout.addWidget(self.salt_label)
        salt_layout.addWidget(self.saltProb)
        self.noiseParamLayout.addLayout(salt_layout)

        self.pepper_label = QLabel("Pepper Prob")
        pepper_layout = QHBoxLayout()
        pepper_layout.addWidget(self.pepper_label)
        pepper_layout.addWidget(self.pepperProb)
        self.noiseParamLayout.addLayout(pepper_layout)

        self.noiseParamLayout.addWidget(self.btn_noise)

        noise_layout.addLayout(self.noiseParamLayout)
        noise_and_filter_layout.addWidget(noise_frame)

        # Filter Frame
        filter_frame = QFrame()
        filter_frame.setObjectName("filter_frame")
        filter_layout = QVBoxLayout(filter_frame)
        filter_layout.setAlignment(Qt.AlignTop)

        # Filter UI Components
        self.filterType = QComboBox()
        self.filterType.addItems(["average", "gaussian", "median"])

        self.kernelSize = QSpinBox()
        self.kernelSize.setRange(1, 15)
        self.kernelSize.setValue(3)

        self.sigmaValue = QDoubleSpinBox()
        self.sigmaValue.setRange(0.1, 10.0)
        self.sigmaValue.setValue(1.0)

        self.btn_filter = QPushButton("Apply Filter")
        self.btn_filter.clicked.connect(parent.apply_filter)

        self.filterParamLayout = QVBoxLayout()
        filter_type_layout = QHBoxLayout()
        filter_type_layout.addWidget(QLabel("Filter Type"))
        filter_type_layout.addWidget(self.filterType)
        self.filterParamLayout.addLayout(filter_type_layout)

        kernel_size_layout = QHBoxLayout()
        kernel_size_layout.addWidget(QLabel("Kernel Size"))
        kernel_size_layout.addWidget(self.kernelSize)
        self.filterParamLayout.addLayout(kernel_size_layout)

        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma"))
        sigma_layout.addWidget(self.sigmaValue)
        self.filterParamLayout.addLayout(sigma_layout)

        self.filterParamLayout.addWidget(self.btn_filter)

        filter_layout.addLayout(self.filterParamLayout)
        noise_and_filter_layout.addWidget(filter_frame)

        self.update_noise_params_visibility()

    def update_noise_params_visibility(self):
        noise_type = self.noiseType.currentText()
        self.noiseIntensity.setVisible(noise_type == "uniform")
        self.intensity_label.setVisible(noise_type == "uniform")
        self.gaussianMean.setVisible(noise_type == "gaussian")
        self.mean_label.setVisible(noise_type == "gaussian")
        self.gaussianStd.setVisible(noise_type == "gaussian")
        self.std_label.setVisible(noise_type == "gaussian")
        self.saltProb.setVisible(noise_type == "salt_pepper")
        self.salt_label.setVisible(noise_type == "salt_pepper")
        self.pepperProb.setVisible(noise_type == "salt_pepper")
        self.pepper_label.setVisible(noise_type == "salt_pepper")
