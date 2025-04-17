from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QComboBox, QSpinBox, 
    QDoubleSpinBox, QPushButton, QLabel
)
from PyQt5.QtCore import Qt

class ThresholdingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        thresholding_frame = QFrame()
        thresholding_frame.setObjectName("thresholding_frame")
        thresholding_layout = QVBoxLayout(thresholding_frame)
        thresholding_layout.setAlignment(Qt.AlignTop)

        self.thresholdType = QComboBox()
        self.thresholdType.addItems(["global", "local"])
        self.thresholdType.currentTextChanged.connect(self.update_threshold_params_visibility)

        self.globalThreshold = QSpinBox()
        self.globalThreshold.setRange(0, 255)
        self.globalThreshold.setValue(128)

        self.kernelSizeThreshold = QSpinBox()
        self.kernelSizeThreshold.setRange(1, 15)
        self.kernelSizeThreshold.setValue(4)

        self.kValue = QDoubleSpinBox()
        self.kValue.setRange(0.0, 5.0)
        self.kValue.setSingleStep(0.1)
        self.kValue.setValue(2.0)

        self.btn_threshold = QPushButton("Apply Thresholding")
        self.btn_threshold.clicked.connect(parent.apply_thresholding)

        self.thresholdLayout = QVBoxLayout()
        threshold_type_layout = QHBoxLayout()
        threshold_type_layout.addWidget(QLabel("Threshold Type"))
        threshold_type_layout.addWidget(self.thresholdType)
        self.thresholdLayout.addLayout(threshold_type_layout)

        self.global_threshold_label = QLabel("Global Threshold")
        global_threshold_layout = QHBoxLayout()
        global_threshold_layout.addWidget(self.global_threshold_label)
        global_threshold_layout.addWidget(self.globalThreshold)
        self.thresholdLayout.addLayout(global_threshold_layout)

        self.kernel_size_label = QLabel("Kernel Size")
        kernel_size_threshold_layout = QHBoxLayout()
        kernel_size_threshold_layout.addWidget(self.kernel_size_label)
        kernel_size_threshold_layout.addWidget(self.kernelSizeThreshold)
        self.thresholdLayout.addLayout(kernel_size_threshold_layout)

        self.k_value_label = QLabel("K Value")
        k_value_layout = QHBoxLayout()
        k_value_layout.addWidget(self.k_value_label)
        k_value_layout.addWidget(self.kValue)
        self.thresholdLayout.addLayout(k_value_layout)

        self.thresholdLayout.addWidget(self.btn_threshold)

        thresholding_layout.addLayout(self.thresholdLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(thresholding_frame)

        self.update_threshold_params_visibility()

    def update_threshold_params_visibility(self):
        threshold_type = self.thresholdType.currentText()
        is_global = threshold_type == "global"
        is_local = threshold_type == "local"

        self.globalThreshold.setVisible(is_global)
        self.global_threshold_label.setVisible(is_global)
        self.kernelSizeThreshold.setVisible(is_local)
        self.kernel_size_label.setVisible(is_local)
        self.kValue.setVisible(is_local)
        self.k_value_label.setVisible(is_local)
