from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QComboBox, QSpinBox, 
    QDoubleSpinBox, QPushButton, QLabel
)
from PyQt5.QtCore import Qt

class EdgeDetectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        edge_detection_frame = QFrame()
        edge_detection_frame.setObjectName("edge_detection_frame")
        edge_detection_layout = QVBoxLayout(edge_detection_frame)
        edge_detection_layout.setAlignment(Qt.AlignTop)

        self.edgeType = QComboBox()
        self.edgeType.addItems(["sobel", "canny", "prewitt", "roberts"])
        self.edgeType.currentTextChanged.connect(self.update_edge_params_visibility)

        self.sobelKernelSize = QSpinBox()
        self.sobelKernelSize.setRange(1, 15)
        self.sobelKernelSize.setValue(3)

        self.sobelSigma = QDoubleSpinBox()
        self.sobelSigma.setRange(0.1, 10.0)
        self.sobelSigma.setValue(1.0)

        self.cannyLowThreshold = QSpinBox()
        self.cannyLowThreshold.setRange(0, 255)
        self.cannyLowThreshold.setValue(50)

        self.cannyHighThreshold = QSpinBox()
        self.cannyHighThreshold.setRange(0, 255)
        self.cannyHighThreshold.setValue(150)

        self.cannyMaxEdgeVal = QSpinBox()
        self.cannyMaxEdgeVal.setRange(0, 255)
        self.cannyMaxEdgeVal.setValue(255)

        self.cannyMinEdgeVal = QSpinBox()
        self.cannyMinEdgeVal.setRange(0, 255)
        self.cannyMinEdgeVal.setValue(0)

        self.prewittThreshold = QSpinBox()
        self.prewittThreshold.setRange(0, 255)
        self.prewittThreshold.setValue(50)

        self.prewittValue = QSpinBox()
        self.prewittValue.setRange(0, 255)
        self.prewittValue.setValue(255)

        self.btn_edge_detection = QPushButton("Detect Edges")
        self.btn_edge_detection.clicked.connect(parent.detect_edges)

        self.edgeParamLayout = QVBoxLayout()
        edge_type_layout = QHBoxLayout()
        edge_type_layout.addWidget(QLabel("Edge Type"))
        edge_type_layout.addWidget(self.edgeType)
        self.edgeParamLayout.addLayout(edge_type_layout)

        self.sobel_kernel_label = QLabel("Kernel Size")
        sobel_kernel_layout = QHBoxLayout()
        sobel_kernel_layout.addWidget(self.sobel_kernel_label)
        sobel_kernel_layout.addWidget(self.sobelKernelSize)
        self.edgeParamLayout.addLayout(sobel_kernel_layout)

        self.sobel_sigma_label = QLabel("Sigma")
        sobel_sigma_layout = QHBoxLayout()
        sobel_sigma_layout.addWidget(self.sobel_sigma_label)
        sobel_sigma_layout.addWidget(self.sobelSigma)
        self.edgeParamLayout.addLayout(sobel_sigma_layout)

        self.canny_low_label = QLabel("Low Threshold")
        canny_low_layout = QHBoxLayout()
        canny_low_layout.addWidget(self.canny_low_label)
        canny_low_layout.addWidget(self.cannyLowThreshold)
        self.edgeParamLayout.addLayout(canny_low_layout)

        self.canny_high_label = QLabel("High Threshold")
        canny_high_layout = QHBoxLayout()
        canny_high_layout.addWidget(self.canny_high_label)
        canny_high_layout.addWidget(self.cannyHighThreshold)
        self.edgeParamLayout.addLayout(canny_high_layout)

        self.canny_max_label = QLabel("Max Edge")
        canny_max_layout = QHBoxLayout()
        canny_max_layout.addWidget(self.canny_max_label)
        canny_max_layout.addWidget(self.cannyMaxEdgeVal)
        self.edgeParamLayout.addLayout(canny_max_layout)

        self.canny_min_label = QLabel("Min Edge")
        canny_min_layout = QHBoxLayout()
        canny_min_layout.addWidget(self.canny_min_label)
        canny_min_layout.addWidget(self.cannyMinEdgeVal)
        self.edgeParamLayout.addLayout(canny_min_layout)

        self.prewitt_threshold_label = QLabel("Threshold")
        prewitt_threshold_layout = QHBoxLayout()
        prewitt_threshold_layout.addWidget(self.prewitt_threshold_label)
        prewitt_threshold_layout.addWidget(self.prewittThreshold)
        self.edgeParamLayout.addLayout(prewitt_threshold_layout)

        self.prewitt_value_label = QLabel("Value")
        prewitt_value_layout = QHBoxLayout()
        prewitt_value_layout.addWidget(self.prewitt_value_label)
        prewitt_value_layout.addWidget(self.prewittValue)
        self.edgeParamLayout.addLayout(prewitt_value_layout)

        self.edgeParamLayout.addWidget(self.btn_edge_detection)

        edge_detection_layout.addLayout(self.edgeParamLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(edge_detection_frame)

        self.update_edge_params_visibility()

    def update_edge_params_visibility(self):
        edge_type = self.edgeType.currentText()
        is_sobel = edge_type == "sobel"
        is_canny = edge_type == "canny"
        is_prewitt = edge_type == "prewitt"
        is_roberts = edge_type == "roberts"

        self.sobelKernelSize.setVisible(is_sobel)
        self.sobel_kernel_label.setVisible(is_sobel)
        self.sobelSigma.setVisible(is_sobel)
        self.sobel_sigma_label.setVisible(is_sobel)

        self.cannyLowThreshold.setVisible(is_canny)
        self.canny_low_label.setVisible(is_canny)
        self.cannyHighThreshold.setVisible(is_canny)
        self.canny_high_label.setVisible(is_canny)
        self.cannyMaxEdgeVal.setVisible(is_canny)
        self.canny_max_label.setVisible(is_canny)
        self.cannyMinEdgeVal.setVisible(is_canny)
        self.canny_min_label.setVisible(is_canny)

        self.prewittThreshold.setVisible(is_prewitt)
        self.prewitt_threshold_label.setVisible(is_prewitt)
        self.prewittValue.setVisible(is_prewitt)
        self.prewitt_value_label.setVisible(is_prewitt)
