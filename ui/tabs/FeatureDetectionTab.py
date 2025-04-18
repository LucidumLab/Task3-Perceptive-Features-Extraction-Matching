from PyQt5.QtWidgets import (
     QPushButton, QLabel,  QFrame,
    QVBoxLayout, QWidget,  QSpinBox, QDoubleSpinBox, QHBoxLayout, QComboBox, QGridLayout, QLineEdit, QCheckBox, QTabWidget, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)

from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QImage, QPixmap


class Feature_DetectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Main layout for the tab
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        # Edge Detector Selection
        edge_detector_group = QFrame()
        edge_detector_group.setObjectName("edge_detector_group")
        edge_detector_layout = QVBoxLayout(edge_detector_group)

        # === Edge Type Frame ===
        self.edgeTypeFrame = QFrame()
        edge_type_layout = QHBoxLayout(self.edgeTypeFrame)
        edge_type_layout.addWidget(QLabel("Edge Detector"))
        self.edgeType = QComboBox()
        self.edgeType.addItems(["canny", "sobel", "prewitt", "roberts"])
        self.edgeType.currentTextChanged.connect(self.update_edge_params_visibility)
        edge_type_layout.addWidget(self.edgeType)
        edge_detector_layout.addWidget(self.edgeTypeFrame)

        # === Kernel Size Frame ===
        self.kernelSizeFrame = QFrame()
        kernel_size_layout = QHBoxLayout(self.kernelSizeFrame)
        kernel_size_layout.addWidget(QLabel("Kernel Size"))
        self.kernelSize = QSpinBox()
        self.kernelSize.setRange(1, 15)
        self.kernelSize.setValue(3)
        kernel_size_layout.addWidget(self.kernelSize)
        edge_detector_layout.addWidget(self.kernelSizeFrame)

        # === Low Threshold Frame ===
        self.lowThresholdFrame = QFrame()
        low_threshold_layout = QHBoxLayout(self.lowThresholdFrame)
        low_threshold_layout.addWidget(QLabel("Low Threshold"))
        self.lowThreshold = QSpinBox()
        self.lowThreshold.setRange(0, 255)
        self.lowThreshold.setValue(50)
        low_threshold_layout.addWidget(self.lowThreshold)
        edge_detector_layout.addWidget(self.lowThresholdFrame)

        # === High Threshold Frame ===
        self.highThresholdFrame = QFrame()
        high_threshold_layout = QHBoxLayout(self.highThresholdFrame)
        high_threshold_layout.addWidget(QLabel("High Threshold"))
        self.highThreshold = QSpinBox()
        self.highThreshold.setRange(0, 255)
        self.highThreshold.setValue(150)
        high_threshold_layout.addWidget(self.highThreshold)
        edge_detector_layout.addWidget(self.highThresholdFrame)

        main_layout.addWidget(edge_detector_group)

        # Feature Detector Selection
        feature_detector_group = QFrame()
        feature_detector_group.setObjectName("feature_detector_group")
        feature_detector_layout = QVBoxLayout(feature_detector_group)

        # === Feature Type Frame ===
        self.featureTypeFrame = QFrame()
        feature_type_layout = QHBoxLayout(self.featureTypeFrame)
        feature_type_layout.addWidget(QLabel("Feature Detector"))
        self.featureType = QComboBox()
        self.featureType.addItems(["MOPs", "SIFT", "Harris"])
        self.featureType.currentTextChanged.connect(self.update_feature_params_visibility)
        feature_type_layout.addWidget(self.featureType)
        feature_detector_layout.addWidget(self.featureTypeFrame)

        # === Harris Controls Frame ===
        self.harrisFrame = QFrame()
        harris_layout = QHBoxLayout(self.harrisFrame)
        self.harrisKLabel = QLabel("Harris K")
        self.harrisK = QDoubleSpinBox()
        self.harrisK.setRange(0.01, 0.1)
        self.harrisK.setSingleStep(0.01)
        self.harrisK.setValue(0.04)
        harris_layout.addWidget(self.harrisKLabel)
        harris_layout.addWidget(self.harrisK)
        feature_detector_layout.addWidget(self.harrisFrame)

        # === SIFT Controls Frame ===
        self.siftFrame = QFrame()
        sift_layout = QHBoxLayout(self.siftFrame)
        self.siftFeaturesLabel = QLabel("SIFT Features")
        self.siftFeatures = QSpinBox()
        self.siftFeatures.setRange(1, 5000)
        self.siftFeatures.setValue(500)
        sift_layout.addWidget(self.siftFeaturesLabel)
        sift_layout.addWidget(self.siftFeatures)
        feature_detector_layout.addWidget(self.siftFrame)

        main_layout.addWidget(feature_detector_group)

        # Update visibility initially
        self.update_edge_params_visibility()
        self.update_feature_params_visibility()

    def update_edge_params_visibility(self):
        edge_type = self.edgeType.currentText()
        is_canny = edge_type == "canny"

        # Show/Hide Low and High Threshold frames together
        self.lowThresholdFrame.setVisible(is_canny)
        self.highThresholdFrame.setVisible(is_canny)

    def update_feature_params_visibility(self):
        feature_type = self.featureType.currentText()

        self.harrisFrame.setVisible(feature_type == "Harris")
        self.siftFrame.setVisible(feature_type == "SIFT")

class Feature_Detection_Frame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)

        grid_layout = QGridLayout()
        grid_layout.setAlignment(Qt.AlignCenter)
        main_layout.addLayout(grid_layout)

        self.image_1 = QLabel("Image 1")
        self.image_1.setObjectName("original_label")
        self.image_1.setFixedSize(600, 425)
        self.image_1.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.image_1, 0, 0)

        self.image_2 = QLabel("Image 2")
        self.image_2.setObjectName("template_label")
        self.image_2.setFixedSize(600, 425)
        self.image_2.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.image_2, 0, 1)

        self.roi_image = QLabel("ROI Image")
        self.roi_image.setObjectName("roi_label")
        self.roi_image.setFixedSize(600, 425)
        self.roi_image.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.roi_image, 1, 0)

        self.confusion_image = QLabel("Confusion Matrix")
        self.confusion_image.setObjectName("confusion_label")
        self.confusion_image.setFixedSize(600, 425)
        self.confusion_image.setAlignment(Qt.AlignCenter)
        grid_layout.addWidget(self.confusion_image, 1, 1)


    def display_image(self, image, label_number):
        """
        Displays the given image in the specified label.

        Args:
            image (numpy.ndarray): The image to display.
            label_number (int): The label number (1 for Original, 2 for Template, 3 for ROI, 4 for Confusion Matrix).
        """
        if image is None:
            raise ValueError("The image cannot be None.")

        if len(image.shape) == 3: 
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        elif len(image.shape) == 2:  
            h, w = image.shape
            qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            raise ValueError("Unsupported image format. The image must be 2D (grayscale) or 3D (color).")

        pixmap = QPixmap.fromImage(qimg)

        if label_number == 1:
            self.image_1.setPixmap(pixmap.scaled(self.image_1.width(), self.image_1.height(), Qt.KeepAspectRatio))
        elif label_number == 2:
            self.image_2.setPixmap(pixmap.scaled(self.image_2.width(), self.image_2.height(), Qt.KeepAspectRatio))
        elif label_number == 3:
            self.roi_image.setPixmap(pixmap.scaled(self.roi_image.width(), self.roi_image.height(), Qt.KeepAspectRatio))
        elif label_number == 4:
            self.confusion_image.setPixmap(pixmap.scaled(self.confusion_image.width(), self.confusion_image.height(), Qt.KeepAspectRatio))
        else:
            raise ValueError("Invalid label number. Must be 1 (Original), 2 (Template), 3 (ROI), or 4 (Confusion Matrix).")
