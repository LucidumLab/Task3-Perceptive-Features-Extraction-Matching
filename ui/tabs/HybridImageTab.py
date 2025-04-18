from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QComboBox, QSpinBox, 
    QPushButton, QLabel, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2

class HybridImageTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.parent = parent
        self.image1 = None
        self.image2 = None

        hybrid_image_frame = QFrame()
        hybrid_image_frame.setObjectName("hybrid_image_frame")
        hybrid_image_layout = QVBoxLayout(hybrid_image_frame)
        hybrid_image_layout.setAlignment(Qt.AlignTop)

        self.image1_label = QLabel("No Image 1 Loaded")
        self.image1_label.setAlignment(Qt.AlignCenter)
        self.image1_label.setStyleSheet("border: 1px solid rgb(40, 57, 153);")
        self.image1_label.setFixedSize(330, 330)
        self.image1_label.mouseDoubleClickEvent = lambda event: self.load_image(1)

        self.image2_label = QLabel("No Image 2 Loaded")
        self.image2_label.setAlignment(Qt.AlignCenter)
        self.image2_label.setStyleSheet("border: 1px solid rgb(40, 57, 153);")
        self.image2_label.setFixedSize(330, 330)
        self.image2_label.mouseDoubleClickEvent = lambda event: self.load_image(2)

        self.cutoff1 = QSpinBox()
        self.cutoff1.setRange(1, 100)
        self.cutoff1.setValue(10)

        self.cutoff2 = QSpinBox()
        self.cutoff2.setRange(1, 100)
        self.cutoff2.setValue(10)

        self.type1 = QComboBox()
        self.type1.addItems(["lp", "hp"])

        self.type2 = QComboBox()
        self.type2.addItems(["lp", "hp"])

        self.btn_hybrid = QPushButton("Create Hybrid Image")
        self.btn_hybrid.clicked.connect(self.create_hybrid_image)

        self.hybridLayout = QVBoxLayout()

        cutoff1_layout = QHBoxLayout()
        cutoff1_layout.addWidget(QLabel("Cutoff 1"))
        cutoff1_layout.addWidget(self.cutoff1)
        self.hybridLayout.addLayout(cutoff1_layout)

        type1_layout = QHBoxLayout()
        type1_layout.addWidget(QLabel("Type 1"))
        type1_layout.addWidget(self.type1)
        self.hybridLayout.addLayout(type1_layout)

        cutoff2_layout = QHBoxLayout()
        cutoff2_layout.addWidget(QLabel("Cutoff 2"))
        cutoff2_layout.addWidget(self.cutoff2)
        self.hybridLayout.addLayout(cutoff2_layout)

        type2_layout = QHBoxLayout()
        type2_layout.addWidget(QLabel("Type 2"))
        type2_layout.addWidget(self.type2)
        self.hybridLayout.addLayout(type2_layout)

        self.hybridLayout.addWidget(self.btn_hybrid)
        
        images_layout = QVBoxLayout()
        images_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        images_layout.addSpacing(10)
        images_layout.addWidget(self.image1_label)
        images_layout.addWidget(self.image2_label)
        self.hybridLayout.addLayout(images_layout)

        hybrid_image_layout.addLayout(self.hybridLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(hybrid_image_frame)

    def load_image(self, image_number):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            image = cv2.imread(file_path)
            if image_number == 1:
                self.image1 = image
                self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
                self.display_image(self.image1, self.image1_label)
            elif image_number == 2:
                
                self.image2 = image
                self.image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
                self.display_image(self.image2, self.image2_label)

    def display_image(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))

    def create_hybrid_image(self):
        if self.image1 is None or self.image2 is None:
            QMessageBox.warning(self, "Warning", "Please load both images first.")
            return

        cutoff1 = self.cutoff1.value()
        cutoff2 = self.cutoff2.value()
        type1 = self.type1.currentText()
        type2 = self.type2.currentText()

        hybrid_image = self.parent.processors['frequency'].create_hybrid_image(self.image1, self.image2, cutoff1, cutoff2, type1, type2)
        self.parent.display_image(hybrid_image)
