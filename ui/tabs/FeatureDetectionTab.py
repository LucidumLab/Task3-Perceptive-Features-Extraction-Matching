from PyQt5.QtWidgets import (
     QPushButton, QLabel,  QFrame,
    QVBoxLayout, QWidget, QFileDialog, QSplitter,QSpinBox, QDoubleSpinBox, QHBoxLayout, QComboBox, QGridLayout, QLineEdit, QCheckBox, QTabWidget, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)

from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QImage, QPixmap
import cv2


class Feature_DetectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        
        self.btn_sift = QPushButton("Apply SIFT")
        self.btn_sift.clicked.connect(self.apply_sift)
        self.btn_sift.setFixedSize(450, 50)  
        layout.addWidget(self.btn_sift, alignment=Qt.AlignCenter)

        
        self.btn_harris = QPushButton("Apply Harris")
        self.btn_harris.setFixedSize(450, 50)
        self.btn_harris.clicked.connect(self.apply_harris)
        layout.addWidget(self.btn_harris, alignment=Qt.AlignCenter)

    def apply_sift(self):
        """
        Function to apply SIFT.
        """
        print("SIFT function applied.")
        

    def apply_harris(self):
        """
        Function to apply Harris corner detection.
        """
        print("Harris function applied.")
        

class Feature_Detection_Frame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)

        
        splitter = QSplitter(Qt.Horizontal)
        splitter.setObjectName("image_splitter")

        
        self.image_1 = QLabel("Input (Double-click to load image)")
        self.image_1.setMinimumSize(QSize(650, 650))  
        self.image_1.setObjectName("original_label")
        self.image_1.setAlignment(Qt.AlignCenter)
        self.image_1.mouseDoubleClickEvent = self.on_input_double_click
        splitter.addWidget(self.image_1)

        
        self.image_2 = QLabel("Output")
        self.image_2.setMinimumSize(QSize(650, 650))  
        self.image_2.setObjectName("template_label")
        self.image_2.setAlignment(Qt.AlignCenter)
        splitter.addWidget(self.image_2)

        
        main_layout.addWidget(splitter)

        
        self.input_image = None
        self.output_image = None


    def on_input_double_click(self, event):
        """Handle double-click on the input label to load an image"""
        self.load_input_image()
    
    def load_input_image(self):
        """Load an image from disk and display it in the input label"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp *.jpeg)")
        
        if file_path:
            
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return
                
            
            self.input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            
            self.display_image(self.input_image, 1)
            
            
            
            
            
    
    def display_image(self, image, label_number):
        """
        Displays the given image in the specified label.

        Args:
            image (numpy.ndarray): The image to display.
            label_number (int): The label number (1 for Input, 2 for Output).
        """
        if image is None:
            return
            
        if len(image.shape) == 3: 
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        elif len(image.shape) == 2:  
            h, w = image.shape
            qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            QMessageBox.critical(self, "Error", "Unsupported image format.")
            return

        pixmap = QPixmap.fromImage(qimg)
        
        if label_number == 1:
            self.image_1.setPixmap(pixmap.scaled(self.image_1.width(), self.image_1.height(), Qt.KeepAspectRatio))
        elif label_number == 2:
            self.image_2.setPixmap(pixmap.scaled(self.image_2.width(), self.image_2.height(), Qt.KeepAspectRatio))
