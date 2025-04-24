from PyQt5.QtWidgets import (
     QPushButton, QLabel,  QFrame,
    QVBoxLayout, QWidget, QFileDialog,QSplitter, QSpinBox, QDoubleSpinBox, QHBoxLayout, QComboBox, QGridLayout, QLineEdit, QCheckBox, QTabWidget, QGroupBox, QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)

from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtGui import QImage, QPixmap
import cv2


from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QScrollArea, QDoubleSpinBox,
    QSpinBox, QLineEdit, QPushButton, QLabel, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt
from sift.const import * 

class Feature_DetectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QFormLayout(scroll_widget)

        
        self.inputs = {}

        
        variables = [
            ("Number of Octaves", 'int', nr_octaves),
            ("Scales per Octave", 'int', scales_per_octave),
            ("Auxiliary Scales", 'int', auxiliary_scales),
            ("Original Pixel Distance", 'float', orig_pixel_dist),
            ("Minimum Pixel Distance", 'float', min_pixel_dist),
            ("First Upscale", 'float', first_upscale),
            ("Original Sigma", 'float', orig_sigma),
            ("Minimum Sigma", 'float', min_sigma),
            ("Initial Sigma", 'float', init_sigma),
            ("Magnitude Threshold", 'float', magnitude_thresh),
            ("Coarse Magnitude Threshold", 'float', coarse_magnitude_thresh),
            ("Offset Threshold", 'float', offset_thresh),
            ("Edge Ratio Threshold", 'float', edge_ratio_thresh),
            ("Maximum Interpolations", 'int', max_interpolations),
            ("Number of Bins", 'int', nr_bins),
            ("Reference Locality", 'float', reference_locality),
            ("Reference Patch Width Scalar", 'float', reference_patch_width_scalar),
            ("Number of Smoothing Iterations", 'int', nr_smooth_iter),
            ("Relative Peak Threshold", 'float', rel_peak_thresh),
            ("Mask Neighbors", 'int', mask_neighbors),
            ("Max Orientations per Keypoint", 'int', max_orientations_per_keypoint),
            ("Descriptor Locality", 'float', descriptor_locality),
            ("Number of Descriptor Histograms", 'int', nr_descriptor_histograms),
            ("Inter-Histogram Distance", 'float', inter_hist_dist),
            ("Number of Descriptor Bins", 'int', nr_descriptor_bins),
            ("Descriptor Bin Width", 'float', descriptor_bin_width),
            ("Descriptor Clip Max", 'float', descriptor_clip_max),
            ("Relative Distance Match Threshold", 'float', rel_dist_match_thresh),
        ]

        
        for var_name, var_type, default in variables:
            if var_type == 'int':
                input_widget = QSpinBox()
                input_widget.setRange(-1000, 1000)
                input_widget.setValue(default)
                input_widget.valueChanged.connect(self.update_constants)  # Connect valueChanged signal

            elif var_type == 'float':
                input_widget = QDoubleSpinBox()
                input_widget.setDecimals(5)
                input_widget.setRange(-10000.0, 10000.0)
                input_widget.setValue(default)
                input_widget.valueChanged.connect(self.update_constants)  # Connect valueChanged signal

            else:
                input_widget = QLineEdit()
                input_widget.setText(str(default))
                input_widget.valueChanged.connect(self.update_constants)  # Connect valueChanged signal


            self.inputs[var_name] = input_widget
            scroll_layout.addRow(f"{var_name}:", input_widget)

        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        
        self.btn_sift = QPushButton("Apply SIFT")
        self.btn_sift.setFixedSize(450, 50)
        layout.addWidget(self.btn_sift, alignment=Qt.AlignCenter)

        self.btn_harris = QPushButton("Apply Harris")
        self.btn_harris.setFixedSize(450, 50)
        self.btn_harris.clicked.connect(self.apply_harris)
        layout.addWidget(self.btn_harris, alignment=Qt.AlignCenter)

    def update_constants(self):
        """
        Function to update the constants with the new values from the input fields.
        """
        for var_name, input_widget in self.inputs.items():
            if isinstance(input_widget, QSpinBox):
                # Update the corresponding variable in your constants file
                globals()[var_name.replace(" ", "_").lower()] = input_widget.value()
            elif isinstance(input_widget, QDoubleSpinBox):
                globals()[var_name.replace(" ", "_").lower()] = input_widget.value()
            elif isinstance(input_widget, QLineEdit):
                globals()[var_name.replace(" ", "_").lower()] = input_widget.text()

        print("Updated constants:")
        for var_name in self.inputs:
            print(f"{var_name}: {globals()[var_name.replace(' ', '_').lower()]}")  
   
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
        self.image_1.setMinimumSize(QSize(650,650))  
        self.image_1.setObjectName("original_label")
        self.image_1.setAlignment(Qt.AlignCenter)
        self.image_1.mouseDoubleClickEvent = self.on_input_double_click
        splitter.addWidget(self.image_1)

        
        self.image_2 = QLabel("Output")
        self.image_2.setMinimumSize(QSize(650,650))  
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
