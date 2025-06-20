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
                input_widget.valueChanged.connect(self.update_constants)  

            elif var_type == 'float':
                input_widget = QDoubleSpinBox()
                input_widget.setDecimals(5)
                input_widget.setRange(-10000.0, 10000.0)
                input_widget.setValue(default)
                input_widget.valueChanged.connect(self.update_constants)  

            else:
                input_widget = QLineEdit()
                input_widget.setText(str(default))
                input_widget.valueChanged.connect(self.update_constants)  


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

        
        grid_layout = QGridLayout()

        self.image1_array = None
        self.image2_array = None
        self.image3_array = None
        self.image_1 = QLabel("Input (Double-click to load image)")
        self.image_1.setMinimumSize(QSize(650, 430))
        self.image_1.setObjectName("original_label")
        self.image_1.setAlignment(Qt.AlignCenter)
        self.image_1.mouseDoubleClickEvent = lambda event: self.on_input_double_click(self.image_1, 1)

        self.image_2 = QLabel("Output")
        self.image_2.setMinimumSize(QSize(650, 430))
        self.image_2.setObjectName("template_label")
        self.image_2.setAlignment(Qt.AlignCenter)
        self.image_2.mouseDoubleClickEvent = lambda event: self.on_input_double_click(self.image_2, 2)

        self.image_3 = QLabel("Output 2")
        self.image_3.setMinimumSize(QSize(650, 430))
        self.image_3.setObjectName("template_label")
        self.image_3.setAlignment(Qt.AlignCenter)
        self.image_3.mouseDoubleClickEvent = lambda event: self.on_input_double_click(self.image_3)

        
        grid_layout.addWidget(self.image_1, 0, 0)  
        grid_layout.addWidget(self.image_2, 0, 1)  
        grid_layout.addWidget(self.image_3, 1, 1)  

        
        main_layout.addLayout(grid_layout)

        
        self.input_image = None
        self.output_image = None
        self.output_image_2 = None

    def on_input_double_click(self, label, idx = 0):
        """Handle double-click on a label to load an image"""
        self.load_input_image(label, idx)

    def load_input_image(self, label, idx):
        """Load an image from disk and display it in the specified label"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp *.jpeg)")

        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            print(image.shape)
            
            if idx == 1:
                self.image1_array = image
                print(f'image_uploaded: {self.image1_array.shape}')
            elif idx == 2:
                self.image2_array = image
            else:
                self.image3_array = image
            if image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return

            self.display_image(image, label)

    def display_image(self, image, label):
        """
        Displays the given image in the specified label.

        Args:
            image (numpy.ndarray): The image to display.
            label (QLabel): The label to display the image in.
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
        label.setPixmap(pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio))