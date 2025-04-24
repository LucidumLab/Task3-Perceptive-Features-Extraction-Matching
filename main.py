import sys
import cv2
import numpy as np
import os
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QSize

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QFrame, QTabWidget, QSpacerItem, QSizePolicy,
    QVBoxLayout, QWidget, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout, QLineEdit, QCheckBox,
    QStackedWidget, QGridLayout
)

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,QComboBox, QSpinBox,QDoubleSpinBox, QFrame
)


from sift_matching import *
from sift.harris import CornerDetection
from models.ImageModel import ImageModel

# from ui.tabs.ActiveContourTab import ActiveContourTab
# from ui.tabs.HoughTransformTab import HoughTransformTab
# from ui.tabs.NoiseFilterTab import NoiseFilterTab
# from ui.tabs.EdgeDetectionTab import EdgeDetectionTab
# from ui.tabs.ThresholdingTab import ThresholdingTab
# from ui.tabs.FrequencyFilterTab import FrequencyFilterTab
# from ui.tabs.HybridImageTab import HybridImageTab
from ui.tabs.FeatureDetectionTab import Feature_Detection_Frame , Feature_DetectionTab





class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Image Processing App")
        self.setGeometry(50, 50, 1200, 800)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout()
        central_widget.setLayout(self.main_layout)
        
        self.init_ui(self.main_layout)

        # Single data structure to store all parameters
        self.params = {
            "noise_filter": {},
            "filtering": {},
            "edge_detection": {},
            "thresholding": {},
            "frequency_filter": {},
            "hybrid_image": {},
            "shape_detection":{},
            "active_contour":{}
            
        }
        
        # self.connect_signals()
        # Image & Processor Variables
        self.image = None
        self.original_image = None
        self.modified_image = None



    def init_ui(self, main_layout):
        # Left Frame
        left_frame = QFrame()
        left_frame.setFixedWidth(500)
        left_frame.setObjectName("left_frame")
        left_layout = QVBoxLayout(left_frame)
        
        tab_widget = QTabWidget()
        tab_widget.setObjectName("tab_widget")

        # Feature Detection Tab


        self.feature_detection_tab = Feature_DetectionTab(self)
        tab_widget.addTab(self.feature_detection_tab, "Feature Detection")

        tab_widget.currentChanged.connect(self.on_tab_changed)

        left_layout.addWidget(tab_widget)
        main_layout.addWidget(left_frame)
        
        # Right Frame
        self.right_frame = QFrame()
        self.right_frame.setObjectName("right_frame")
        self.right_layout = QVBoxLayout(self.right_frame)
        self.right_layout.setAlignment(Qt.AlignTop)  # Center the content vertically and horizontally

        # Control Buttons Frame
        control_frame = QFrame()
        control_frame.setMaximumHeight(100)
        control_layout = QHBoxLayout(control_frame)

        control_layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.btn_confirm = QPushButton()
        self.btn_confirm.setIcon(QIcon(os.path.join(os.path.dirname(__file__), './resources/confirm.png')))
        self.btn_confirm.setIconSize(QSize(28, 28))
        self.btn_confirm.clicked.connect(self.confirm_edit)
        control_layout.addWidget(self.btn_confirm)

        self.btn_discard = QPushButton()
        self.btn_discard.setIcon(QIcon(os.path.join(os.path.dirname(__file__), './resources/discard.png')))
        self.btn_discard.setIconSize(QSize(28, 28))
        self.btn_discard.clicked.connect(self.discard_edit)
        control_layout.addWidget(self.btn_discard)

        self.btn_reset = QPushButton()
        self.btn_reset.setIcon(QIcon(os.path.join(os.path.dirname(__file__), './resources/reset.png')))
        self.btn_reset.setIconSize(QSize(28, 28))
        self.btn_reset.clicked.connect(self.reset_image)
        control_layout.addWidget(self.btn_reset)


        self.right_layout.addWidget(control_frame)

        # Feature Detection Frame
        self.feature_detection_frame = Feature_Detection_Frame()
        self.feature_detection_layout = QVBoxLayout(self.feature_detection_frame)
        self.feature_detection_layout.setAlignment(Qt.AlignCenter)  # Center the content vertically and horizontally
        self.right_layout.addWidget(self.feature_detection_frame, alignment=Qt.AlignCenter)  # Center the frame
        self.feature_detection_tab.btn_sift.clicked.connect(self.sift_matching)
        self.feature_detection_tab.btn_harris.clicked.connect(self.detect_corners)

        main_layout.addWidget(self.right_frame)

    def sift_matching(self):
        image_1 = self.feature_detection_frame.image1_array
        image_2 = self.feature_detection_frame.image2_array
        
        output_image, _, _, _ = apply_sift(image_1, image_2)
        
 
        label = self.feature_detection_frame.image_3
        self.feature_detection_frame.display_image(output_image, label)
    
    def detect_corners(self):
        image_1 = self.feature_detection_frame.image1_array
        
        print(image_1.shape)
        image_model1 = ImageModel()
        image_model1.set_image(image_1)  

        corner_detector = CornerDetection()

        harris_corners = corner_detector.corner_detection_harris(image_model1)
        shi_corners = corner_detector.corner_detection_shi(image_model1)
        label = self.feature_detection_frame.image_2

        output_image = corner_detector.visualize_corners(image_1, harris_corners)
        self.feature_detection_frame.display_image(output_image, label)


              
    def on_tab_changed(self, index):
        """
        Switch the content of the right frame based on the selected tab.
        """
        if index == 7:  # Assuming "Feature Detection" is the 8th tab (index 7)
            self.content_stack.setCurrentWidget(self.feature_detection_frame)
        else:
            self.content_stack.setCurrentWidget(self.image_display_frame)
    
    def update_params(self, tab_name, ui_components):
        """
        Update the parameters for a specific tab based on the UI components.
        
        Args:
            tab_name (str): The name of the tab (e.g., "noise_filter").
            ui_components (dict): A dictionary of UI components and their keys.
        """
        print("Updating params for", tab_name)
        self.params[tab_name] = {}
        for key, widget in ui_components.items():
            if isinstance(widget, (QComboBox, QLineEdit)):
                self.params[tab_name][key] = widget.currentText() if isinstance(widget, QComboBox) else widget.text()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                self.params[tab_name][key] = widget.value()
            elif isinstance(widget, QCheckBox):
                self.params[tab_name][key] = widget.isChecked()
        
        print(self.params[tab_name])        

    def on_image_label_double_click(self, event):
        self.load_image()
    

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap.scaled(self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio))

    def apply_noise(self):
        """
        Applies noise to the image based on the selected noise type and parameters from the UI.
        """
        # Retrieve noise parameters from the params dictionary
        noise_params = self.params["noise_filter"]

        # Call the add_noise function with the retrieved parameters
        self._add_noise(**noise_params)
        print("Applying noise:", noise_params)
        self.display_image(self.modified_image)
        
    def _add_noise(self, **kwargs):
        """
        Adds noise to the image based on the specified noise type and parameters.

        Args:
            noise_type (str): Type of noise to add. Options: "uniform", "gaussian", "salt_pepper".
            **kwargs: Additional parameters for the noise (e.g., intensity, mean, std, salt_prob, pepper_prob).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying new noise

        if self.image is not None:
            # Call the noise processor with the specified noise type and parameters
            noisy_image = self.processors['noise'].add_noise(**kwargs)
            self.modified_image = noisy_image
            self.display_image(self.modified_image, modified=True)
        else:
            raise ValueError("No image loaded. Please load an image before applying noise.")
            
    def apply_filter(self):
        """
        Applies a filter to the image based on the selected filter type and parameters from the UI.
        """
        # Retrieve filter parameters from the params dictionary
        filter_params = self.params.get("filtering", {})
        # Call the apply_filters function with the retrieved parameters
        self._apply_filters(**filter_params)
        print("Applying filter:", filter_params)
        self.display_image(self.modified_image)

    def _apply_filters(self, **kwargs):
        """
        Applies a filter to the image based on the specified filter type and parameters.

        Args:
            filter_type (str): Type of filter to apply. Options: "average", "gaussian", "median".
            **kwargs: Additional parameters for the filter (e.g., kernel_size, sigma).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying a new filter

        if self.image is not None:
            # Apply the filter using the specified parameters
            filtered_image = self.processors['noise'].apply_filters( **kwargs)
            self.modified_image = filtered_image.get(kwargs.get("filter_type", "median")) # Default to "median" filter
        else:
            raise ValueError("No image loaded. Please load an image before applying a filter.")
    def detect_edges(self):
        """
        Detects edges in the image based on the selected edge detection type and parameters from the UI.
        """
        # Retrieve edge detection parameters from the params dictionary
        edge_params = self.params.get("edge_detection", {})

        # Call the _detect_edges function with the retrieved parameters
        self._detect_edges(**edge_params)
        print("Detecting edges:", edge_params)
        
        self.display_image(self.modified_image, modified=True)

    def _detect_edges(self, **kwargs):
        """ 
        Detects edges in the image based on the specified edge detection type and parameters.

        Args:
            edge_type (str): Type of edge detection to apply. Options: "sobel", "canny", "prewitt", "roberts".
            **kwargs: Additional parameters for the edge detection (e.g., kernel_size, sigma, thresholds).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying edge detection

        if self.image is not None:
            # Apply edge detection using the specified parameters
            edge_map = self.processors['edge_detector'].detect_edges(**kwargs)
            self.modified_image = edge_map
        else:
            raise ValueError("No image loaded. Please load an image before detecting edges.")
    
    def apply_thresholding(self):
        """
        Applies thresholding to the image based on the selected thresholding type and parameters from the UI.
        """
        # Retrieve thresholding parameters from the params dictionary
        threshold_params = self.params.get("thresholding", {})
        # Call the _apply_thresholding function with the retrieved parameters
        print("Applying thresholding:", threshold_params)

        self._apply_thresholding( **threshold_params)
        self.display_image(self.modified_image, modified=True)
    
        
        
        
    def _apply_thresholding(self,  **kwargs):
        """
        Applies thresholding to the image based on the specified thresholding type and parameters.

        Args:
            threshold_type (str): Type of thresholding to apply. Options: "global", "local".
            **kwargs: Additional parameters for the thresholding (e.g., threshold_value, kernel_size, k_value).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying thresholding

        if self.image is not None:
            # Apply thresholding using the specified parameters
            thresholded_image = self.processors['thresholding'].apply_thresholding(**kwargs)
            self.modified_image = thresholded_image
        else:
            raise ValueError("No image loaded. Please load an image before applying thresholding.")
    

    
    def apply_frequency_filter(self):
        """
        Applies a frequency filter to the image based on the selected filter type and parameters from the UI.
        """
        # Retrieve frequency filter parameters from the params dictionary
        frequency_params = self.params.get("frequency_filter", {})
        # Call the _apply_frequency_filter function with the retrieved parameters
        self._apply_frequency_filter(**frequency_params)
        print("Applying frequency filter:", frequency_params)
        self.display_image(self.modified_image, modified=True)
        

    def _apply_frequency_filter(self, **kwargs):
        """
        Applies a frequency filter to the image based on the specified filter type and parameters.

        Args:
            filter_type (str): Type of frequency filter to apply. Options: "low_pass", "high_pass".
            **kwargs: Additional parameters for the frequency filter (e.g., radius).
        """
        if self.modified_image is not None:
            self.confirm_edit()  # Confirm any previous edits before applying frequency filtering

        if self.image is not None:
            # Apply frequency filter using the specified parameters
            filtered_image = self.processors['frequency'].apply_filter(**kwargs)
            self.modified_image = filtered_image
        else:
            raise ValueError("No image loaded. Please load an image before applying frequency filtering.")
   
    def equalize(self):
        self.modified_image = self.processors['image'].get_equalized_image()
        self.display_image(self.modified_image)

    def normalize(self):
        self.modified_image = self.processors['image'].get_normalized_image()
        self.display_image(self.modified_image)

    def load_image(self, hybird = False):
        """
        Load an image from disk and display it in the UI.
        """
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path and hybird == False:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.original_image = self.image
            if self.image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return
          
            self.display_image(self.image)
        elif hybird == True:
            self.extra_image = cv2.imread(file_path)
            if self.extra_image is None:
                QMessageBox.critical(self, "Error", "Failed to load image.")
                return

            self.display_image(self.extra_image, hybird = True)
        else:
            QMessageBox.information(self, "Info", "No file selected.")

    def display_image(self, img, hybrid=False, modified=False):
        """
        Convert a NumPy BGR image to QImage and display it in lbl_image.
        """
        if len(img.shape) == 3:
            # Convert BGR to RGB
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        else:
            # Grayscale
            h, w = img.shape
            # Ensure the image is in uint8 format
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            # Convert the NumPy array to bytes
            img_bytes = img.tobytes()
            qimg = QImage(img_bytes, w, h, w, QImage.Format_Indexed8)
        
        pixmap = QPixmap.fromImage(qimg)
        self.lbl_image.setPixmap(pixmap.scaled(
            self.lbl_image.width(), self.lbl_image.height(), Qt.KeepAspectRatio
        ))
    

    def equalize(self):
        """
        Example: Using the factory to create a HistogramProcessor.
        """
        if self.modified_image is not None:
            self.confirm_edit()
            
        if self.image is not None:
            equalized_image = self.processors['image'].get_equalized_image() 
            self.modified_image = equalized_image
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError("No image available. Load an image first.")
    def normalize(self):
        """
        Example: Using the factory to create a HistogramProcessor.
        """
        if self.modified_image is not None:
            self.confirm_edit()
        if self.image is not None:
            normalized_image = self.processors['image'].get_normalized_image() 
            self.modified_image = normalized_image
            self.display_image(self.modified_image, modified = True)
        else:
            raise ValueError("No image available. Load an image first.")
    
    def confirm_edit(self):
        """
        Confirm the edit.
        """
        if self.modified_image is not None:
            self.image = self.modified_image
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.modified_image = None
            self.display_image(self.image)
        else:
            raise ValueError("No image available. Load an image first.")
    
    def discard_edit(self):
        """
        Discard the edit.
        """
        if self.modified_image is not None:
            self.modified_image = None
            self.display_image(self.image)
        else:
            raise ValueError("No image available. Load an image first.")
    def reset_image(self):
        """
        Reset the image to the original.
        """
        if self.original_image is not None:
            self.image = self.original_image
            for processor in self.processors.values():
                processor.set_image(self.image)
            self.modified_image = None
            self.display_image(self.image)
        else:
            raise ValueError("No original image available. Load an image first.")

def main():
    app = QApplication(sys.argv)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    qss_path = os.path.join(script_dir, "resources\\styles.qss")
    
    with open(qss_path, "r") as file:
        app.setStyleSheet(file.read())
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


import cv2
import numpy as np

if __name__ == "__main__":
    main()
