from PyQt5.QtWidgets import (
    QPushButton, QLabel, QFrame,
    QVBoxLayout, QWidget, QScrollArea, QSpinBox, QDoubleSpinBox, QHBoxLayout
)

from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton,  QScrollArea)
from PyQt5.QtCore import Qt
import numpy as np


class HoughTransformTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create a scroll area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # Hide horizontal scrollbar
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)    # Hide vertical scrollbar

        # Create a widget to hold the content
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        # Main layout for the tab
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)

        # Layout for the content inside the scroll area
        hough_transform_layout = QVBoxLayout(content_widget)

        ############################## Shared Canny Detector Parameters ##############################
        canny_group_frame = QFrame()
        canny_group_frame.setObjectName("canny_group_frame")
        canny_layout = QVBoxLayout(canny_group_frame)

        self.cannyLowThreshold = QSpinBox()
        self.cannyLowThreshold.setRange(0, 255)
        self.cannyLowThreshold.setValue(50)

        self.cannyHighThreshold = QSpinBox()
        self.cannyHighThreshold.setRange(0, 255)
        self.cannyHighThreshold.setValue(150)

        self.cannyBlurKSize = QSpinBox()
        self.cannyBlurKSize.setRange(1, 15)
        self.cannyBlurKSize.setValue(5)

        low_threshold_layout = QHBoxLayout()
        low_threshold_layout.addWidget(QLabel("Canny Low Threshold"))
        low_threshold_layout.addWidget(self.cannyLowThreshold)
        canny_layout.addLayout(low_threshold_layout)

        high_threshold_layout = QHBoxLayout()
        high_threshold_layout.addWidget(QLabel("Canny High Threshold"))
        high_threshold_layout.addWidget(self.cannyHighThreshold)
        canny_layout.addLayout(high_threshold_layout)

        blur_ksize_layout = QHBoxLayout()
        blur_ksize_layout.addWidget(QLabel("Blur Kernel Size"))
        blur_ksize_layout.addWidget(self.cannyBlurKSize)
        canny_layout.addLayout(blur_ksize_layout)

        hough_transform_layout.addWidget(canny_group_frame)

        ############################## Line Detection Parameters ##############################
        line_group_frame = QFrame()
        line_group_frame.setObjectName("line_group_frame")
        line_layout = QVBoxLayout(line_group_frame)

        self.numRho = QSpinBox()
        self.numRho.setRange(1, 500)
        self.numRho.setValue(180)

        self.numTheta = QSpinBox()
        self.numTheta.setRange(1, 500)
        self.numTheta.setValue(180)

        self.houghThresholdRatio = QDoubleSpinBox()
        self.houghThresholdRatio.setRange(0.0, 1.0)
        self.houghThresholdRatio.setSingleStep(0.1)
        self.houghThresholdRatio.setValue(0.6)

        self.btn_detect_lines = QPushButton("Detect Lines")
        self.btn_detect_lines.clicked.connect(parent.detect_lines)

        num_rho_layout = QHBoxLayout()
        num_rho_layout.addWidget(QLabel("Num Rho"))
        num_rho_layout.addWidget(self.numRho)
        line_layout.addLayout(num_rho_layout)

        num_theta_layout = QHBoxLayout()
        num_theta_layout.addWidget(QLabel("Num Theta"))
        num_theta_layout.addWidget(self.numTheta)
        line_layout.addLayout(num_theta_layout)

        hough_threshold_ratio_layout = QHBoxLayout()
        hough_threshold_ratio_layout.addWidget(QLabel("Hough Threshold Ratio"))
        hough_threshold_ratio_layout.addWidget(self.houghThresholdRatio)
        line_layout.addLayout(hough_threshold_ratio_layout)

        line_layout.addWidget(self.btn_detect_lines)

        hough_transform_layout.addWidget(line_group_frame)

        ############################## Circle Detection Parameters ##############################
        circle_group_frame = QFrame()
        circle_group_frame.setObjectName("circle_group_frame")
        circle_layout = QVBoxLayout(circle_group_frame)

        self.rMin = QSpinBox()
        self.rMin.setRange(1, 100)
        self.rMin.setValue(20)

        self.rMax = QSpinBox()
        self.rMax.setRange(1, 500)
        self.rMax.setValue(100)

        self.numThetas = QSpinBox()
        self.numThetas.setRange(1, 360)
        self.numThetas.setValue(50)

        self.btn_detect_circles = QPushButton("Detect Circles")
        self.btn_detect_circles.clicked.connect(parent.detect_circles)

        r_min_layout = QHBoxLayout()
        r_min_layout.addWidget(QLabel("Min Radius"))
        r_min_layout.addWidget(self.rMin)
        circle_layout.addLayout(r_min_layout)

        r_max_layout = QHBoxLayout()
        r_max_layout.addWidget(QLabel("Max Radius"))
        r_max_layout.addWidget(self.rMax)
        circle_layout.addLayout(r_max_layout)

        num_thetas_layout = QHBoxLayout()
        num_thetas_layout.addWidget(QLabel("Num Thetas"))
        num_thetas_layout.addWidget(self.numThetas)
        circle_layout.addLayout(num_thetas_layout)

        circle_layout.addWidget(self.btn_detect_circles)

        hough_transform_layout.addWidget(circle_group_frame)

        ############################## Ellipse Detection Parameters ##############################
        ellipse_group_frame = QFrame()
        ellipse_group_frame.setObjectName("ellipse_group_frame")
        ellipse_layout = QVBoxLayout(ellipse_group_frame)

        self.aMin = QSpinBox()
        self.aMin.setRange(1, 500)
        self.aMin.setValue(20)

        self.aMax = QSpinBox()
        self.aMax.setRange(1, 500)
        self.aMax.setValue(100)


        self.thetaStep = QSpinBox()
        self.thetaStep.setRange(1, 180)
        self.thetaStep.setValue(10)

        self.ellipseThresholdRatio = QDoubleSpinBox()
        self.ellipseThresholdRatio.setRange(0.0, 1.0)
        self.ellipseThresholdRatio.setSingleStep(0.1)
        self.ellipseThresholdRatio.setValue(0.5)

        self.minDist = QSpinBox()
        self.minDist.setRange(1, 100)
        self.minDist.setValue(20)

        self.btn_detect_ellipses = QPushButton("Detect Ellipses")
        self.btn_detect_ellipses.clicked.connect(parent.detect_ellipses)

        a_min_layout = QHBoxLayout()
        a_min_layout.addWidget(QLabel("Min Major Axis (aMin)"))
        a_min_layout.addWidget(self.aMin)
        ellipse_layout.addLayout(a_min_layout)

        a_max_layout = QHBoxLayout()
        a_max_layout.addWidget(QLabel("Max Major Axis (aMax)"))
        a_max_layout.addWidget(self.aMax)
        ellipse_layout.addLayout(a_max_layout)


        theta_step_layout = QHBoxLayout()
        theta_step_layout.addWidget(QLabel("Theta Step"))
        theta_step_layout.addWidget(self.thetaStep)
        ellipse_layout.addLayout(theta_step_layout)

        ellipse_threshold_ratio_layout = QHBoxLayout()
        ellipse_threshold_ratio_layout.addWidget(QLabel("Threshold Ratio"))
        ellipse_threshold_ratio_layout.addWidget(self.ellipseThresholdRatio)
        ellipse_layout.addLayout(ellipse_threshold_ratio_layout)


        ellipse_layout.addWidget(self.btn_detect_ellipses)

        hough_transform_layout.addWidget(ellipse_group_frame)
