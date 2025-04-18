from PyQt5.QtWidgets import (
     QPushButton, QLabel,  QFrame,
    QVBoxLayout, QWidget,  QSpinBox, QDoubleSpinBox, QHBoxLayout
)

from matplotlib.figure import Figure


from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                             QPushButton)


from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)
from PyQt5.QtCore import Qt

class ActiveContourTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        active_contour_frame = QFrame()
        active_contour_frame.setObjectName("active_contour_frame")
        active_contour_layout = QVBoxLayout(active_contour_frame)
        active_contour_layout.setAlignment(Qt.AlignTop)

        self.centerX = QSpinBox()
        self.centerX.setRange(0, 1000)
        self.centerX.setValue(250)

        self.centerY = QSpinBox()
        self.centerY.setRange(0, 1000)
        self.centerY.setValue(200)

        self.radius = QSpinBox()
        self.radius.setRange(1, 500)
        self.radius.setValue(200)

        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 10.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(0.5)

        self.beta = QDoubleSpinBox()
        self.beta.setRange(0.0, 10.0)
        self.beta.setSingleStep(0.1)
        self.beta.setValue(0.7)

        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.0, 10.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setValue(1)

        self.iterations = QSpinBox()
        self.iterations.setRange(1, 10000)
        self.iterations.setValue(1000)

        self.points = QSpinBox()
        self.points.setRange(1, 10000)
        self.points.setValue(100)
        
        self.w_edge = QSpinBox()
        self.w_edge.setRange(1, 10)
        self.w_edge.setValue(10)
        
        self.convergence = QSpinBox()
        self.convergence.setRange(0, 1)
        self.beta.setSingleStep(0.01)
        self.convergence.setValue(0)
        
        self.btn_run_snake = QPushButton("Run Active Contour")
        self.btn_run_snake.clicked.connect(parent.run_active_contour)

        center_layout = QHBoxLayout()
        center_layout.addWidget(QLabel("Center X"))
        center_layout.addWidget(self.centerX)
        center_layout.addWidget(QLabel("Center Y"))
        center_layout.addWidget(self.centerY)
        active_contour_layout.addLayout(center_layout)

        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius"))
        radius_layout.addWidget(self.radius)
        active_contour_layout.addLayout(radius_layout)

        alpha_layout = QHBoxLayout()
        alpha_layout.addWidget(QLabel("Alpha"))
        alpha_layout.addWidget(self.alpha)
        active_contour_layout.addLayout(alpha_layout)

        beta_layout = QHBoxLayout()
        beta_layout.addWidget(QLabel("Beta"))
        beta_layout.addWidget(self.beta)
        active_contour_layout.addLayout(beta_layout)

        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma"))
        gamma_layout.addWidget(self.gamma)
        active_contour_layout.addLayout(gamma_layout)

        iterations_layout = QHBoxLayout()
        iterations_layout.addWidget(QLabel("Iterations"))
        iterations_layout.addWidget(self.iterations)
        active_contour_layout.addLayout(iterations_layout)

        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel("Points"))
        points_layout.addWidget(self.points)
        active_contour_layout.addLayout(points_layout)
        
        w_edge_layout = QHBoxLayout()
        w_edge_layout.addWidget(QLabel("Edge weight"))
        w_edge_layout.addWidget(self.w_edge)
        active_contour_layout.addLayout(w_edge_layout)

        convergence_layout = QHBoxLayout()
        convergence_layout.addWidget(QLabel("Convergance"))
        convergence_layout.addWidget(self.convergence)
        active_contour_layout.addLayout(convergence_layout)

        active_contour_layout.addWidget(self.btn_run_snake)

        main_layout.addWidget(active_contour_frame)