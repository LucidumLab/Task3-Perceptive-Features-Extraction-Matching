from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel,QCheckBox
from PyQt5.QtCore import Qt


class ImageFrame(QWidget):
    def __init__(self, title, group, checked=False):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        self.checkbox = QCheckBox()
        self.checkbox.setMaximumWidth(30)
        self.checkbox.setChecked(checked)
        self.checkbox.setStyleSheet("color: white; font: 600 12pt 'Segoe UI';border:none;")
        group.append(self.checkbox)
        
        self.label = QLabel(title)
        self.label.setStyleSheet("background-color: black;font: 600 15pt 'Segoe UI';border:none;")
        self.label.setMaximumHeight(70)
        self.label.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel("")
        self.image_label.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;border: none; font: 600 15pt 'Segoe UI';")
        
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.checkbox)
        h_layout.addWidget(self.label)
        
        self.layout.addLayout(h_layout)
        self.layout.addWidget(self.image_label)