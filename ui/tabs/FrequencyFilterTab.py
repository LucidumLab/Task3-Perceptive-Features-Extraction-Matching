from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QComboBox, QSpinBox, 
    QPushButton, QLabel
)
from PyQt5.QtCore import Qt

class FrequencyFilterTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        frequency_filter_frame = QFrame()
        frequency_filter_frame.setObjectName("frequency_filter_frame")
        frequency_filter_layout = QVBoxLayout(frequency_filter_frame)
        frequency_filter_layout.setAlignment(Qt.AlignTop)

        self.freqType = QComboBox()
        self.freqType.addItems(["low_pass", "high_pass"])

        self.freqRadius = QSpinBox()
        self.freqRadius.setRange(1, 100)
        self.freqRadius.setValue(10)

        self.btn_freq_filter = QPushButton("Apply Frequency Filter")
        self.btn_freq_filter.clicked.connect(parent.apply_frequency_filter)

        self.freqLayout = QVBoxLayout()
        freq_type_layout = QHBoxLayout()
        freq_type_layout.addWidget(QLabel("Filter Type"))
        freq_type_layout.addWidget(self.freqType)
        self.freqLayout.addLayout(freq_type_layout)

        freq_radius_layout = QHBoxLayout()
        freq_radius_layout.addWidget(QLabel("Radius"))
        freq_radius_layout.addWidget(self.freqRadius)
        self.freqLayout.addLayout(freq_radius_layout)

        self.freqLayout.addWidget(self.btn_freq_filter)

        frequency_filter_layout.addLayout(self.freqLayout)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(frequency_filter_frame)
