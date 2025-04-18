from PyQt5.QtWidgets import QWidget, QVBoxLayout, QFrame, QLabel, QPushButton, QRadioButton, QButtonGroup, QHBoxLayout, QSizePolicy, QSpacerItem
from PyQt5.QtCore import Qt

class FilterTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(800)
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(20)

        # Noise frame
        noise_frame = QFrame()
        noise_frame.setMinimumHeight(300)
        noise_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        noise_frame.setStyleSheet("border: 2px solid rgb(40, 57, 153); border-radius: 10px;padding: 10px;")
        noise_layout = QVBoxLayout(noise_frame)
        noise_layout.setAlignment(Qt.AlignTop)
        noise_layout.setSpacing(20)

        noise_label = QLabel("Noise")
        noise_label.setStyleSheet("border: none; font: 600 15pt 'Segoe UI';")
        noise_label.setAlignment(Qt.AlignCenter)
        noise_label.setMaximumHeight(70)
        noise_layout.addWidget(noise_label)

        gaussian_noise_button = QPushButton("Gaussian Noise")
        gaussian_noise_button.setMinimumHeight(50)
        gaussian_noise_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        uniform_noise_button = QPushButton("Uniform Noise")
        uniform_noise_button.setMinimumHeight(50)
        uniform_noise_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        salt_pepper_noise_button = QPushButton("Salt & Pepper Noise")
        salt_pepper_noise_button.setMinimumHeight(50)
        salt_pepper_noise_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")

        noise_layout.addWidget(gaussian_noise_button)
        noise_layout.addWidget(uniform_noise_button)
        noise_layout.addWidget(salt_pepper_noise_button)

        filters_frame = QFrame()
        filters_frame.setMinimumHeight(300)
        filters_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        filters_frame.setStyleSheet("border: 2px solid rgb(40, 57, 153); border-radius: 10px; padding: 10px;")
        filters_layout = QVBoxLayout(filters_frame)
        filters_layout.setAlignment(Qt.AlignTop)
        filters_layout.setSpacing(20)

        filters_label = QLabel("Filters")
        filters_label.setStyleSheet("border: none; font: 600 15pt 'Segoe UI';")
        filters_label.setAlignment(Qt.AlignCenter)
        filters_label.setMaximumHeight(70)
        filters_layout.addWidget(filters_label)

        radio_frame = QFrame()
        radio_frame.setStyleSheet("border: none;margin: 0px; padding: 0px;")
        radio_layout = QHBoxLayout(radio_frame)
        radio_layout.setAlignment(Qt.AlignCenter)
        radio_layout.setSpacing(20)

        radio_button_group = QButtonGroup()
        radio_3x3 = QRadioButton("3x3")
        radio_3x3.setStyleSheet("color: white; font: 600 12pt 'Segoe UI';")
        radio_5x5 = QRadioButton("5x5")
        radio_5x5.setStyleSheet("color: white; font: 600 12pt 'Segoe UI';")
        radio_button_group.addButton(radio_3x3)
        radio_button_group.addButton(radio_5x5)

        radio_layout.addWidget(radio_3x3)
        radio_layout.addWidget(radio_5x5)

        filters_layout.addWidget(radio_frame)

        gaussian_filter_button = QPushButton("Gaussian Filter")
        gaussian_filter_button.setMinimumHeight(50)
        gaussian_filter_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        median_filter_button = QPushButton("Median Filter")
        median_filter_button.setMinimumHeight(50)
        median_filter_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        average_filter_button = QPushButton("Average Filter")
        average_filter_button.setMinimumHeight(50)
        average_filter_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")

        filters_layout.addWidget(gaussian_filter_button)
        filters_layout.addWidget(median_filter_button)
        filters_layout.addWidget(average_filter_button)

        self.layout.addWidget(noise_frame)
        self.layout.addWidget(filters_frame)
        self.layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))