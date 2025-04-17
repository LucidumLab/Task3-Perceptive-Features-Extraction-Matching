import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QTabWidget, QLabel, QPushButton, QCheckBox, QFileDialog, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from ImageFrame import ImageFrame
from FilterTab import FilterTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Processing App")
        self.setGeometry(100, 100, 1800, 800)
        self.setStyleSheet("background-color: rgb(4, 3, 26); color: white;")
        self.current_image = None
        self.current_output = None

        main_layout = QHBoxLayout()

        ################################ Left Frame ################################
        
        left_frame = QFrame()
        left_frame.setMinimumSize(300, 600)
        left_frame.setMaximumWidth(500)
        left_frame.setStyleSheet("background-color: rgb(4, 3, 26); color: white;")
        left_layout = QVBoxLayout(left_frame)

        tab_widget = QTabWidget()
        tab_widget.setStyleSheet("""
            QTabWidget::pane { /* The tab widget frame */
                border: none;
                background-color: rgb(4, 3, 26);
            }
            QTabBar::tab {
                background:  rgb(4, 3, 26);
                color: white;
                padding: 10px;
                border: 1px solid rgb(4, 3, 26);
                border-radius: 5px;
                font: 600 10pt 'Segoe UI';
                width: 70px;
                height: 25px;
            }
            QTabBar::tab:selected {
                background: rgb(40, 57, 153);
                color: white;
            }
        """)

        filter_tab = FilterTab()
        histogram_tab = QWidget()
        histogram_layout = QVBoxLayout(histogram_tab)
        histogram_label = QLabel("Histogram Page")
        histogram_layout.addWidget(histogram_label)

        tab_widget.addTab(filter_tab, "Filter")
        tab_widget.addTab(histogram_tab, "Histogram")
        tab_widget.addTab(QFrame(), "Edges")
        tab_widget.addTab(QFrame(), "FFT filter")

        left_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum))
        left_layout.addWidget(tab_widget)
        main_layout.addWidget(left_frame)

        ################################ Right Frame ################################

        right_frame = QFrame()
        right_frame.setMinimumSize(800, 600)
        right_frame.setStyleSheet("background-color: rgb(4, 3, 26); color: white;")
        right_layout = QVBoxLayout(right_frame)

        ################################ Input Frames ################################

        input_output_frame = QFrame()
        input_output_layout = QHBoxLayout(input_output_frame)
        input_output_layout.setAlignment(Qt.AlignTop)
        input_output_layout.setSpacing(20)

        v_layout_input = QVBoxLayout()
        self.image_selection_group = []

        frame_1=QFrame()
        frame_1.setFixedSize(700, 430)
        frame_1.setStyleSheet("border: 2px solid rgb(40, 57, 153); border-radius: 10px; padding: 10px;background-color: black;")
        self.input_frame_1 = ImageFrame("Image 1", self.image_selection_group, True)
        frame_1.setLayout(self.input_frame_1.layout)

        frame_2=QFrame()
        frame_2.setFixedSize(700, 430)
        frame_2.setStyleSheet("border: 2px solid rgb(40, 57, 153); border-radius: 10px; padding: 10px;background-color: black;")
        self.input_frame_2 = ImageFrame("Image 2", self.image_selection_group)
        frame_2.setLayout(self.input_frame_2.layout)

        v_layout_input.addWidget(frame_1)
        v_layout_input.addWidget(frame_2)
        input_output_layout.addLayout(v_layout_input)

        v_layout_output = QVBoxLayout()
        self.plot_selection_group = []

        ################################ Output Frames ################################
        
        frame_3=QFrame()
        frame_3.setFixedSize(700, 430)
        frame_3.setStyleSheet("border: 2px solid rgb(40, 57, 153); border-radius: 10px; padding: 10px;background-color: black;")
        self.output_frame_1 = ImageFrame("Output 1", self.plot_selection_group, True)
        frame_3.setLayout(self.output_frame_1.layout)

        frame_4=QFrame()
        frame_4.setFixedSize(700, 430)
        frame_4.setStyleSheet("border: 2px solid rgb(40, 57, 153); border-radius: 10px; padding: 10px;background-color: black;")
        self.output_frame_2 = ImageFrame("Output 2", self.plot_selection_group)
        frame_4.setLayout(self.output_frame_2.layout)
        
        v_layout_output.addWidget(frame_3)
        v_layout_output.addWidget(frame_4)
        input_output_layout.addLayout(v_layout_output)

        control_frame = QFrame()
        control_frame.setMaximumHeight(60)
        control_layout = QHBoxLayout(control_frame)
        control_layout.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        spacer = QSpacerItem(50, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        control_layout.addItem(spacer)

        load_button = QPushButton("Load Image")
        load_button.setMinimumSize(100, 50)
        load_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        load_button.clicked.connect(self.load_image)

        clear_button = QPushButton("Clear Image")
        clear_button.setMinimumSize(100, 50)
        clear_button.setStyleSheet("border: none; background-color: rgb(40, 57, 153); font: 600 10pt 'Segoe UI'; border-radius: 5px;")
        clear_button.clicked.connect(self.clear_image)

        control_layout.addWidget(load_button)
        control_layout.addWidget(clear_button)

        right_layout.addWidget(control_frame)
        right_layout.addWidget(input_output_frame)
        main_layout.addWidget(right_frame)

        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.xpm *.jpg *.bmp *.gif *.jpeg)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            if self.input_frame_1.checkbox.isChecked():
                self.input_frame_1.image_label.setPixmap(pixmap.scaled(self.input_frame_1.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            if self.input_frame_2.checkbox.isChecked():
                self.input_frame_2.image_label.setPixmap(pixmap.scaled(self.input_frame_2.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

            if self.output_frame_1.checkbox.isChecked():
                self.output_frame_1.image_label.setPixmap(pixmap.scaled(self.output_frame_1.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            if self.output_frame_2.checkbox.isChecked():
                self.output_frame_2.image_label.setPixmap(pixmap.scaled(self.output_frame_2.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def clear_image(self):
        if self.output_frame_1.checkbox.isChecked():
            self.output_frame_1.image_label.clear()
        if self.output_frame_2.checkbox.isChecked():
            self.output_frame_2.image_label.clear()

        if self.input_frame_1.checkbox.isChecked():
            self.input_frame_1.image_label.clear()
        if self.input_frame_2.checkbox.isChecked():
            self.input_frame_2.image_label.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())