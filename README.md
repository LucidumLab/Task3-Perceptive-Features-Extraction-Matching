# Perceptive Features Extraction & Matching Application

This application is a user-friendly application for digital image processing and feature extraction, built with Python and PyQt. It provides an interactive graphical interface where users can load images and experiment with a wide variety of perceptual feature extraction and matching techniques. All core algorithms—including SIFT, Harris, and Shi-Tomasi corner detection, descriptor extraction, feature matching, and visualization—are implemented from scratch, without relying on external libraries like OpenCV for the main processing logic.

---

## Features

- **SIFT Feature Extraction**
  - Detect and visualize SIFT keypoints and descriptors
  - Assign orientations and compute descriptors from scratch

- **Harris & Shi-Tomasi Corner Detection**
  - Detect corners using Harris and Shi-Tomasi algorithms
  - Visualize detected corners on images

- **Feature Matching**
  - Match features between two images using custom descriptor matching
  - Visualize correspondences with lines and overlays

- **Interactive GUI**
  - User-friendly PyQt interface for loading, processing, and saving images
  - Adjustable panels and responsive design for different screen sizes

_All core feature extraction and matching algorithms are implemented from scratch, without using OpenCV for the main processing steps in Python._

---

## Project Structure

```
Task3-Perceptive-Features-Extraction-Matching/
  ├── data/
  ├── docs/
  ├── notebooks/
  ├── resources/
  ├── sift/
  ├── models/
  ├── ui/
  ├── main.py
  ├── requirements.txt
  └── tests/
```

---

## Screenshots

<!-- Add screenshots or demo images here -->
*SIFT Feature Matching*

![image](https://github.com/user-attachments/assets/b3406a6f-85bd-497e-b83f-f7ba7e6402d1)

*Harris Corner Detection*

![image](https://github.com/user-attachments/assets/d2321a7b-fd78-4ffc-9d1c-c45473f33673)


---

## Getting Started

### Prerequisites

- Python 3.8+
- PyQt5
- NumPy
- OpenCV (for image I/O and display only)
- Matplotlib

### Installation

```sh
cd Task3-Perceptive-Features-Extraction-Matching
pip install -r requirements.txt
python main.py
```

---

## Contributors

* **Rawan Shoaib**: [GitHub Profile](https://github.com/RawanAhmed444)
* **Ahmed Aldeeb**: [GitHub Profile](https://github.com/AhmedXAlDeeb)
* **Ahmed Mahmoud**: [GitHub Profile](https://github.com/ahmed-226)
* **Eman Emad**: [GitHub Profile](https://github.com/Alyaaa16)

---

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [PyQt Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [Anatomy of the SIFT Method Paper](https://drive.google.com/file/d/1aooA1-s8PyVCkE6tOOlqWCm-Njgoj0cm/view?usp=sharing)

*You can also have a look at our [report]()*

---
