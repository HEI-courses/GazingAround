from gaze_estimator import GazeEstimator

import cv2

from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtWidgets import QApplication, QMainWindow, QCheckBox, QPushButton, QRadioButton

import sys
import numpy as np

import cv2

gaze = GazeEstimator()
webcam = cv2.VideoCapture(0)

X_train = []
y_train = []


tmps = 0

prevh = []
prevv = []

app = QApplication(sys.argv)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.target = QRadioButton(self)
        self.target.move(QPoint(0,0))

        self.buttonStart = QPushButton(self)
        self.buttonStart.setFixedWidth(200)
        self.buttonStart.setText("Start calibration...")


        # Place checkboxes
        self.update_checkbox_positions()
        self.set_checkbox_styles()

        self.buttonStart.pressed.connect(self.onStart)

        # Connect window resize event to update positions
        self.resizeEvent = self.on_resize

    def update_checkbox_positions(self):
        self.buttonStart.move(int(self.width()/2)-100, int(self.height()/2))

    def onStart(self):
        self.buttonStart.setText("Follow the target")

        self.calibration_time_remaining = 30

        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.calibrate)
        self.calibration_timer.start(33)  # 33 ms interval (30fps)

    def calibrate(self):
        global tmps, gaze, X_train, y_train
        self.calibration_time_remaining -= 0.033

        features = None
        blink_detected = None
        if self.calibration_time_remaining <= 29:
            self.buttonStart.hide()
            # get the frame from webcam
            _, frame = webcam.read()
            # We send this frame to GazeTracking to analyze it
            features, blink_detected = gaze.extract_features(frame)

            #try to get the face
            while blink_detected or features is None:
                _, frame = webcam.read()
                features, blink_detected = gaze.extract_features(frame)
        
            tmps = tmps + 0.033
            x, y = self.lissajous_curve(tmps, self.width()*0.5, self.height()*0.5, 3, 2, 0.033)
            X_train.append(features)
            y_train.append([x, y])
            self.target.move(int(x), int(y))
        if self.calibration_time_remaining < 0:
            print("Done")
            self.calibration_timer.stop()
            gaze.train(X_train, y_train)
            self.tracking()
        
    def lissajous_curve(self, t, A, B, a, b, delta):
        x = A * np.sin(a * t + delta) + self.width() / 2
        y = B * np.sin(b * t) + self.height() / 2
        return x, y
    
    def tracking(self):

        self.hide()
        self.setStyleSheet("background: transparent;")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.showMaximized()
        self.target.setChecked(True)

        self.tracking_timer = QTimer(self)
        self.tracking_timer.timeout.connect(self.tracking_step)
        self.tracking_timer.start(33)  # 33 ms interval, ~30fps

    def tracking_step(self):
        global gaze

        _, frame = webcam.read()
        # We send this frame to GazeEstimator to analyze it
        features, blink_detected = gaze.extract_features(frame)

        #try to get the face
        if blink_detected or features is None:
            print("Stop blinking! (or be detectable pls)")
        else:
            posh, posv = gaze.predict([features])[0]

            prevh.append(posh)
            prevv.append(posv)

            if len(prevh) > 3:
                prevh.pop(0)
                prevv.pop(0)

            self.target.move(int(sum(prevh)/len(prevh)), int(sum(prevv)/len(prevv)))

    def on_resize(self, event):
        super().resizeEvent(event)
        self.update_checkbox_positions()
    
    def set_checkbox_styles(self):
        # Set styles for all checkboxes
        style = """
        QRadioButton::indicator {
            width: 20px;
            height: 20px;
            border-radius: 10px;  /* Makes the indicator circular */
            background-color: white;
            border: 2px solid white;
        }
        QRadioButton::indicator:hover {
            border: 2px solid gray;  /* Optional: Highlight border on hover */
        }
        QRadioButton::indicator:checked {
            background-color: transparent;
            border: 3px solid red;
        }
        """
        self.target.setStyleSheet(style)

window = MainWindow()
window.showMaximized() 

app.exec()
