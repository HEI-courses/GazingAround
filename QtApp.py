from gaze_estimator import GazeEstimator

import cv2

import active_window

from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtWidgets import QApplication, QMainWindow, QCheckBox, QPushButton, QRadioButton

import sys
import numpy as np

import datetime

import pandas as pd

import cv2

gaze = GazeEstimator()
webcam = cv2.VideoCapture(0)

X_train = []
y_train = []

kalman_array = []

tmps = 0

prevh = []
prevv = []

app = QApplication(sys.argv)

window_getter = active_window.WindowGetter() #Class used to get the active window. /!\ Works on linux but not with Wayland window manager (use X11) /!\

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.target = QRadioButton(self)
        self.target.move(QPoint(0,0))

        self.buttonStart = QPushButton(self)
        self.buttonStart.setFixedWidth(200)
        self.buttonStart.setText("Start calibration...")

        self.start_log_button = QPushButton(self)
        self.start_log_button.setFixedWidth(200)
        self.start_log_button.hide()

        self.log_active = False

        self.df = pd.DataFrame(columns=["time", "gaze", "app"])

        self.top_left = QCheckBox(self)
        self.top_left.move(50, 50)
        self.top_left.hide()
        self.top_right = QCheckBox(self)
        self.top_right.move(self.width()-50, 50)
        self.top_right.hide()
        self.bottom = QCheckBox(self)
        self.bottom.move(int((self.width()-50)/2), self.height()-50)
        self.bottom.hide()

        # Place checkboxes
        self.update_checkbox_positions()
        self.set_checkbox_styles()

        self.buttonStart.pressed.connect(self.onStart)

        # Connect window resize event to update positions
        self.resizeEvent = self.on_resize

    def update_checkbox_positions(self):
        self.buttonStart.move(int(self.width()/2)-100, int(self.height()/2))
        self.top_right.move(self.width()-50, 50)
        self.bottom.move(int((self.width()-50)/2), self.height()-50)
        self.start_log_button.move(int(self.width()/2 - self.start_log_button.width()/2), int(self.height()/2))


    def onStart(self):
        self.buttonStart.setText("Follow the target")

        self.calibration_time_remaining = 5

        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.calibrate)
        self.calibration_timer.start(33)  # 33 ms interval (30fps)

    def calibrate(self):
        global tmps, gaze, X_train, y_train
        self.calibration_time_remaining -= 0.033

        features = None
        blink_detected = None
        if self.calibration_time_remaining <= 4:
            self.buttonStart.hide()
            # get the frame from webcam
            _, frame = webcam.read()
            # We send this frame to GazeTracking to analyze it
            features, blink_detected = gaze.extract_features(frame)

            #try to get the face
            while blink_detected or features is None:
                _, frame = webcam.read()
                features, blink_detected = gaze.extract_features(frame)
        
            tmps = tmps + 0.015
            x, y = self.lissajous_curve(tmps, (self.width()-100)*0.5, (self.height()-100)*0.5, 3, 2, 0.033)
            X_train.append(features)
            y_train.append([x, y])
            self.target.move(int(x), int(y))

        if self.calibration_time_remaining < 0:
            print("Done")
            self.calibration_timer.stop()
            gaze.train(X_train, y_train)

            self.top_left.show()

            self.kalman = cv2.KalmanFilter(4, 2)
            self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            self.kalman.transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
            )
            self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1
            self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
            self.kalman.statePre = np.zeros((4, 1), np.float32)
            self.kalman.statePost = np.zeros((4, 1), np.float32)

            self.kalman_t = 1
            self.kalman_timer = QTimer(self)
            self.kalman_timer.timeout.connect(self.kalman_tune)
            self.kalman_timer.start(10)
        
    def kalman_tune(self):
        global gaze, kalman_array
        self.kalman_t -= 0.01
        if self.kalman_t < 0.5:
            _, frame = webcam.read()
            features, blink_detected = gaze.extract_features(frame)

            while blink_detected or features is None:
                _, frame = webcam.read()
                features, blink_detected = gaze.extract_features(frame)
            if not self.top_left.isChecked():
                pred = gaze.predict([features])[0]
                kalman_array.append([int(pred[0]),int(pred[1])])
                if self.kalman_t < 0:
                    self.top_left.setChecked(True)
                    self.kalman_t = 1
                    self.top_right.show()
                    print("kalman for p1 done")
            
            elif not self.top_right.isChecked():
                pred = gaze.predict([features])[0]
                kalman_array.append([int(pred[0]),int(pred[1])])
                if self.kalman_t < 0:
                    self.top_right.setChecked(True)
                    self.kalman_t = 1
                    self.bottom.show()
                    print("kalman for p2 done")
            
            elif not self.bottom.isChecked():
                pred = gaze.predict([features])[0]
                kalman_array.append([int(pred[0]),int(pred[1])])
                if self.kalman_t < 0:
                    self.bottom.setChecked(True)
                    self.kalman_timer.stop()

                    kalman_array = np.array(kalman_array)

                    kalman_var = np.var(kalman_array, axis=0)
                    kalman_var[kalman_var == 0] = 1e-4
                    print(kalman_var)
                    self.kalman.measurementNoiseCov = np.array(
                        [[kalman_var[0], 0], [0, kalman_var[1]]], dtype=np.float32
                    )
                    print("Kalman done...")
                    self.top_left.hide()
                    self.top_right.hide()
                    self.bottom.hide()

                    #Choose tracking (for verifications) or logging (For runtime)
                    # self.tracking()
                    self.running()
    
    def running(self):
        self.buttonStart.hide()
        self.target.hide()
        self.start_log_button.setText("Start recording gaze")
        self.start_log_button.show()
        self.start_log_button.pressed.connect(self.log_timer)

    def log_timer(self):
        if not self.log_active:
            self.start_log_button.setText("Stop recording...")
            self.log_active = True
            self.timer_log = QTimer(self)
            self.timer_log.timeout.connect(self.log)
            self.timer_log.start(100) #log gaze 10x by sec.
        else:
            self.timer_log.stop()
            print(self.df)

    def log(self):
        global gaze
        _, frame = webcam.read()

        features, blink_detected = gaze.extract_features(frame)

        while blink_detected or features is None:
                _, frame = webcam.read()
                features, blink_detected = gaze.extract_features(frame)
        
        res = gaze.predict([features])[0]
        pred = self.kalman.predict()

        x_pred, y_pred = int(pred[0]), int(pred[1])

        x_pred = max(0, min(x_pred, self.width() - 1))
        y_pred = max(0, min(y_pred, self.height() - 1))

        measurement = np.array([[np.float32(res[0])], [np.float32(res[1])]])

        if np.count_nonzero(self.kalman.statePre) == 0:
            self.kalman.statePre[:2] = measurement
            self.kalman.statePost[:2] = measurement

        self.kalman.correct(measurement)

        process = window_getter.get_activityname()['processname2']

        log_dict = {
            'time': datetime.datetime.now().isoformat(),
            'gaze': (x_pred, y_pred),
            'app': process
        }

        self.df.loc[len(self.df)] = [log_dict['time'], log_dict['gaze'], log_dict['app']]


    def lissajous_curve(self, t, A, B, a, b, delta):
        x = A * np.sin(a * t + delta) + self.width() / 2
        y = B * np.sin(b * t) + self.height() / 2
        return x, y
    
    def tracking(self):

        self.hide()

        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")
        
        self.showMaximized()
        self.repaint()  # Force a redraw
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
            res = gaze.predict([features])[0]
            pred = self.kalman.predict()
            x_pred, y_pred = int(pred[0]), int(pred[1])

            x_pred = max(0, min(x_pred, self.width() - 1))
            y_pred = max(0, min(y_pred, self.height() - 1))

            measurement = np.array([[np.float32(res[0])], [np.float32(res[1])]])
            if np.count_nonzero(self.kalman.statePre) == 0:
                self.kalman.statePre[:2] = measurement
                self.kalman.statePost[:2] = measurement
            self.kalman.correct(measurement)

            self.target.move(x_pred, y_pred)

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

        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border-radius: 10px;  /* Makes the indicator circular */
            background-color: red;
            border: 2px solid red;
        }
        QCheckBox::indicator:hover {
            border: 2px solid gray;  /* Optional: Highlight border on hover */
        }
        QCheckBox::indicator:checked {
            background-color: green;
            border: 3px solid green;
        }
        """
        self.target.setStyleSheet(style)
        self.top_left.setStyleSheet(style)
        self.top_right.setStyleSheet(style)
        self.bottom.setStyleSheet(style)

window = MainWindow()
window.showMaximized()

app.exec()
