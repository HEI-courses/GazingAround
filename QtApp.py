from gaze_estimator import GazeEstimator

import cv2

import active_window

import math

from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtWidgets import QApplication, QMainWindow, QCheckBox, QPushButton, QRadioButton, QComboBox, QButtonGroup, QLineEdit, QLabel

import sys
import numpy as np

import datetime

import pandas as pd

import cv2

import create_reports

report_creator = create_reports.ReportGenerator()


gaze = GazeEstimator()
webcam = cv2.VideoCapture(0)

X_train = []
y_train = []

kalman_array = []

tmps = 0

acc = [0,1]

prevh = []
prevv = []

app = QApplication(sys.argv)

window_getter = active_window.WindowGetter() #Class used to get the active window. /!\ Works on linux but not with Wayland window manager (use X11) /!\

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.target = QRadioButton(self)
        self.target.move(QPoint(0,0))
        self.target.hide()

        self.intro_text = QLabel(self)
        self.intro_text.setText("Welcome to Gazing Around! \nThe goal of this app is to analyse gaze patterns using the laptop's \nwebcam. Please proceed to the calibration. First, you have to follow \nthe white dot. Then You have to fix the red dots until they become green. \nThe red dots will appear top left, top right and bottom in that order.")
        self.intro_text.setFixedSize(500, 100)
        self.intro_text.show()

        self.buttonStart = QPushButton(self)
        self.buttonStart.setFixedWidth(200)
        self.buttonStart.setText("Start calibration...")

        self.start_log_button = QPushButton(self)
        self.start_log_button.setFixedWidth(200)
        self.start_log_button.hide()

        self.log_active = False

        self.df = pd.DataFrame(columns=["time", "gazex", "gazey", "app"])

        self.top_left = QCheckBox(self)
        self.top_left.move(50, 50)
        self.top_left.hide()
        self.top_right = QCheckBox(self)
        self.top_right.move(self.width()-50, 50)
        self.top_right.hide()
        self.bottom = QCheckBox(self)
        self.bottom.move(int((self.width()-50)/2), self.height()-50)
        self.bottom.hide()

        index = 0
        arr = []
        i = 10
        while i > 0: # we use this to get all the available webcams
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                arr.append(index)
                cap.release()
            index += 1
            i -= 1
        
        self.cam_select = QComboBox(self)
        for i in arr:
            self.cam_select.addItem(f"Webcam {i}")
        self.cam_select.setFixedWidth(200)

        # Place checkboxes
        self.update_checkbox_positions()
        self.set_checkbox_styles()

        

        self.buttonStart.pressed.connect(self.onStart)

        # Connect window resize event to update positions
        self.resizeEvent = self.on_resize


    def update_checkbox_positions(self): #update the positions of widgets when window size chanages
        self.buttonStart.move(int(self.width()/2)-100, int(self.height()/2))
        self.top_right.move(self.width()-50, 50)
        self.bottom.move(int((self.width()-50)/2), self.height()-50)
        self.intro_text.move(int(self.width()/2 - 250), int(self.height()/2-150))
        self.cam_select.move(int(self.width()/2)-100, int(self.height()/2)+30)

    def onStart(self):
        self.intro_text.hide()
        self.cam_select.hide()
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
            self.target.show()
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
            #Lissajous curve user must follow to train the model
            x, y = self.lissajous_curve(tmps, (self.width()-100)*0.5, (self.height()-100)*0.5, 3, 2, 0.033)
            X_train.append(features)
            y_train.append([x, y])
            self.target.move(int(x), int(y))

        if self.calibration_time_remaining < 0:
            print("Done")
            self.target.hide()
            self.calibration_timer.stop()
            #Train model with data collected
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

            self.kalman_t = 6
            self.kalman_timer = QTimer(self)
            self.kalman_timer.timeout.connect(self.kalman_tune)
            self.kalman_timer.start(100)
    
    def dist(self, xp, yp, xi, yi):
        return(math.sqrt((xp-xi)**2+(yp-yi)**2))

    #kalman filter for fine tunning
    def kalman_tune(self):
        global gaze, kalman_array, acc
        self.target.setChecked(True)
        self.target.show()
        self.df.loc[0] = [0, self.width(), self.height(), "None"]
        self.kalman_t -= 0.1
        _, frame = webcam.read()
        features, blink_detected = gaze.extract_features(frame)

        while blink_detected or features is None:
            _, frame = webcam.read()
            features, blink_detected = gaze.extract_features(frame)
        pred = gaze.predict([features])[0]
        self.target.move(int(pred[0]), int(pred[1]))

        if self.kalman_t < 5:
            if not self.top_left.isChecked():
                pred = gaze.predict([features])[0]

                while self.dist(self.top_left.pos().x() + 10, self.top_left.pos().y() + 10, int(pred[0]), int(pred[1])) > 100:
                    _, frame = webcam.read()
                    features, blink_detected = gaze.extract_features(frame)
                    while blink_detected or features is None:
                        _, frame = webcam.read()
                        features, blink_detected = gaze.extract_features(frame)
                    pred = gaze.predict([features])[0]

                print("yeah")
                kalman_array.append([int(pred[0]),int(pred[1])])

                if self.kalman_t < 0:
                    self.top_left.setChecked(True)
                    self.kalman_t = 6
                    self.top_right.show()
                    print("kalman for p1 done")
            
            elif not self.top_right.isChecked():
                pred = gaze.predict([features])[0]

                while self.dist(self.top_right.pos().x() + 10, self.top_right.pos().y() + 10, int(pred[0]), int(pred[1])) > 100:
                    _, frame = webcam.read()
                    features, blink_detected = gaze.extract_features(frame)
                    while blink_detected or features is None:
                        _, frame = webcam.read()
                        features, blink_detected = gaze.extract_features(frame)
                    pred = gaze.predict([features])[0]

                kalman_array.append([int(pred[0]),int(pred[1])])

                if self.kalman_t < 0:
                    self.top_right.setChecked(True)
                    self.kalman_t = 6
                    self.bottom.show()
                    print("kalman for p2 done")
            
            elif not self.bottom.isChecked():
                pred = gaze.predict([features])[0]

                while self.dist(self.bottom.pos().x() + 10, self.bottom.pos().y() + 10, int(pred[0]), int(pred[1])) > 100:
                    _, frame = webcam.read()
                    features, blink_detected = gaze.extract_features(frame)
                    while blink_detected or features is None:
                        _, frame = webcam.read()
                        features, blink_detected = gaze.extract_features(frame)
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
        global acc
        right = 0
        tot = 0
        for i in acc:
            if i == 1:
                right += 1
            tot += 1
        est = (right/tot)*100
        self.buttonStart.hide()
        self.target.hide()

        self.calibrate_text = QLabel(self)
        self.calibrate_text.setText(f"Calibration done! Accuracy estimate: {int(est)}%")
        self.calibrate_text.setFixedWidth(300)
        self.calibrate_text.move(int(self.width()/2 - 150), int(self.height()/2-50))
        self.calibrate_text.show()
        self.start_log_button.setText("Start recording gaze")
        self.start_log_button.move(int(self.width()/2 - self.start_log_button.width()/2), int(self.height()/2))
        self.start_log_button.show()
        self.start_log_button.pressed.connect(self.log_timer)
        self.restart_calibrate = QPushButton(self)
        self.restart_calibrate.setText("Recalibrate")
        self.restart_calibrate.move(int(self.width()/2 - self.restart_calibrate.width()/2), int(self.height()/2+30))
        self.restart_calibrate.show()
        self.restart_calibrate.pressed.connect(self.calibrate2)
    
    def calibrate2(self):
        global kalman_array, acc
        self.calibrate_text.hide()
        self.start_log_button.hide()
        self.restart_calibrate.hide()
        self.top_left.setChecked(False)
        self.top_right.setChecked(False)
        self.bottom.setChecked(False)
        self.start_log_button.pressed.disconnect()

        kalman_array = []
        acc = []

        self.calibration_time_remaining = 30

        self.calibration_timer = QTimer(self)
        self.calibration_timer.timeout.connect(self.calibrate)
        self.calibration_timer.start(33)  # 33 ms interval (30fps)

    def log_timer(self):
        if not self.log_active:
            self.start_log_button.show()
            self.start_log_button.setText("Stop recording...")
            self.start_log_button.setFixedWidth(150)
            self.log_active = True
            self.timer_log = QTimer(self)
            self.timer_log.timeout.connect(self.log)
            self.timer_log.start(100) #log gaze 10x by sec.
            
            self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
            self.hide()
            self.setGeometry(200, 30, 160, 70)
            self.setFixedHeight(32)
            self.setFixedWidth(160)
            self.start_log_button.move(5,1)
            self.show()
            self.update()
            

        else:
            self.timer_log.stop()
            self.start_log_button.hide()
            self.log_active = False
            self.df.to_csv("log.csv")
            self.report_settings()

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

        #x_pred = max(0, min(x_pred, int(self.df.iloc[0]['gazex']) - 1)) #we dont lock the gaze into the screen anymore
        #y_pred = max(0, min(y_pred, int(self.df.iloc[0]['gazey']) - 1)) #to detect gazes outside of the screen

        measurement = np.array([[np.float32(res[0])], [np.float32(res[1])]])

        if np.count_nonzero(self.kalman.statePre) == 0:
            self.kalman.statePre[:2] = measurement
            self.kalman.statePost[:2] = measurement

        self.kalman.correct(measurement)

        process = window_getter.get_activityname()['processname2']

        log_dict = {
            'time': datetime.datetime.now(),
            'gazex': x_pred,
            'gazey': y_pred,
            'app': process
        }

        if log_dict['app'] is not None:

            self.df.loc[len(self.df)] = [log_dict['time'], log_dict['gazex'], log_dict['gazey'], log_dict['app']]

    def report_settings(self):
        self.button_group1 = QButtonGroup(self)
        self.button_group2 = QButtonGroup(self)

        self.all_apps = QRadioButton(self)
        self.all_apps.setFixedWidth(500)
        self.all_apps.setText("Generate reports for all apps")
        self.button_group1.addButton(self.all_apps)
        self.specific_app = QRadioButton(self)
        self.specific_app.setFixedWidth(500)
        self.specific_app.setText("Generate reports for specific app (Choose from menu)")
        self.button_group1.addButton(self.specific_app)
        
        self.choose_app = QComboBox(self)
        for i in self.df["app"].unique():
            self.choose_app.addItem(i)

        self.create_one = QRadioButton(self)
        self.create_one.setFixedWidth(500)
        self.create_one.setText("Create one report by app")
        self.button_group2.addButton(self.create_one)
        self.create_time = QRadioButton(self)
        self.create_time.setFixedWidth(500)
        self.create_time.setText("Create multiple reports by app, specify time interval")
        self.button_group2.addButton(self.create_time)
        self.time_box = QLineEdit(self)
        self.time_box.setFixedWidth(150)
        self.time_box.setText("Time interval (sec)")

        self.discard_button = QPushButton(self)
        self.discard_button.setFixedWidth(150)
        self.discard_button.setText("Discard report")
        self.generate_button = QPushButton(self)
        self.generate_button.setFixedWidth(150)
        self.generate_button.setText("Generate")

        self.all_apps.move(30, 40)
        self.specific_app.move(30, 80)
        self.choose_app.move(30, 120)

        self.create_one.move(30, 160)
        self.create_time.move(30, 200)
        self.time_box.move(30, 240)

        self.discard_button.move(30, 280)
        self.generate_button.move(200, 280)
        self.setFixedHeight(350)
        self.setFixedWidth(440)

        self.all_apps.show()
        self.specific_app.show()
        self.choose_app.show()
        self.create_one.show()
        self.create_time.show()
        self.time_box.show()
        self.discard_button.show()
        self.generate_button.show()
        self.hide()
        self.show()

        self.discard_button.pressed.connect(self.running_again)
        self.generate_button.pressed.connect(self.report_gen)

    def running_again(self):
        self.df = pd.DataFrame(columns=["time", "gazex", "gazey", "app"])
        for i in self.children():
            try:
                i.hide()
            except: pass

        self.start_log_button.setText("Start recording gaze")
        self.start_log_button.move(int(self.width()/2 - self.start_log_button.width()/2), int(self.height()/2))
        self.start_log_button.show()


    def report_gen(self):
        global report_creator
        print(self.df)
        if self.specific_app.isChecked():
            if self.create_one.isChecked():
                app = self.choose_app.currentText()
                report_creator.from_df(self.df, app)
            else:
                report_creator.from_df(self.df, app, int(self.time_box.text()))
        else:
            if self.create_one.isChecked():
                print("i")
                report_creator.from_df(self.df)
            else:
                report_creator.from_df(self.df, time_interval = int(self.time_box.text()))
                

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
        self.target.show()

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
