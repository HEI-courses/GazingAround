# Gazing Around
This is a gaze tracking program with the goal of logging where the user gazes on different apps to create reports about gaze patterns and monitor time spent on different parts of an app.

# Credit
As of right now, this project uses PyQt6 for the UI part and the **[EyePy](https://github.com/ck-zhang/EyePy)** library by github user "ck-zhang" for eye and gaze tracking functionalites.

# Installation
Please install the required libraries:

```shell
pip install -r requirements.txt
pip install PyQt6
pip install pandas
```

# Running the app
To run the app, launch the QtApp.py script:

```shell
python QtApp.py
```

# Devlog
Reviewed multiple libraries (**[py-gaze](https://github.com/esdalmaijer/PyGaze)**, **[GazeTracking](https://github.com/antoinelame/GazeTracking)**, **[pyEyeTrack](https://github.com/algoasylum/pyEyeTrack)**, **[openFace](https://github.com/TadasBaltrusaitis/OpenFace)**). Implemented a solution with GazeTracking which worked poorly. While reaserching eye and gaze tracking, found the **[EyePy](https://github.com/ck-zhang/EyePy)** library and tested it. This library looked interesting because it seemed lightweight and effective.

Created the QtApp.py script, which allows the user to train the eye tracking model and displays a prediction for the gaze location on screen. This is a first shot and it's accuracy can be improved.