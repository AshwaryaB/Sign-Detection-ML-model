# Sign-Detection-ML-model
In this ML model, as the name suggests, I have developed a Hand Sign Detection model which as per American Hand Sign Conventions detects which letter is being shown via the web cam of the device.
Majorly I have used libraries like OpenCV, MediaPipe, os, scikit-learn and others

Instructions to run the model:<br>
First run the the python file data_collection.py which will collect the sample hand sign data from the webcam of the device (you have to pose the different hand sign in front of your webcam).<br>
Then run the landmark_collection.py to imprint landmarks of your hands in the frame.<br>
After this run the model_creation.py to create models (Random Forest, Linear Regression, K Nearest Neighbours, Decision tree). <br>
Finally running the testing.py would start the model and showing appropriate hand sign to the model, it will detect the corresponding letter.
