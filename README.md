# SM23-AI-ROS-03
Teachable Machine Model

Train a computer to recognize your own images, sounds, & poses.

A fast, easy way to create machine learning models for your sites, apps, and more, no expertise or coding required.

What is Teachable Machine?

Teachable Machine is a web-based tool that makes creating machine learning models fast, easy, and accessible to everyone.

# How do I use it?

+ Gather
Gather and group your examples into classes, or categories, that you want the computer to learn (Gather samples that reads Train Model)

+ Train
Train your model, then instantly test it out to see whether it can correctly classify new examples.Video: Train your model of a desktop and mobile web browser containing a sample teachable machine project

+ Export
Export your model for your projects: sites, apps, and more. You can download your model or host it online.

# What can I use to teach it?
Teachable Machine is flexible – use files or capture examples live. It’s respectful of the way you work. 

You can even choose to use it entirely on-device, without any webcam or microphone data leaving your computer.

+ Images
Teach a model to classify images using files or your webcam.
+ Sounds
Teach a model to classify audio by recording short sound samples.
+ Poses
Teach a model to classify body positions using files or striking poses in your webcam.

The models you make with Teachable Machine are real TensorFlow.js models that work anywhere javascript runs, so they play nice with tools like Glitch, P5.js, Node.js & more.
Plus, export to different formats to use your models elsewhere, like Coral, Arduino & more.

# Using the Model
After the model is trained, you can use it for many different purposes.

This example predicts input image, given 3 input images for class named nature, and a class named beach, then calls a function to give a percentage prediction of each class in the input test image based on the given classes:

+ percentage prediction result:
  = First image prediction result:
<img width="960" alt="2023-08-10 (3)" src="https://github.com/Naif-Al-Ajlani/SM23-AI-ROS-03/assets/98528261/dac9f931-1ef0-455b-baf9-c1d6a2d86b15">
  = Seconed image prediction result:
<img width="960" alt="2023-08-10 (4)" src="https://github.com/Naif-Al-Ajlani/SM23-AI-ROS-03/assets/98528261/40bd2cbb-9fe9-47a6-98be-3dc940209b77">
  = Third image prediction result:
<img width="960" alt="2023-08-10 (5)" src="https://github.com/Naif-Al-Ajlani/SM23-AI-ROS-03/assets/98528261/f3abd51a-64ca-4f99-b671-b6cfa13d60a6">

# Open CV Keras code used to train the model:

+ Note(TensorFlow is required for Keras to work)
 ```
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

```

# Resorces:
+ Machine model training site: https://teachablemachine.withgoogle.com
  
+ Teachable Machine Tutorial: Snap, Clap, Whistle: https://medium.com/@warronbebster/teachable-machine-tutorial-snap-clap-whistle-4212fd7f3555
  
+ Teachable Machine Tutorial: Head Tilt: https://medium.com/@warronbebster/teachable-machine-tutorial-head-tilt-f4f6116f491
