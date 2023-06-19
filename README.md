## Libraries needed:
- numpy
- tkinter
- customtkinter
- PIL
- cv2

## How to run:
- Run the file `objectDetection.py` in the terminal


## How to use:
- Theres 3 buttons on the left side of the screen
- Allows video and image input
- In the video input, you can choose to use the webcam or a video file


## How it works:
- The program uses the SSD MobileNet V3 model to detect objects
- Takes in the input and runs a frame by frame detection
- The program then draws a rectangle around the object and displays the name of the object
- Also displays the confidence level of the object