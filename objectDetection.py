
import cv2
import numpy as np
import time
import os
from tkinter import filedialog
import customtkinter as ctk
from PIL import Image
from customtkinter import CTkImage

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

root = ctk.CTk()
root.title("Object Detection")
root.resizable(False, False)
root.geometry("1600x900")

leftContainer = ctk.CTkFrame(root, width=180, height=900)
leftContainer.pack(padx=10, pady=10, side=ctk.LEFT,
                   fill=ctk.BOTH, expand=False)

rightContainer = ctk.CTkFrame(root, width=1000, height=900)
rightContainer.pack(padx=10, pady=10, side=ctk.RIGHT,
                    fill=ctk.BOTH, expand=True)

leftFrame = ctk.CTkFrame(leftContainer, width=180, height=900,
                         fg_color=('#3d3d3d'))
leftFrame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)
rightFrame = ctk.CTkFrame(rightContainer, width=1000, height=900,
                          fg_color=('#3d3d3d'))
rightFrame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True)

outputLabel = ctk.CTkLabel(rightFrame, text="", fg_color=('#3d3d3d'),
                           height=900, width=1000)
outputLabel.pack(side=ctk.TOP, fill=ctk.BOTH, expand=False)


class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.configPath, self.modelPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        self.classes.insert(0, 'Background')

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def processFrame(self, frame):
        height, width, _ = frame.shape

        aspectRatio = width / height

        if width > 800:
            newWidth = 800
            newHeight = newWidth / aspectRatio

            frame = cv2.resize(frame, (int(newWidth), int(newHeight)))

        classLabelIDs, confidences, bbox = self.net.detect(
            frame, confThreshold=0.5)

        bboxs = list(bbox)
        confidences = list(np.array(confidences).reshape(1, -1)[0])
        confidences = list(map(float, confidences))

        bboxIdx = cv2.dnn.NMSBoxes(
            bbox, confidences, score_threshold=0.5, nms_threshold=0.2)

        try:
            if len(bboxIdx != 0):
                for i in range(0, len(bboxIdx)):

                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(
                        classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classes[classLabelID]
                    classColor = [int(c) for c in self.colors[classLabelID]]

                    displayStr = '{}: {:.2f}'.format(
                        classLabel, classConfidence)

                    x, y, w, h = bbox

                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  color=classColor, thickness=1)

                    cv2.putText(frame, displayStr, (x, y - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, classColor, 2)

                    lineWidth = min(int(w * 0.3), int(h * 0.3))

                    cv2.line(frame, (x, y), (x + lineWidth, y),
                             classColor, thickness=3)
                    cv2.line(frame, (x, y), (x, y + lineWidth),
                             classColor, thickness=3)

                    cv2.line(frame, (x + w, y), (x + w - lineWidth, y),
                             classColor, thickness=3)
                    cv2.line(frame, (x + w, y), (x + w, y + lineWidth),
                             classColor, thickness=3)

                    cv2.line(frame, (x, y + h), (x + lineWidth, y + h),
                             classColor, thickness=3)
                    cv2.line(frame, (x, y + h), (x, y + h - lineWidth),
                             classColor, thickness=3)

                    cv2.line(frame, (x + w, y + h),
                             (x + w - lineWidth, y + h
                              ), classColor, thickness=3)
                    cv2.line(frame, (x + w, y + h), (x + w, y +
                                                     h - lineWidth), classColor, thickness=3)

            return frame

        except:
            return frame

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Cannot open video")
            return None

        (success, frame) = cap.read()

        cv2Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2Image = Image.fromarray(cv2Image)
        height, width = cv2Image.size
        ctkImage = CTkImage(cv2Image, size=(width, height))

        startTime = time.time()

        while success:
            currentTime = time.time()
            if (currentTime - startTime) > 0:
                fps = 1 / (currentTime - startTime)
            else:
                fps = 0

            startTime = currentTime + 0.00001

            frame = self.processFrame(frame)

            cv2.putText(frame, 'FPS: {:.2f}'.format(
                fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Press Q to exit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if cv2.getWindowProperty("Press Q to exit", cv2.WND_PROP_VISIBLE) < 1:
                break

            (success, frame) = cap.read()

        cv2.destroyAllWindows()

    def classifyImage(self, imagePath):
        image = cv2.imread(imagePath)
        image = self.processFrame(image)

        cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)

        height, width = img.size

        ctkImage = CTkImage(img, size=(width, height))

        outputLabel.image = ctkImage
        outputLabel.configure(image=ctkImage)


def main():
    def openVideo():
        outputLabel.image = None
        outputLabel.configure(
            image=None,
            text="Due to the limitations of the Tkinter library, and some performance issues, the video will not be displayed. Please check the pop-up window for the video.")

        def openCamera():
            detector = Detector(0, configPath, modelPath, classesPath)
            detector.onVideo()

        def openFile():
            root.filename = filedialog.askopenfilename(
                initialdir="/", title="Select file", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
            detector = Detector(root.filename, configPath,
                                modelPath, classesPath)

            detector.onVideo()

        def backToMain():
            for widget in leftFrame.winfo_children():
                widget.destroy()

            main()

        for widget in leftFrame.winfo_children():
            widget.destroy()

        label = ctk.CTkLabel(leftFrame, text="Video Classification")
        label.pack(pady=10, padx=50)

        button1 = ctk.CTkButton(
            leftFrame, text="Open Camera", command=openCamera)
        button1.pack(pady=10, padx=50)

        button2 = ctk.CTkButton(leftFrame, text="Open File", command=openFile)
        button2.pack(pady=10, padx=50)

        button3 = ctk.CTkButton(leftFrame, text="Back", command=backToMain)
        button3.pack(pady=10, padx=50)

    def openImage():
        outputLabel.configure(
            text="")

        root.filename = filedialog.askopenfilename(
            initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg;*.jpeg"), ("all files", "*.*")))
        detector.classifyImage(root.filename)

    os.system("cls")
    print("Object Detection using OpenCV")
    configPath = os.path.join(
        "model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(None, configPath, modelPath, classesPath)

    # left frame
    label = ctk.CTkLabel(leftFrame, text="Object Detection using OpenCV")
    label.pack(pady=10, padx=50)
    button1 = ctk.CTkButton(leftFrame, text="Open Video", command=openVideo)
    button1.pack(pady=10, padx=50)
    button2 = ctk.CTkButton(leftFrame, text="Open Image", command=openImage)
    button2.pack(pady=10, padx=50)
    button3 = ctk.CTkButton(leftFrame, text="Exit", command=root.destroy)
    button3.pack(pady=10, padx=50)

    root.mainloop()


if __name__ == "__main__":
    main()
