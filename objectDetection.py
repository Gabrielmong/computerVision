
import cv2
import numpy as np
import time
import os
import tkinter as tk
from tkinter import filedialog


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
            cv2.imshow("Output, press Q to exit", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if cv2.getWindowProperty("Output, press Q to exit", cv2.WND_PROP_VISIBLE) < 1:
                break

            (success, frame) = cap.read()

        cv2.destroyAllWindows()

    def classifyImage(self, imagePath):
        image = cv2.imread(imagePath)
        image = self.processFrame(image)

        cv2.imshow("Output, press Q to exit", image)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if cv2.getWindowProperty("Output, press Q to exit", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()


def main():
    os.system("cls")
    print("Object Detection using OpenCV")
    configPath = os.path.join(
        "model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(None, configPath, modelPath, classesPath)

    root = tk.Tk()
    root.title("Object Detection")
    root.geometry("300x70")
    root.eval('tk::PlaceWindow . center')

    def openVideo():
        root.destroy()

        root2 = tk.Tk()
        root2.title("Video")
        root2.geometry("300x70")
        root2.eval('tk::PlaceWindow . center')

        def openCamera():
            detector = Detector(0, configPath, modelPath, classesPath)
            detector.onVideo()

        def openFile():
            root2.filename = filedialog.askopenfilename(
                initialdir="/", title="Select file", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
            detector = Detector(root2.filename,
                                configPath, modelPath, classesPath)
            detector.onVideo()

        def backToMain():
            root2.destroy()
            main()

        cameraButton = tk.Button(root2, text="Camera", command=openCamera)
        cameraButton.pack()

        fileButton = tk.Button(root2, text="File", command=openFile)
        fileButton.pack()

        root2.protocol("WM_DELETE_WINDOW", backToMain)

        root2.mainloop()

    def openImage():
        root.filename = filedialog.askopenfilename(
            initialdir="/", title="Select file", filetypes=(("jpeg files", "*.jpg;*.jpeg"), ("all files", "*.*")))
        detector.classifyImage(root.filename)

    videoButton = tk.Button(root, text="Video", command=openVideo)
    videoButton.pack()

    imageButton = tk.Button(root, text="Image", command=openImage)
    imageButton.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
