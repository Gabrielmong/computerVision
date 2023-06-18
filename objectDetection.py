import cv2
import numpy as np
import time
import os


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

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if not cap.isOpened():
            print("Cannot open video")
            exit()

        (success, frame) = cap.read()

        startTime = time.time()

        while success:
            currentTime = time.time()
            if (currentTime - startTime) > 0:
                fps = 1 / (currentTime - startTime)
            else:
                fps = 0

            startTime = currentTime + 0.00001

            classLabelIDs, confidences, bbox = self.net.detect(
                frame, confThreshold=0.5)

            bboxs = list(bbox)
            confidences = list(np.array(confidences).reshape(1, -1)[0])
            confidences = list(map(float, confidences))

            bboxIdx = cv2.dnn.NMSBoxes(
                bbox, confidences, score_threshold=0.5, nms_threshold=0.2)

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
                    cv2.putText(frame, displayStr, (x, y - 10),
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

                    cv2.line(frame, (x, y + h), (x + lineWidth,
                             y + h), classColor, thickness=3)
                    cv2.line(frame, (x, y + h), (x, y + h - lineWidth),
                             classColor, thickness=3)

                    cv2.line(frame, (x + w, y + h), (x + w -
                             lineWidth, y + h), classColor, thickness=3)
                    cv2.line(frame, (x + w, y + h), (x + w, y +
                             h - lineWidth), classColor, thickness=3)

            cv2.putText(frame, 'FPS: {:.2f}'.format(
                fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Output", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            (success, frame) = cap.read()

        cv2.destroyAllWindows()

    def classifyImage(self, imagePath):
        image = cv2.imread(imagePath)

        classLabelIDs, confidences, bbox = self.net.detect(
            image, confThreshold=0.5)

        bboxs = list(bbox)
        confidences = list(np.array(confidences).reshape(1, -1)[0])
        confidences = list(map(float, confidences))

        bboxIdx = cv2.dnn.NMSBoxes(
            bbox, confidences, score_threshold=0.5, nms_threshold=0.2)

        if len(bboxIdx != 0):
            for i in range(0, len(bboxIdx)):

                bbox = bboxs[np.squeeze(bboxIdx[i])]
                classConfidence = confidences[np.squeeze(bboxIdx[i])]
                classLabelID = np.squeeze(
                    classLabelIDs[np.squeeze(bboxIdx[i])])
                classLabel = self.classes[classLabelID]
                classColor = [int(c) for c in self.colors[classLabelID]]

                displayStr = '{}: {:.2f}'.format(classLabel, classConfidence)

                x, y, w, h = bbox

                cv2.rectangle(image, (x, y), (x + w, y + h),
                              color=classColor, thickness=1)
                cv2.putText(image, displayStr, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, classColor, 2)

                lineWidth = min(int(w * 0.3), int(h * 0.3))

                cv2.line(image, (x, y), (x + lineWidth, y),
                         classColor, thickness=3)
                cv2.line(image, (x, y), (x, y + lineWidth),
                         classColor, thickness=3)

                cv2.line(image, (x + w, y), (x + w - lineWidth, y),
                         classColor, thickness=3)
                cv2.line(image, (x + w, y), (x + w, y + lineWidth),
                         classColor, thickness=3)

                cv2.line(image, (x, y + h), (x + lineWidth, y + h),
                         classColor, thickness=3)
                cv2.line(image, (x, y + h), (x, y + h - lineWidth),
                         classColor, thickness=3)

                cv2.line(image, (x + w, y + h),
                         (x + w - lineWidth, y + h), classColor, thickness=3)
                cv2.line(image, (x + w, y + h), (x + w, y +
                         h - lineWidth), classColor, thickness=3)

        cv2.imshow("Output", image)
        cv2.waitKey(0)


def main():
    # 0 nikon camera
    # 3 logitech capture
    videoPath = 0

    imagePath = os.path.join("images", "city.jpg")

    configPath = os.path.join(
        "model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    detector = Detector(None, configPath, modelPath, classesPath)
    # detector.onVideo()

    detector.classifyImage(imagePath)


if __name__ == "__main__":
    main()
