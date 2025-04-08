import cv2
import torch
import numpy as np

from collections import Counter, deque
from nets.PIPNetConverter import Converter


class LandmarksDetectorPT:
    def __init__(self, config, pt_path=None, input_size=256, resolution=None, device="cpu", half=True) -> None:
        self.half = half
        self.device = device
        self.config = config
        self.pt_path = pt_path
        self.resolution = resolution
        self.input_size = input_size

        # Normalization parameters (ImageNet)
        self.mean = np.array([123.675, 116.28, 103.53], np.float64).reshape(1, -1)
        self.std = np.array([58.395, 57.12, 57.375], np.float64).reshape(1, -1)

        # Buffer parameters
        self.maxlen = 1
        self.most_frequent_drowsiness = None
        self.drowsiness_deque = deque(maxlen=self.maxlen)

        # Zooming parameters
        self.scale = 1.2            # (default=0), Apply extra padding for detected face

        # Thresholds parameters (conf_threshold is used together with maxConfMask)
        self.eye_dist_thresh = 2.5  # (default=5.5),    Set eye distance threshold (pixels)

    def load_model(self, model=None):
        if model is None:
            model = torch.load(self.pt_path, map_location=self.device)['model']
            
        self.model = model.float().eval()
        if self.half and self.device == "cuda":
            self.model.half()
        
        # Create output Converter for post-processing
        self.converter = Converter()
        return self
    
    def eye_aspect_ratio(eye):
        """ Calculate the eye aspect ratio """
        A = np.linalg.norm(eye[1] - eye[5])      # vertical-left
        B = np.linalg.norm(eye[2] - eye[4])      # vertical-right
        C = np.linalg.norm(eye[0] - eye[3])      # horizontal
        ear = (A + B) / (2.0 * C)
        return ear

    def eye_opening_distance(self, eye):
        """ Calculate the eye opening distance """
        A = np.linalg.norm(eye[1] - eye[5])      # vertical-left
        B = np.linalg.norm(eye[2] - eye[4])      # vertical-right
        dist = (A + B) / 2.0
        return dist

    def postprocess(self, landmark_2D_12, mask, avg_dist=False):
        """ Assess eye landmarks to predict drowsiness """
        # Get left and right eyes landmarks with their masks
        leftEye, leftMask = landmark_2D_12[6:], mask[6:]
        rightEye, rightMask = landmark_2D_12[:6], mask[:6]

        # Calculate eye opening distance
        if avg_dist:
            leftEAR = self.eye_opening_distance(leftEye)
            rightEAR = self.eye_opening_distance(rightEye)
            dist = (leftEAR + rightEAR) / 2.0
        else:
            dist = self.eye_opening_distance(rightEye)
        
        # Check if there are at least N uncertain landmarks
        if sum(rightMask) > self.maxConfMask:
            dist = 0.0
        
        # Making decision based on the eye aspect ratio threshold
        if dist < self.eye_dist_thresh:
            self.drowsiness_deque.append("Drowsy")
        else:
            self.drowsiness_deque.append("Awake")

        # Get the most frequent result
        if self.drowsiness_deque:
            most_frequent_drowsiness = Counter(self.drowsiness_deque).most_common(n=1)[0][0]
        else:
            most_frequent_drowsiness = False
        return most_frequent_drowsiness

    def visualize(self, frame, landmark_2D_12, bbox, x_min, y_min, x_max, y_max):
        # Get left and right eyes landmarks
        leftEye = landmark_2D_12[6:]
        rightEye = landmark_2D_12[:6]

        # Form convex hulls for both eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # Round the form convex hulls for both eyes
        leftEyeHull = np.round(leftEyeHull).astype(np.int32)
        rightEyeHull = np.round(rightEyeHull).astype(np.int32)

        if self.most_frequent_drowsiness == "Drowsy":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        # Plot facial landmarks
        lmk = np.round(landmark_2D_12).astype(np.int32)
        for i, (x, y) in enumerate(lmk):
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)

        # Draw eyes contours and drowsiness result
        # cv2.drawContours(frame, [leftEyeHull], -1, color, 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, color, 1)
        cv2.putText(frame, self.most_frequent_drowsiness, (rightEyeHull[0][0][0], rightEyeHull[0][0][1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw original and preprocessed face bboxes
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    def test(self, sample, target):
        """ Test PT Model Performance """
        sample = sample.to(self.device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output = self.model(sample)
        
        output = output.cpu().numpy()
        return self.converter(output, target)
    
    def predict(self, frame, bbox, verbose=False):
        """ Demo: ONNX Inference Pipeline to detect facial landmarks """
        # Pre-process the input
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]

        box_w = x_max - x_min
        box_h = y_max - y_min

        # Remove a part of top area for alignment, see paper for details
        x_min -= int(box_w * (self.scale - 1) / 2)
        y_min += int(box_h * (self.scale - 1) / 2)
        x_max += int(box_w * (self.scale - 1) / 2)
        y_max += int(box_h * (self.scale - 1) / 2)

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)
        x_max = min(x_max, self.resolution[0] - 1)
        y_max = min(y_max, self.resolution[1] - 1)

        box_w = x_max - x_min + 1
        box_h = y_max - y_min + 1

        image = frame[y_min:y_max, x_min:x_max, :]
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = image.astype(np.float32)

        cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)   # inplace
        cv2.subtract(image, self.mean, image)           # inplace
        cv2.multiply(image, 1 / self.std, image)        # inplace

        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).unsqueeze(0)

        if self.device == "cuda":
            image = image.cuda()

        if self.half and self.device == "cuda":
            image = image.half()

        # Get prediction
        with torch.amp.autocast("cuda"):
            output = self.model(image)

        # Convert to landmarks
        landmark_2D_12, mask = self.converter.postprocess_output_data(output)
        landmark_2D_12[:, 0] = landmark_2D_12[:, 0] * box_w + x_min
        landmark_2D_12[:, 1] = landmark_2D_12[:, 1] * box_h + y_min

        # Assess eye landmarks to make decision
        self.most_frequent_drowsiness = self.postprocess(landmark_2D_12, mask, avg_dist=False)

        # For testing visualizsation
        if verbose:
            self.visualize(frame, landmark_2D_12, bbox, x_min, y_min, x_max, y_max)
        return landmark_2D_12, self.most_frequent_drowsiness
