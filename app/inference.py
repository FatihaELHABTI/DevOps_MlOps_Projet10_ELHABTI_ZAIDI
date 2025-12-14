import cv2
import numpy as np
import time
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

class EdgeDetector:
    def __init__(self, model_path):
        print(f"Chargement du modèle : {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]
        self.is_quantized = (self.input_details[0]['dtype'] == np.uint8)

        # Mapping automatique
        self.boxes_idx, self.classes_idx, self.scores_idx = -1, -1, -1
        for i, detail in enumerate(self.output_details):
            shape = detail['shape']
            if len(shape) == 3 and shape[-1] == 4: self.boxes_idx = i
            elif len(shape) == 2:
                if 'score' in detail['name'].lower(): self.scores_idx = i
                elif 'class' in detail['name'].lower(): self.classes_idx = i
        
        # Fallback
        if self.boxes_idx == -1:
            sorted_outputs = sorted(self.output_details, key=lambda x: x['index'])
            for det in sorted_outputs:
                if len(det['shape']) == 3 and det['shape'][-1] == 4:
                    self.boxes_idx = self.output_details.index(det)
            leftover = [k for k, d in enumerate(self.output_details) if len(d['shape']) == 2 and k != self.boxes_idx]
            if len(leftover) >= 2:
                self.classes_idx, self.scores_idx = leftover[0], leftover[1]

    def predict(self, image):
        start_time = time.time()
        img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        if self.is_quantized:
            # Mode Int8 : On garde 0-255
            input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)
        else:
            # Mode Float32 : IL FAUT NORMALISER !
            # MobileNet attend souvent des valeurs entre -1 et 1
            input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
            input_data = input_data / 255.0  # <--- Juste diviser par 255

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        raw_boxes = self.interpreter.get_tensor(self.output_details[self.boxes_idx]['index'])
        raw_classes = self.interpreter.get_tensor(self.output_details[self.classes_idx]['index'])
        raw_scores = self.interpreter.get_tensor(self.output_details[self.scores_idx]['index'])

        # Normalisation
        if raw_scores.dtype == np.uint8: scores = raw_scores.astype(np.float32) / 255.0
        else: scores = raw_scores.astype(np.float32)

        if raw_boxes.dtype == np.uint8: boxes = raw_boxes.astype(np.float32) / 255.0
        else: boxes = raw_boxes.astype(np.float32)

        classes = raw_classes.astype(np.float32)

        # Flatten
        scores = scores.flatten()
        classes = classes.flatten()
        boxes = np.squeeze(boxes)
        boxes = np.clip(boxes, 0, 1)

        processing_time = (time.time() - start_time) * 1000
        return boxes, classes, scores, processing_time