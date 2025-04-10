import cv2
import mlflow
from ultralytics import YOLO
from typing import Tuple, Dict, Any
from datetime import datetime
import os
import numpy as np
from src.config.config import load_config

class GlassesDetectionPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = self._load_model()
        self.class_names = self.model.names
        self.glasses_class_id = self._get_glasses_class_id()
        mlflow_config = load_config('configs/mlflow.yaml')
        mlflow.set_tracking_uri(mlflow_config['tracking']['uri'])
        mlflow.set_experiment(f"{mlflow_config['tracking']['experiment_name']}-inference")
        
    def _load_model(self):
        """Load YOLO model with error handling"""
        try:
            model = YOLO(self.config['inference']['path'])
            return model
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _get_glasses_class_id(self) -> int:
        """Find glasses class ID in model"""
        for class_id, class_name in self.class_names.items():
            if 'glass' in class_name.lower():
                return class_id
        raise ValueError("Glasses class not found in model")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Process single frame for glasses detection"""
        results = self.model(frame, verbose=False)
        metrics = {'glasses_detected': False, 'confidence': 0.0}
        
        annotated_frame = frame.copy()
        for result in results:
            for box in result.boxes:
                if box.cls == self.glasses_class_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    annotated_frame[y1:y2, x1:x2] = 0  # Black out glasses
                    metrics.update({
                        'glasses_detected': True,
                        'confidence': float(box.conf),
                        'bbox': [x1, y1, x2, y2]
                    })
                    confidence = float(box.conf)
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    # Draw bounding box and label
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        
        return annotated_frame, metrics
    # def process_frame(self, frame):
    # # Run the YOLO model on the frame
    #     results = self.model(frame)

    #     # Iterate over detections and draw bounding boxes
    #     for result in results[0].boxes:
    #         x1, y1, x2, y2 = map(int, result.xyxy[0])
    #         class_id = int(result.cls[0])
    #         confidence = result.conf[0]
    #         class_name = self.model.names[class_id]

    #         # Draw bounding box and label
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         label = f"{class_name} {confidence:.2f}"
    #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     return frame

    def run(self):
        """Main pipeline execution"""
        cap = cv2.VideoCapture(self.config['camera']['source'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])
        #set mlflow experiment
        #mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        #mlflow.set_experiment(self.config['mlflow']['experiment_name'])

        with mlflow.start_run():
            self._log_parameters()
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame, metrics = self.process_frame(frame)
                self._log_metrics(metrics, frame_count)

                #processed_frame= self.process_frame(frame)
                cv2.imshow('Glasses Detection', processed_frame)
                self._save_output(processed_frame, frame_count)
                
                if cv2.waitKey(1) == ord('q'):
                    break
                
                frame_count += 1
            
            cap.release()
            cv2.destroyAllWindows()
    
    def _log_parameters(self):
        """Log parameters to MLflow"""
        mlflow.log_params({
            'model': os.path.basename(self.config['model']['path']),
            'confidence_threshold': self.config['model']['confidence_threshold'],
            'resolution': f"{self.config['camera']['width']}x{self.config['camera']['height']}"
        })
    
    def _log_metrics(self, metrics: dict, frame_count: int):
        """Log metrics to MLflow"""
        mlflow.log_metrics({
            'glasses_detected': int(metrics['glasses_detected']),
            'confidence': metrics.get('confidence', 0),
        }, step=frame_count)
    
    def _save_output(self, frame: np.ndarray, frame_count: int):
        """Save output frames periodically"""
        if frame_count % self.config['output']['save_interval'] == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.config['output']['save_dir'],
                f"frame_{timestamp}_{frame_count}.jpg"
            )
            cv2.imwrite(output_path, frame)
            #mlflow.log_artifact(output_path)