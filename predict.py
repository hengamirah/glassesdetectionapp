import os
import cv2
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import mlflow
from src.config.config import load_config
from src.logging.logger import setup_logging
from src.processing.detector import GlassesDetectionPipeline

def process_image(pipeline, image_path, output_dir):
    """Process a single image using the pipeline"""
     # Resolve paths dynamically
    output_dir = os.path.join(os.getcwd(), output_dir)
    image_path = os.path.join(os.getcwd(), image_path)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to read image: {image_path}")
        return None, {}
    
    # Process image
    processed_image, metrics = pipeline.process_frame(image)
    
    # Save output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        cv2.imwrite(output_path, processed_image)
        logger.info(f"Saved processed image to: {output_path}")
        
        if metrics.get('glasses_detected', False):
            logger.info(f"Glasses detected in {filename} with confidence: {metrics.get('confidence', 0):.4f}")
    
    return processed_image, metrics

def process_batch(pipeline, input_dir, output_dir, mlflow_logging=True):
    """Process all images in a directory"""
    input_path = Path(input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [str(p) for p in input_path.glob('**/*') if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        logger.warning(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    if mlflow_logging:
        mlflow_config = load_config('configs/mlflow.yaml')
        mlflow.set_tracking_uri(mlflow_config['tracking']['uri'])
        mlflow.set_experiment(f"{mlflow_config['tracking']['experiment_name']}-inference")
        
        with mlflow.start_run(run_name=f"batch_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("input_dir", input_dir)
            mlflow.log_param("output_dir", output_dir)
            mlflow.log_param("image_count", len(image_paths))
            
            detection_results = {}
            
            for idx, image_path in enumerate(image_paths):
                logger.info(f"Processing image {idx+1}/{len(image_paths)}: {image_path}")
                _, metrics = process_image(pipeline, image_path, output_dir)
                
                # Log detection results
                if metrics.get('glasses_detected', False):
                    detection_results[os.path.basename(image_path)] = metrics.get('confidence', 0)
            
            # Log summary metrics
            detected_count = sum(1 for m in detection_results.values() if m > 0)
            mlflow.log_metrics({
                "images_processed": len(image_paths),
                "glasses_detected_count": detected_count,
                "detection_rate": detected_count / len(image_paths) if image_paths else 0
            })
            
            # Log detailed results
            if detection_results:
                results_path = os.path.join(output_dir, "detection_results.csv")
                with open(results_path, 'w') as f:
                    f.write("image,glasses_detected,confidence\n")
                    for img_path in image_paths:
                        img_name = os.path.basename(img_path)
                        confidence = detection_results.get(img_name, 0)
                        detected = confidence > 0
                        f.write(f"{img_name},{detected},{confidence:.4f}\n")
                
                mlflow.log_artifact(results_path)
    else:
        for idx, image_path in enumerate(image_paths):
            logger.info(f"Processing image {idx+1}/{len(image_paths)}: {image_path}")
            process_image(pipeline, image_path, output_dir)

def run_live_inference(config_path):
    """Run live inference using camera"""
    global logger
    
    # Load configuration
    config_path = os.path.join(os.getcwd(), config_path)
    config = load_config(config_path)  # Load the configuration file

    # Setup logging
    logger = setup_logging(config['logging'])
    
    # Initialize pipeline
    try:
        logger.info("Initializing glasses detection pipeline")
        pipeline = GlassesDetectionPipeline(config)
        
        logger.info("Starting live inference")
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

def run_batch_inference(config_path, input_dir, output_dir):
    """Run batch inference on a directory of images"""
    global logger
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config['logging'])
    
    # Initialize pipeline
    try:
        logger.info("Initializing glasses detection pipeline")
        pipeline = GlassesDetectionPipeline(config)
        
        logger.info(f"Starting batch inference on {input_dir}")
        process_batch(pipeline, input_dir, output_dir)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run glasses detection inference")
    parser.add_argument('--config', default='configs/app_config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', choices=['live', 'batch'], default='live', help='Inference mode: live (camera) or batch (images)')
    parser.add_argument('--input', help='Input directory for batch mode')
    parser.add_argument('--output', help='Output directory for processed images')
    
    args = parser.parse_args()
    
    if args.mode == 'live':
        run_live_inference(args.config)
    elif args.mode == 'batch':
        if not args.input:
            print("Error: Input directory required for batch mode")
            parser.print_help()
            exit(1)
        run_batch_inference(args.config, args.input, args.output or 'output')