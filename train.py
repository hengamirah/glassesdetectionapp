import os
import re
import yaml
from datetime import datetime
from ultralytics import YOLO
from dotenv import load_dotenv
import mlflow
from roboflow import Roboflow
from src.config.config import load_config
from src.logging.logger import setup_logging

# Load environment variables from .env file
load_dotenv()

def on_train_start(trainer):
    """Callback function when training starts"""
    logger.info(f"Training started with batch size: {trainer.args.batch}")
    logger.info(f"Training on device: {trainer.device}")

def on_fit_epoch_end(trainer):
    """Callback function at the end of each epoch"""
    logger.info(f"Completed epoch {trainer.epoch}/{trainer.epochs}")
    
    # Log metrics to MLflow
    metrics_dict = {f"{re.sub('[()]', '', k)}": float(v) for k, v in trainer.metrics.items()}
    mlflow.log_metrics(metrics=metrics_dict, step=trainer.epoch)

def train_model():
    """Train YOLO model for glasses detection with MLflow tracking"""
    global logger
    # Set default configuration file path
    config_path = 'configs/app_config.yaml'

    # Load configurations
    app_config = load_config(config_path)
    mlflow_config = load_config('configs/mlflow.yaml')
    
    # Setup logging
    logger = setup_logging(app_config['logging'])
    model_path = "src/yolo11l.pt"
    data_path = os.path.join(os.getcwd(), app_config['training']['data'])
    test_data_path = os.path.join(os.getcwd(), app_config['testing']['data'])
    output_dir = os.path.join(os.getcwd(), app_config['output']['save_dir'])


    

    # Configure MLflow
    mlflow.set_tracking_uri(mlflow_config['tracking']['uri'])
    mlflow.set_experiment(mlflow_config['tracking']['experiment_name'])
    
    # End any active run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()
    
    # Create run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{mlflow_config['tracking']['run_name']}_{timestamp}"
    
    logger.info(f"Starting training run: {run_name}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log training parameters
        mlflow.log_params({
            'model_base': model_path,
            'epochs': app_config['training']['epochs'],
            'batch_size': app_config['training']['batch_size'],
            'image_size': app_config['training']['img_size'],
            'dataset': data_path,
        })
        
        # Initialize model
        try:
            model = YOLO(model_path)
            model.add_callback("on_train_start", on_train_start)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
            
            # Train model
            results = model.train(
                data=app_config['training']['data'],
                epochs=app_config['training']['epochs'],
                imgsz=app_config['training']['img_size'],
                batch=app_config['training']['batch_size'],
                name=run_name,
                device=app_config['training'].get('device', 0),
                save=True,
                classes=[0]
            )
            
            logger.info("Training completed successfully")
            
            # Validate and log metrics
            logger.info("Running validation")
            metrics = model.val(data=test_data_path)
            
            mlflow.log_metrics({
                'test_mAP50': metrics.box.map50,
                'test_mAP50-95': metrics.box.map,
                'test_precision': metrics.box.mp,
                'test_recall': metrics.box.mr
            })
            
            # Save and log model
            best_model_path = os.path.join('runs', 'detect', run_name, 'weights', 'best.pt')
            
            if os.path.exists(best_model_path):
                logger.info(f"Logging best model: {best_model_path}")
                mlflow.log_artifact(best_model_path)
                
                # Update config with new model path
                app_config['inference']['path'] = best_model_path
                
                # Save updated config
                updated_config_path = f"configs/app_config_{timestamp}.yaml"
                with open(updated_config_path, 'w') as f:
                    yaml.dump(app_config, f)
                
                logger.info(f"Updated config saved to: {updated_config_path}")
                mlflow.log_artifact(updated_config_path)
            else:
                logger.error(f"Best model not found at: {best_model_path}")
        
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    return model, results

if __name__ == "__main__":

    trained_model, results = train_model()