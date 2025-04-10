import argparse # Import necessary libraries for argument parsing and logging 
from src.config.config import load_config
from src.logging.logger import setup_logging
from src.processing.detector import GlassesDetectionPipeline

def main():
    # Load configuration
    config_path = 'configs/app_config.yaml'

    # Load configurations
    config = load_config(config_path)
    
    # Setup logging
    logger = setup_logging(config['logging'])
    
    # Initialize pipeline
    pipeline = GlassesDetectionPipeline(config)
    
    try:
        logger.info("Starting glasses detection pipeline")
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()