import yaml
from typing import Dict, Any
import os

def load_config(config_path: str = 'configs/app_config.yaml') -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    # Ensure output directory exists
        if 'output' in config and 'save_dir' in config['output']:
            os.makedirs(config['output']['save_dir'], exist_ok=True)
            
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {e}")