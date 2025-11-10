import sys
import mlflow
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

## Editable Configuration 
MLFLOW_URI = "https://ard-mlflow.slac.stanford.edu"
EXPERIMENT_NAME = "LCLS_FEL_Surrogate"

# Metadata - EDIT FOR EACH MODEL
EMAIL = "gopikab@slac.stanford.edu"
REPO = "https://github.com/slaclab/LCLS_FEL_Surrogate"
BEAM_PATH = "cu_hxr"
DESCRIPTION = "Self-contained ML-based surrogate model of the LCLS FEL pulse intensity"
READY_TO_DEPLOY = "true"  # Set to "true" for production, "false" for dev
STAGE = "production"  # development, staging, or production

def register_model(model_name, model_directory, config_file=None):
    """Register model with all artifacts and metadata to MLflow"""
    
    model_dir = Path(model_directory)
    
    # Validate directory exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    logger.info(f"Registering model: {model_name}")
    logger.info(f"Model directory: {model_dir}")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name=model_name) as run:
        
        # 1. Log config file first (if provided)
        if config_file:
            config_path = Path(config_file)
            if config_path.exists():
                logger.info(f"Logging config file: {config_path}")
                mlflow.log_artifact(str(config_path))
            else:
                logger.warning(f"Config file not found: {config_path}")
        
        # 2. Log all artifacts from the model directory
        # This includes .pth pytorch files and any other resources
        logger.info("Logging artifacts from resources directory...")
        for item in model_dir.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(model_dir)
                artifact_path = str(rel_path.parent) if rel_path.parent != Path('.') else None
                logger.info(f"  → {rel_path}")
                mlflow.log_artifact(str(item), artifact_path=artifact_path)
        
        # 3. Set metadata tags
        logger.info("Setting tags...")
        tags = {
            "model_name": model_name,
            "email": EMAIL,
            "repo": REPO,
            "beam_path": BEAM_PATH,
            "description": DESCRIPTION,
            "ready_to_deploy": READY_TO_DEPLOY,
            "stage": STAGE,
        }
        
        for key, value in tags.items():
            mlflow.set_tag(key, value)
            logger.info(f"  Tag: {key} = {value}")
        
        logger.info(f"\n✓ Model registered successfully!")
        logger.info(f"  Run ID: {run.info.run_id}")
        logger.info(f"  Experiment: {EXPERIMENT_NAME}")
        logger.info(f"  Artifacts URI: {run.info.artifact_uri}")
        
        return run.info.run_id


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python register_model_to_mlflow.py <model_name> <model_directory> [config_file]")
        print("Example: python register_model_to_mlflow.py cu_hxr_v1 ./resources ./model_config.yaml")
        sys.exit(1)
    
    model_name = sys.argv[1]
    model_directory = sys.argv[2]
    config_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    register_model(model_name, model_directory, config_file)
