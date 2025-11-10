import mlflow
from lume_model.models import TorchModel

## Editable Configuration 
MLFLOW_URI = "https://ard-mlflow.slac.stanford.edu"
EXPERIMENT_NAME = "LCLS_FEL_Surrogate"

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Metadata - EDIT FOR EACH MODEL
EMAIL = "gopikab@slac.stanford.edu"
REPO = "https://github.com/slaclab/LCLS_FEL_Surrogate"
BEAM_PATH = "cu_hxr"
DESCRIPTION = "Self-contained ML-based surrogate model of the LCLS FEL pulse intensity"
READY_TO_DEPLOY = "true"  # Set to "true" for production, "false" for dev
STAGE = "production"  # development, staging, or production


# Load model from yaml
model = TorchModel("model_config.yaml")

# Set tags
model_tags = {
    "email": EMAIL,
    "repo": REPO,
    "beam_path": BEAM_PATH,
    "description": DESCRIPTION,
    "stage": STAGE,
}

version_tags = {
    "ready_to_deploy": READY_TO_DEPLOY,
}

# Register model
model.register_to_mlflow(
    artifact_path="models",
    registered_model_name="lcls_fel_surrogate",
    tags=model_tags,
    version_tags=version_tags
)