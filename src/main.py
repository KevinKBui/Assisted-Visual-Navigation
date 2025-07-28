import model
import argparse
import mlflow
from pathlib import Path

if __name__ == '__main__':
    # Set up MLFlow Tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment('Prostate Cancer Risk Prediction')

    # Create models folder if it doesn't exist
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description='Train an XGBoost model with hyperparameter tuning.')
    parser.add_argument('--file_path', required=False, type=str, default=r'/workspaces/Assisted-Visual-Navigation/Data/raw/synthetic_prostate_cancer_risk.csv', help='Add file path to the CSV dataset used for training.')
    args = parser.prase_args()
    model.run(file_path=args.file_path)