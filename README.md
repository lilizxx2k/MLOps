# Machine Learning Operations – Facial Emotion Detection

## Overall Goal
The goal of this project is to develop an end-to-end **Machine Learning Operations (MLOps) pipeline**
for **facial emotion detection**.  
The focus of the project is on **reproducibility, automation, version control, and deployment**
rather than achieving high model performance.

This follows the course guideline that MLOps engineers are evaluated on how fast and
reliably they can set up a production-ready pipeline, not on model accuracy.

---

## Dataset
We use the **AffectNet – YOLO Format** dataset from Kaggle.

- **Source:** https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format
- **Modality:** Image data
- **Task:** Facial emotion detection
- **Annotation format:** YOLO object detection format (bounding boxes + labels)
- **Emotion classes:** anger, contempt, disgust, fear, happy, neutral, sad, surprise
- **Dataset split:** train / validation / test
- **Dataset size:** 264.08 MB

The dataset is already structured for YOLO-based object detection, which minimizes preprocessing
effort and allows rapid experimentation. This aligns with the course recommendation to choose
datasets with simple data loading and manageable size.

---

## Model
The project starts with a **public baseline YOLO-based model** using pre-trained weights.

- **Framework:** PyTorch ecosystem
- **Model type:** YOLO (baseline)
- **Goal:** Fast experimentation and deployment

Model architecture and hyperparameters may evolve during the project. Model performance is **not**
a grading criterion.

---



---

## MLOps Focus
The main focus areas of this project are:
- Version control of code, configuration, and data
- Reproducible training pipelines
- Experiment tracking
- Continuous integration for automated testing
- Deployment of a trained model for user interaction
- Collaboration using shared workflows and repository structure

---
## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## API Usage
This project provides a REST API for classifying emotions from images using a deep learning model (PyTorch). The service is containerized with Docker and deployed to **Google Cloud Run**.

## Live Deployment

The API is live and can be accessed at the following endpoint:
- **Base URL:** [https://mlops-api-1041024007691.us-central1.run.app](https://mlops-api-1041024007691.us-central1.run.app)

### Interactive API Documentation
FastAPI automatically generates interactive documentation. You can test the API directly from your browser without writing any code:
- **Swagger UI:** [https://mlops-api-1041024007691.us-central1.run.app/docs](https://mlops-api-1041024007691.us-central1.run.app/docs)

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check to verify the API and model are loaded. |
| `POST` | `/predict` | Upload an image to receive emotion classification and confidence score. |

### Example Usage (Python)
```python
import requests

url = "[https://mlops-api-1041024007691.us-central1.run.app/predict](https://mlops-api-1041024007691.us-central1.run.app/predict)"
files = {'file': open('sample_image.jpg', 'rb')}
response = requests.post(url, files=files)

print(response.json())
# Output: {"emotion": "happy", "confidence": 0.982}))

