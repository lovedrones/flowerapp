# Flower Classification ML Project

A machine learning project that classifies flower images using deep learning, with a user-friendly Streamlit web interface.

## Features

- **Deep Learning Model**: Uses ResNet50 with transfer learning trained on the Flowers-102 dataset
- **102 Flower Classes**: Classifies images into 102 different flower categories
- **Web Interface**: Beautiful Streamlit dashboard for easy image upload and classification
- **Real-time Predictions**: Instant classification results with confidence scores

## Project Structure

```
.
├── app.py              # Streamlit web application
├── train.py            # Model training script
├── utils.py            # Utility functions for data loading and inference
├── requirements.txt    # Python dependencies
├── models/             # Directory for saved models
└── data/               # Directory for dataset (auto-created)

```

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

First, train the model on the Flowers-102 dataset:

```bash
python train.py
```

This will:
- Automatically download the Flowers-102 dataset
- Train a ResNet50 model with transfer learning
- Save the best model to `models/best_model.pth`

**Note**: Training may take several hours depending on your hardware. The dataset will be automatically downloaded on first run.

### 2. Run the Streamlit App

Once the model is trained, launch the web application:

```bash
streamlit run app.py
```

The app will open in your browser. You can then:
- Upload a flower image (JPG, JPEG, or PNG)
- View the top 5 predictions with confidence scores
- See the classification results instantly

## Model Details

- **Architecture**: ResNet50 (pre-trained on ImageNet)
- **Dataset**: Flowers-102 (102 flower categories)
- **Training**: Transfer learning with fine-tuning
- **Data Augmentation**: Random rotation, flipping, color jitter
- **Optimization**: Adam optimizer with learning rate scheduling
- **Early Stopping**: Prevents overfitting

## Dataset

The Flowers-102 dataset contains:
- 8,189 images
- 102 flower categories
- Predefined train/validation/test splits

The dataset is automatically downloaded from the official source when you run `train.py`.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Streamlit 1.28+
- See `requirements.txt` for complete list

## Notes

- The model requires a trained checkpoint (`models/best_model.pth`) to run the app
- GPU is recommended for training but not required for inference
- First-time dataset download may take some time (~330 MB)

## License

This project uses the Flowers-102 dataset, which is available for research purposes.

