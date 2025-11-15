import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
import numpy as np
from utils import (
    load_model,
    predict_image,
    preprocess_image_for_inference,
    FLOWER_NAMES,
    NUM_CLASSES
)
import os

# Page configuration
st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B9D;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .top-prediction {
        font-size: 1.5rem;
        font-weight: bold;
        color: #FF6B9D;
    }
    </style>
""", unsafe_allow_html=True)

# Model configuration
MODEL_PATH = "models/best_model.pth"
NUM_CLASSES = 102
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def load_model_cached():
    """Load model with caching."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first using train.py")
        return None
    
    try:
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🌸 Flower Classification 🌸</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image of a flower and get instant classification results!</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model_cached()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This app uses a deep learning model trained on the Flowers-102 dataset 
        to classify flower images into 102 different categories.
        
        **How to use:**
        1. Upload a flower image
        2. Wait for the model to process
        3. View the top predictions with confidence scores
        """)
        
        st.header("Model Info")
        st.write(f"**Device:** {DEVICE}")
        st.write(f"**Classes:** {NUM_CLASSES}")
        st.write(f"**Architecture:** ResNet50")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a flower image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image file (JPG, JPEG, or PNG)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.header("Predictions")
        
        if uploaded_file is not None:
            # Show loading spinner
            with st.spinner("Classifying flower..."):
                # Make prediction
                predictions = predict_image(model, image, device=DEVICE, top_k=5)
            
            # Display top prediction prominently
            if predictions:
                top_pred = predictions[0]
                st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f'<p class="top-prediction">🌺 {top_pred["class"].title()}</p>', unsafe_allow_html=True)
                st.markdown(f'<p>Confidence: <strong>{top_pred["confidence"]*100:.2f}%</strong></p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display all top predictions
                st.subheader("Top 5 Predictions")
                
                for i, pred in enumerate(predictions, 1):
                    # Create progress bar for confidence
                    confidence_pct = pred["confidence"] * 100
                    st.write(f"**{i}. {pred['class'].title()}**")
                    st.progress(confidence_pct / 100)
                    st.write(f"   Confidence: {confidence_pct:.2f}%")
                    st.write("")
        else:
            st.info("👆 Please upload an image to get predictions")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with Streamlit and PyTorch | Flowers-102 Dataset"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

