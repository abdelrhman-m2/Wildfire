# =======================
# Imports
# =======================
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array
import gdown
import requests
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import cv2

# =======================
# Page Configuration
# =======================
st.set_page_config(
    page_title="🔥 Wildfire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======================
# Custom CSS Styling
# =======================
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .main {
        padding: 0;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        animation: fadeIn 0.8s ease-in;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    
    .result-card-fire {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(255, 75, 75, 0.3);
        margin: 1rem 0;
    }
    
    .result-card-safe {
        background: linear-gradient(135deg, #51CF66 0%, #69DB7C 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(81, 207, 102, 0.3);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #FF4B4B;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .confidence-gauge {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =======================
# Constants
# =======================
IMG_SIZE = 224
CLASS_LABELS = {0: "✅ No Wildfire", 1: "🔥 Wildfire Detected"}
MODEL_PATH = "wildfire_model.keras"

# =======================
# Load Model
# =======================
@st.cache_resource
def load_model():
    """Load the trained wildfire detection model"""
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model for the first time...")
        # If you have a Google Drive link, use gdown
        # gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.warning("⚠️ Please upload the wildfire_model.keras file to the app directory")
        return None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# =======================
# Image Preprocessing
# =======================
def preprocess_image(image):
    """Preprocess image for model input"""
    try:
        image_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        img = img_to_array(image_resized)
        img = img / 255.0  # Normalize to [0, 1]
        return np.expand_dims(img, axis=0)
    except Exception as e:
        st.error(f"❌ Preprocessing error: {e}")
        return None

# =======================
# Prediction Function
# =======================
def predict(image_array):
    """Make prediction with detailed statistics"""
    if model is None:
        return None, None, None
    
    try:
        preds = model.predict(image_array, verbose=0)
        prob = float(preds[0][0]) if len(preds[0]) == 1 else float(preds[0][1])
        
        label = 1 if prob >= 0.5 else 0
        confidence = prob if label == 1 else (1 - prob)
        
        stats = {
            'wildfire_probability': prob * 100,
            'no_wildfire_probability': (1 - prob) * 100,
            'confidence': confidence * 100,
            'prediction': CLASS_LABELS[label],
            'risk_level': get_risk_level(prob),
            'decision_threshold': 0.5
        }
        
        return label, confidence, stats
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None, None

def get_risk_level(prob):
    """Determine risk level based on probability"""
    if prob < 0.3:
        return "🟢 LOW"
    elif prob < 0.7:
        return "🟡 MODERATE"
    else:
        return "🔴 HIGH"

# =======================
# Visualizations
# =======================
def create_gauge_chart(probability):
    """Create interactive gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={'text': "Wildfire Probability"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=400)
    return fig

def create_probability_chart(stats):
    """Create probability comparison chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Wildfire', 'No Wildfire'],
            y=[stats['wildfire_probability'], stats['no_wildfire_probability']],
            marker=dict(
                color=['#FF4B4B', '#51CF66'],
                line=dict(color=['#c92a2a', '#2b8a3e'], width=2)
            ),
            text=[f"{stats['wildfire_probability']:.1f}%", f"{stats['no_wildfire_probability']:.1f}%"],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.2f}%<extra></extra>'
        )
    ])
    fig.update_layout(
        title="Prediction Probability Distribution",
        yaxis_title="Probability (%)",
        height=400,
        showlegend=False
    )
    return fig

def create_thermometer(prob, label):
    """Create thermometer-style visualization"""
    fig, ax = plt.subplots(figsize=(3, 8), facecolor='white', edgecolor='#e0e0e0')
    
    # Determine color based on risk
    if prob < 0.3:
        color = '#51CF66'
        risk_text = "LOW RISK"
    elif prob < 0.7:
        color = '#FFD93D'
        risk_text = "MODERATE RISK"
    else:
        color = '#FF4B4B'
        risk_text = "HIGH RISK"
    
    # Thermometer body
    ax.barh(0, 1, height=0.3, color='#e0e0e0', edgecolor='#333', linewidth=3)
    ax.barh(0, prob, height=0.3, color=color, edgecolor='#333', linewidth=3)
    
    # Percentage text
    ax.text(0.5, 0, f'{prob*100:.1f}%', ha='center', va='center',
            fontsize=28, fontweight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    # Risk level
    ax.text(0.5, -0.45, risk_text, ha='center', va='top',
            fontsize=14, fontweight='bold', color=color)
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.6, 0.5)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def show_image_comparison(original, processed):
    """Side-by-side image comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Processed
    processed_display = processed[0]
    processed_display = np.clip(processed_display, 0, 1)
    axes[1].imshow(processed_display)
    axes[1].set_title("Model Input (224×224)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    return fig

def create_history_chart(history_data):
    """Create prediction history chart"""
    if not history_data:
        return None
    
    probs = [h['probability'] * 100 for h in history_data]
    times = [h['timestamp'].strftime("%H:%M") for h in history_data]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=probs,
        mode='lines+markers',
        name='Wildfire Probability',
        line=dict(color='#FF4B4B', width=3),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="Prediction History",
        xaxis_title="Time",
        yaxis_title="Probability (%)",
        height=400,
        hovermode='x unified'
    )
    return fig

# =======================
# Display Results
# =======================
def display_results(image, label, confidence, stats, show_comparison=False):
    """Display comprehensive prediction results"""
    
    # Main result card
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.image(image, caption="Input Image", use_container_width=True)
    
    with col2:
        fig = create_thermometer(stats['wildfire_probability']/100, label)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    
    with col3:
        if label == 1:
            st.markdown("""
            <div class="result-card-fire">
                <h2 style="margin-bottom: 1rem;">🔥 WILDFIRE DETECTED</h2>
                <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <strong>Severity:</strong> """ + stats['risk_level'] + """
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card-safe">
                <h2 style="margin-bottom: 1rem;">✅ NO WILDFIRE DETECTED</h2>
                <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
                    <strong>Confidence:</strong> """ + stats['risk_level'] + """
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detailed metrics
    st.subheader("📊 Model Prediction Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Wildfire %",
            f"{stats['wildfire_probability']:.2f}%",
            delta=None,
            delta_color="off"
        )
    
    with col2:
        st.metric(
            "No Wildfire %",
            f"{stats['no_wildfire_probability']:.2f}%",
            delta=None,
            delta_color="off"
        )
    
    with col3:
        st.metric(
            "Confidence",
            f"{stats['confidence']:.2f}%",
            delta=None,
            delta_color="off"
        )
    
    with col4:
        st.metric(
            "Risk Level",
            stats['risk_level'],
            delta=None,
            delta_color="off"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_gauge = create_gauge_chart(stats['wildfire_probability']/100)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        fig_prob = create_probability_chart(stats)
        st.plotly_chart(fig_prob, use_container_width=True)
    
    # Image comparison
    if show_comparison:
        st.markdown("---")
        st.subheader("🔍 Image Processing Details")
        img_array = preprocess_image(image)
        if img_array is not None:
            fig_comp = show_image_comparison(image, img_array)
            st.pyplot(fig_comp, use_container_width=True)
            plt.close()

# =======================
# Main Application
# =======================

# Header
st.markdown('<h1 class="main-header">🔥 Wildfire Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
<p class="subtitle">
    Advanced AI-Powered Wildfire Detection using Deep Learning<br>
    <small>Analyze satellite and aerial imagery for wildfire detection in real-time</small>
</p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    show_comparison = st.checkbox("Show Image Processing Details", value=False)
    st.markdown("---")
    
    # Session state for history
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.prediction_history = []
        st.success("History cleared!")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Image", "📊 Analytics", "ℹ️ About", "⚡ Advanced"])

# =======================
# TAB 1 — Image Upload
# =======================
with tab1:
    st.subheader("Upload Satellite or Aerial Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "png", "jpeg", "tiff"],
            help="Upload satellite or aerial imagery for wildfire detection"
        )
    
    with col2:
        st.info("""
        **Recommended:**
        - Format: JPG, PNG
        - Size: 224×224 or larger
        - Quality: High resolution
        """)
    
    if uploaded_file is not None:
        if model is None:
            st.error("❌ Model not loaded. Please check the model file.")
        else:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(image, caption="Original Upload", use_container_width=True)
                
                with col2:
                    st.info(f"""
                    **Image Information:**
                    - Size: {image.size}
                    - Format: {image.format}
                    - Mode: {image.mode}
                    """)
                
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("🤖 Running AI analysis..."):
                        img_array = preprocess_image(image)
                        if img_array is not None:
                            label, confidence, stats = predict(img_array)
                            
                            if stats is not None:
                                # Store in history
                                st.session_state.prediction_history.append({
                                    'timestamp': datetime.now(),
                                    'probability': stats['wildfire_probability']/100,
                                    'label': label,
                                    'file': uploaded_file.name
                                })
                                
                                st.markdown("---")
                                display_results(image, label, confidence, stats, show_comparison)
            
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")

# =======================
# TAB 2 — Analytics
# =======================
with tab2:
    st.subheader("📊 Prediction Analytics")
    
    if st.session_state.prediction_history:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Predictions",
                len(st.session_state.prediction_history)
            )
        
        with col2:
            fire_count = sum(1 for h in st.session_state.prediction_history if h['label'] == 1)
            st.metric("Wildfires Detected", fire_count)
        
        with col3:
            avg_prob = np.mean([h['probability'] for h in st.session_state.prediction_history]) * 100
            st.metric("Average Probability", f"{avg_prob:.2f}%")
        
        st.markdown("---")
        
        fig_history = create_history_chart(st.session_state.prediction_history)
        if fig_history:
            st.plotly_chart(fig_history, use_container_width=True)
        
        st.markdown("---")
        
        # History table
        st.subheader("Prediction History")
        history_data = [
            {
                'Time': h['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'File': h['file'],
                'Probability': f"{h['probability']*100:.2f}%",
                'Result': '🔥 Fire' if h['label'] == 1 else '✅ Safe'
            }
            for h in reversed(st.session_state.prediction_history)
        ]
        
        st.dataframe(history_data, use_container_width=True)
    
    else:
        st.info("📭 No predictions yet. Upload images to see analytics!")

# =======================
# TAB 3 — About
# =======================
with tab3:
    st.markdown("""
    ## 🔥 About This System
    
    This advanced wildfire detection system uses deep learning to analyze satellite 
    and aerial imagery for signs of active wildfires.
    
    ### 🎯 Key Features
    - **Real-time Analysis**: Instant wildfire detection from images
    - **Deep Learning**: Trained CNN model for accurate predictions
    - **Detailed Metrics**: Probability scores, confidence levels, risk assessment
    - **Interactive Visualizations**: Charts, gauges, and comparisons
    - **Prediction History**: Track all predictions in session
    - **Image Processing**: View preprocessing steps
    
    ### 🧠 Model Information
    - **Architecture**: Convolutional Neural Network (CNN)
    - **Input Size**: 224×224×3 RGB pixels
    - **Training Data**: Wildfire satellite imagery dataset
    - **Output**: Binary classification (Wildfire / No Wildfire)
    - **Regularization**: Dropout + L2 Regularization
    - **Optimization**: Adam optimizer with learning rate decay
    
    ### 📊 Model Metrics
    - **Wildfire Probability**: Likelihood of wildfire (0-100%)
    - **No Wildfire Probability**: Likelihood of no fire (0-100%)
    - **Confidence Score**: Model certainty (0-100%)
    - **Risk Level**: LOW (0-30%), MODERATE (30-70%), HIGH (70-100%)
    - **Decision Threshold**: 0.5 (50% probability cutoff)
    
    ### 🛠️ How to Use
    1. Upload a satellite or aerial image
    2. Click "Analyze Image"
    3. View detailed predictions and metrics
    4. Check prediction history in Analytics tab
    
    ### ⚠️ Important Disclaimer
    This is an **educational and assistive tool** and should **NOT** be used as the 
    sole method for wildfire detection in critical situations. Always verify with 
    official sources and contact emergency services immediately.
    
    ### 📚 Technical Stack
    - **Framework**: TensorFlow/Keras
    - **Visualization**: Plotly, Matplotlib
    - **Interface**: Streamlit
    - **Language**: Python 3.8+
    
    ### 🎓 Model Training Details
    - **Layers**: 3 Conv2D + MaxPooling + Flatten + Dense
    - **Dropout Rate**: 0.4-0.5 (prevent overfitting)
    - **L2 Regularization**: 1e-3 weight decay
    - **Loss Function**: Categorical Crossentropy
    - **Metrics**: AUC, Accuracy
    - **Batch Size**: 256
    - **Epochs**: 50 (with early stopping)
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🌐 Deployment
    This app is ready for deployment on:
    - **Streamlit Cloud** (Free)
    - **Docker** (Custom servers)
    - **AWS/GCP** (Cloud platforms)
    
    ### 📝 Requirements
    ```
    streamlit>=1.28.0
    tensorflow>=2.13.0
    pillow>=10.0.0
    numpy>=1.24.0
    plotly>=5.17.0
    opencv-python>=4.8.0
    streamlit-option-menu>=0.3.12
    ```
    """)

# =======================
# TAB 4 — Advanced
# =======================
with tab4:
    st.subheader("⚡ Advanced Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Model Configuration")
        st.info("""
        **Current Settings:**
        - Input Size: 224×224
        - Output Classes: 2
        - Decision Threshold: 0.5
        - Confidence Metric: Softmax probability
        """)
    
    with col2:
        st.markdown("### 📈 Performance Metrics")
        st.info("""
        **Model Training:**
        - Loss Function: Categorical Crossentropy
        - Optimizer: Adam (lr=0.0005)
        - Batch Size: 256
        - Early Stopping: 10 epochs patience
        """)
    
    st.markdown("---")
    
    st.markdown("### 🎛️ Threshold Adjustment")
    threshold = st.slider("Decision Threshold", 0.0, 1.0, 0.5, 0.05,
                         help="Lower = more sensitive to wildfire, Higher = more conservative")
    
    st.info(f"""
    **Current Threshold: {threshold:.2f}**
    - Predictions >= {threshold:.2f} = Wildfire
    - Predictions < {threshold:.2f} = No Wildfire
    """)
    
    st.markdown("---")
    
    st.markdown("### 🧪 Test with Sample Data")
    if st.button("Generate Test Image", use_container_width=True):
        # Create a random test image
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(test_img, caption="Generated Test Image", use_container_width=True)
        
        with col2:
            if st.button("Analyze Test Image", type="primary", use_container_width=True):
                if model is None:
                    st.error("❌ Model not loaded")
                else:
                    with st.spinner("🤖 Analyzing test image..."):
                        img_array = preprocess_image(test_img)
                        if img_array is not None:
                            label, confidence, stats = predict(img_array)
                            if stats is not None:
                                display_results(test_img, label, confidence, stats, False)

# =======================
# Footer
# =======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999; padding: 2rem 0;'>
    <h4>🔥 Advanced Wildfire Detection System</h4>
    <p>Powered by TensorFlow • Streamlit • Deep Learning</p>
    <p style='font-size: 0.9rem; margin-top: 1rem;'>
        ⚠️ Educational use only • Not for emergency decision-making<br>
        Always contact official emergency services immediately
    </p>
</div>
""", unsafe_allow_html=True)