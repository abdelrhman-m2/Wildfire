# 🔥 AI Wildfire Detection System

Advanced deep learning application for real-time wildfire detection using satellite and aerial imagery.

## ✨ Features

- 🤖 **CNN-Based Detection**: Deep learning model for accurate wildfire identification
- 📊 **Detailed Analytics**: Comprehensive metrics, confidence scores, and risk assessment
- 📈 **Interactive Charts**: Gauge charts, probability distributions, and history tracking
- 🗺️ **Interactive Maps**: Click-to-analyze any location worldwide
- 💾 **Export Capability**: Download analysis history as CSV
- 🎨 **Modern UI**: Beautiful, responsive interface with real-time updates

## 🚀 Quick Start

### Prerequisites
- Python 3.9 - 3.11
- 4GB+ RAM
- Internet connection

### Installation

```bash
# Clone the repository
cd your-project-folder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt



# Run the application
streamlit run app.py
```


## 🎯 Usage

### Method 1: Upload Images
1. Navigate to the **"Upload Image"** tab
2. Select a satellite/aerial image (JPG, PNG, JPEG, TIFF)
3. Click **"Analyze"**
4. View detailed detection results

### Method 2: Satellite Analysis (Requires Earth Engine)
1. Navigate to the **"Satellite Analysis"** tab
2. Enter coordinates or click on the map
3. Click **"Fetch & Analyze"**
4. View live satellite imagery analysis

### View Analytics
- Check the **"Analytics"** tab for historical data
- Export results as CSV for further analysis

## 🛠️ Tech Stack

- **Framework**: Streamlit
- **Machine Learning**: TensorFlow/Keras
- **Satellite Data**: Google Earth Engine (Sentinel-2)
- **Visualization**: Plotly, Matplotlib, Folium
- **Image Processing**: PIL, OpenCV

## 📁 Project Structure

```
wildfire-detection/
├── app.py                      # Main Streamlit application
├── wildfire_model.keras        # Trained CNN model
├── requirements.txt            # Python dependencies
          # Configuration secrets
├── SETUP_GUIDE.md             # Detailed setup instructions
└── README.md                  # This file
```




## 📊 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input**: 224×224×3 RGB images
- **Output**: Binary classification (Wildfire / No Wildfire)
- **Metrics**: Accuracy, AUC, Confidence Score
- **Decision Threshold**: 0.5 (configurable)

## ⚠️ Important Notes

- **Educational Use Only**: This tool is for educational and research purposes
- **Not for Emergency Decisions**: Always contact official emergency services
- **Verify Results**: All detections should be verified by experts
- **Data Privacy**: Satellite data is public; uploaded images remain local

## 🐛 Troubleshooting

### Model Not Found
```bash
# Ensure wildfire_model.keras is in the same directory
ls -lh wildfire_model.keras
```



See [SETUP_GUIDE.md](SETUP_GUIDE.md) for more solutions.

## 📝 Development

# Run app
streamlit run app.py
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **AbdelRhman** - Initial work

