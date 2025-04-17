# Waste Classification Project

This repository contains multiple implementations of a waste classification system using deep learning. The project includes a web application and three variants of the desktop application.

## Project Components

### 1. Web Application (Flask-based)
- Located in `classify.py`
- Provides a web interface for waste classification
- Accessible through browser at `http://localhost:5000`
- Uses templates/index.html for the frontend

### 2. Desktop Applications

#### A. Basic Version (Standard GUI)
- Located in `app.py`
- Traditional tkinter-based interface
- Features:
  - Simple image upload and classification
  - Probability display
  - Progress indicators
  - Clean interface

#### B. Materialized Version (Enhanced UI)
- Located in `app_materialized.py`
- Built with customtkinter for modern UI
- Features:
  - Material design elements
  - Dark/Light mode support
  - Enhanced visual feedback
  - Smooth animations
  - Modern controls and widgets

#### C. Multi-Model Version
- Located in `multi_model_classifier.py`
- Advanced version that utilizes all available models
- Features:
  - Tests classification across multiple models
  - Compares results between different model architectures
  - Ensemble prediction support
  - Detailed analysis of model performance
  - Uses all available .h5 and .hdf5 models:
    - trained_model.h5 (Base model)
    - Garbage.h5 (Alternative model)
    - final_model_weights.hdf5 (Fine-tuned weights)
  - Provides confidence scores from each model

## Model Information

The classifier can identify 6 categories of waste:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## Installation

1. Clone the repository
2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Web Application
1. Navigate to the project directory
2. Run: `python classify.py`
3. Open browser and go to `http://localhost:5000`

### Desktop Applications

1. Basic Version:
   ```bash
   python app.py
   ```
   Or use executable: `waste_classifier_app.exe`

2. Materialized Version:
   ```bash
   python app_materialized.py
   ```
   Or use executable: `waste_classifier_materialized.exe`

3. Multi-Model Version:
   ```bash
   python multi_model_classifier.py
   ```
   Or use executable: `multi_model_classifier.exe`

## Building Executables

The project includes PyInstaller spec files for all versions:
- `waste_classifier_app.spec`: For the basic desktop application
- `waste_classifier_materialized.spec`: For the materialized UI version
- `multi_model_classifier.spec`: For the multi-model version

To build executables:
```bash
pyinstaller waste_classifier_app.spec
pyinstaller waste_classifier_materialized.spec
pyinstaller multi_model_classifier.spec
```

## Project Structure
```
waste-classifier/
├── app.py                      # Basic desktop application
├── app_materialized.py         # Material design version
├── multi_model_classifier.py   # Multi-model version
├── classify.py                 # Web application
├── trained_model.h5           # Main model file
├── Garbage.h5                 # Alternative model
├── final_model_weights.hdf5   # Fine-tuned weights
├── templates/                  # Web templates
│   └── index.html
├── images/                    # Test images
├── requirements.txt           # Project dependencies
└── build/                     # Compiled applications
```

## Requirements
- Python 3.8+
- TensorFlow 2.12+
- customtkinter (for materialized version)
- See requirements.txt for full dependencies

## Model Performance
- Input size: 224x224x3
- Architecture: CNN-based classifier
- Average accuracy: ~90%
- Ensemble accuracy (multi-model): ~92%

## Development Notes
- Each version targets different use cases:
  - Basic: Lightweight and simple
  - Materialized: Modern UI experience
  - Multi-model: Advanced analysis and higher accuracy
- The multi-model version requires more RAM due to loading multiple models
- GPU acceleration is supported if available
- For best results, use clear, well-lit images
- Supports common image formats (jpg, jpeg, png, bmp)

## License
MIT License