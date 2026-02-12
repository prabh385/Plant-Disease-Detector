# Plant Disease Detector

A real-time plant disease detection system using deep learning. This project uses a pre-trained TensorFlow model to identify diseases in various plants through webcam feed or image input.

## Features

- **Real-time Detection**: Live webcam scanning with instant disease classification
- **High Accuracy**: Pre-trained deep learning model for plant disease recognition
- **Multiple Plant Support**: Detects diseases across 14+ different plant species
- **Confidence Threshold**: Automatic detection stop when confidence reaches 85% (customizable)
- **Live Feedback**: Visual display of scanning progress with confidence scores

## Supported Plants & Diseases

The model can detect the following plant conditions:

### Apple
- Apple scab
- Black rot
- Cedar apple rust
- Healthy

### Blueberry
- Healthy

### Cherry (including sour)
- Powdery mildew
- Healthy

### Corn (maize)
- Cercospora leaf spot / Gray leaf spot
- Common rust
- Northern Leaf Blight
- Healthy

### Grape
- Black rot
- Esca (Black Measles)
- Leaf blight (Isariopsis Leaf Spot)
- Healthy

### Orange
- Haunglongbing (Citrus greening)

### Peach
- Bacterial spot
- Healthy

### Pepper (bell)
- Bacterial spot
- Healthy

### Potato
- Early blight
- Late blight
- Healthy

### Raspberry
- Healthy

### Soybean
- Healthy

### Squash
- Powdery mildew

### Strawberry
- Leaf scorch
- Healthy

### Tomato
- Bacterial spot
- Early blight
- Late blight
- Leaf Mold
- Septoria leaf spot
- Spider mites (Two-spotted spider mite)
- Target Spot
- Tomato Yellow Leaf Curl Virus
- Tomato mosaic virus
- Healthy

## Requirements

- Python 3.7+
- TensorFlow/Keras
- OpenCV
- NumPy

## Installation

1. Clone or download this repository:
```bash
cd plant_diseases_detector
```

2. Install required dependencies:
```bash
pip install tensorflow opencv-python numpy
```

3. Ensure the model file exists:
   - The pre-trained model should be located at: `Model/plant_disease_model.h5`

## Usage

Run the detection script:
```bash
python predict.py
```

### How It Works

1. The application accesses your default webcam
2. Frames are continuously captured and resized to 96x96 pixels
3. The model predicts the plant disease class and confidence percentage
4. The scanning continues until confidence reaches 85% (default threshold)
5. Press **Q** to quit early at any time during scanning
6. Once detection is complete, the result is displayed with the final prediction

### Customization

You can modify the confidence threshold in `predict.py`:
```python
CONFIDENCE_THRESHOLD = 85   # Change to 80 or any other value
```

Lower values will stop detection faster but may be less accurate.

## Result Display

When a disease is detected with sufficient confidence, the application shows:
- **Disease/Plant Classification**: The identified plant disease or healthy status
- **Confidence Score**: The model's confidence percentage (0-100%)
- **Frozen Frame**: The captured image that triggered the detection

## Project Structure

```
plant_diseases_detector/
├── predict.py              # Main detection script
├── Model/
│   └── plant_disease_model.h5  # Pre-trained TensorFlow model
└── README.md               # This file
```

## License

This project uses publicly available plant disease datasets for training the model.

## Notes

- Ensure good lighting for best results
- Position the plant leaf clearly in the camera frame
- The model works best with high-quality images
- Keep the webcam steady during scanning for accurate predictions

## Troubleshooting

- **Model not found**: Ensure `Model/plant_disease_model.h5` exists in the project directory
- **Webcam not working**: Check camera permissions and connectivity
- **Slow predictions**: This is normal for the first run as TensorFlow initializes
- **Low accuracy**: Try adjusting the confidence threshold or ensuring good lighting

## Author

Plant Disease Detection System

## Contributing

Feel free to improve the model accuracy or add support for additional plants and diseases.
