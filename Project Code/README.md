To activate virtual environment:
1) Windows: venv\Scripts\activate
2) MacOS: source venv/bin/activate

To install requirements:
pip install -r requirements.txt
if errored install in groups:
pip install numpy==1.23.5 pandas==1.5.3 scikit-learn==1.2.2 scipy==1.10.1
pip install tensorflow==2.12.0 keras==2.12.0
pip install opencv-python==4.7.0.72 opencv-contrib-python==4.7.0.72 mediapipe==0.10.7
pip install deepface==0.0.75 fer==2023.2.22
pip install librosa==0.9.2 speechrecognition==3.10.0 pydub==0.25.1 soundfile==0.12.1
pip install matplotlib==3.7.1 seaborn==0.12.2 plotly==5.14.1
pip install flask==2.3.2 flask-cors==4.0.0
pip install reportlab==4.0.4 weasyprint==59.0
pip install tqdm==4.65.0 joblib==1.2.0 pillow==9.5.0
pip install jupyter==1.0.0 ipykernel==6.22.0

# Acting Performance Assessment System

An AI-powered system for analyzing acting performances through computer vision and deep learning. The system evaluates eye movement, emotions, voice, and gestures to provide comprehensive feedback on audition performances.

## Features

- **Multi-modal Analysis**: Analyzes eye tracking, facial emotions, body pose, gestures, and voice
- **Multiple ML Models**: Implements 8 different deep learning models for comparison
- **Ensemble Learning**: Combines multiple models for improved accuracy
- **Web Interface**: User-friendly Flask web application
- **Detailed Reports**: Generates comprehensive PDF reports with visualizations
- **Model Comparison**: Jupyter notebook for comparing different architectures

## Models Implemented

1. CNN-LSTM
2. LSTM
3. BiLSTM (Bidirectional LSTM)
4. GRU
5. RNN
6. CNN-RNN
7. Keras DNN
8. PyCaret (Classical ML)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd acting_assessment_system