# Neural Style Transfer App 🎨

This application performs neural style transfer using VGG19 and Streamlit. It allows you to combine the content of one image with the artistic style of another image.

## Features

- Modern, user-friendly interface
- Real-time parameter adjustment
- High-quality style transfer using VGG19
- Progress tracking during transfer
- Ability to save results

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Upload your content and style images
3. Adjust the parameters:
   - Number of iterations (more iterations = better results but slower)
   - Style weight (how much to emphasize the style)
   - Content weight (how much to preserve the original content)
4. Click "Start Style Transfer"
5. Save your result if desired

## Deployment on Streamlit Cloud

1. Create a free account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Deploy the app by selecting your repository and the main file (`app.py`)

## Parameters

- **Content Weight**: Controls how much to preserve the content of the original image (default: 1e4)
- **Style Weight**: Controls how much to apply the style (default: 1e-2)
- **Iterations**: Number of optimization steps (default: 100)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Streamlit
- NumPy
- Pillow
