import streamlit as st
from nst import NeuralStyleTransfer
import os
from PIL import Image
import time
import tensorflow as tf

# Set page config
st.set_page_config(
    page_title="Neural Style Transfer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f5f5f5;
    }
    .stButton > button {
        width: 100%;
        background-color: #007BFF;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .upload-section {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #333333;
    }
    .result-image {
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .header-container {
        background-color: #007BFF;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .parameter-section {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #333333;
    }
    .info-box {
        background-color: #e9ecef;
        border-left: 5px solid #007BFF;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("temp", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return os.path.join("temp", uploaded_file.name)
    except:
        return None

# Initialize session state
if 'nst' not in st.session_state:
    st.session_state.nst = NeuralStyleTransfer()

# Create temp directory if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Header Section
st.markdown("""
    <div class='header-container'>
        <h1>🎨 Neural Style Transfer</h1>
        <p style='font-size: 1.2rem; margin-top: 1rem;'>
            Transform your photos into artistic masterpieces using AI!
        </p>
    </div>
""", unsafe_allow_html=True)

# Quick Guide
st.markdown("""
    <div class='info-box'>
        <h3>📝 Quick Guide</h3>
        <ol>
            <li>Upload your main photo (Content Image)</li>
            <li>Upload the artistic style photo (Style Image)</li>
            <li>Adjust the parameters in the sidebar</li>
            <li>Click 'Start Style Transfer' and wait for the magic!</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# Create two columns for content and style images
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.subheader("📸 Content Image")
    st.markdown("*This is your main photo that you want to transform*")
    content_image = st.file_uploader("Choose your content image", type=['png', 'jpg', 'jpeg'], key="content")
    if content_image:
        st.image(content_image, use_container_width=True, caption="Your Content Image")
        content_path = save_uploaded_file(content_image)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.subheader("🎨 Style Image")
    st.markdown("*This is the artistic style you want to apply*")
    style_image = st.file_uploader("Choose your style image", type=['png', 'jpg', 'jpeg'], key="style")
    if style_image:
        st.image(style_image, use_container_width=True, caption="Your Style Image")
        style_path = save_uploaded_file(style_image)
    st.markdown("</div>", unsafe_allow_html=True)

# Parameters in sidebar with explanations
st.sidebar.markdown("""
    <div class='parameter-section'>
        <h2>🎮 Style Transfer Controls</h2>
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
    <div class='info-box'>
        <h4>⚙️ Parameter Guide</h4>
        <p>Adjust these settings to control the style transfer process.</p>
    </div>
""", unsafe_allow_html=True)

iterations = st.sidebar.slider(
    "Number of Iterations",
    min_value=1,
    max_value=1000,
    value=100,
    help="More iterations = better quality but slower processing"
)

style_weight = st.sidebar.slider(
    "Style Intensity",
    min_value=1e-3,
    max_value=1e-1,
    value=1e-2,
    format="%.3f",
    help="Higher values = stronger artistic style effect"
)

content_weight = st.sidebar.slider(
    "Content Preservation",
    min_value=1e3,
    max_value=1e5,
    value=1e4,
    format="%.0f",
    help="Higher values = more original content preserved"
)

# Process button
if st.button("🎨 Start Style Transfer"):
    if content_image is not None and style_image is not None:
        with st.spinner('🎨 Creating your masterpiece... This may take a few minutes.'):
            try:
                # Create status elements
                progress_container = st.container()
                with progress_container:
                    st.markdown("### 🎨 Style Transfer Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    time_text = st.empty()
                
                # Initialize optimizer and image
                image = tf.Variable(st.session_state.nst.load_img(content_path))
                opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
                
                # Extract style and content features
                style_targets = [st.session_state.nst.gram_matrix(style) for style in 
                               st.session_state.nst.extractor(st.session_state.nst.load_img(style_path))[: len(st.session_state.nst.style_layers)]]
                content_targets = st.session_state.nst.extractor(st.session_state.nst.load_img(content_path))[len(st.session_state.nst.style_layers):]
                
                # Perform style transfer with progress update
                start_time = time.time()
                
                # Display initial parameters
                st.sidebar.markdown("### Current Parameters")
                st.sidebar.markdown(f"""
                    - Iterations: {iterations}
                    - Style Weight: {style_weight:.3f}
                    - Content Weight: {content_weight:.0f}
                """)
                
                for i in range(iterations):
                    # Update progress
                    progress = (i + 1) / iterations
                    elapsed = time.time() - start_time
                    remaining = int((elapsed / (i + 1)) * (iterations - (i + 1)))
                    
                    # Update UI with detailed information
                    progress_bar.progress(progress)
                    status_text.markdown(f"""
                        ### Style Transfer Status
                        - **Progress: {int(progress * 100)}%**
                        - **Current Iteration: {i+1}/{iterations}**
                        - **Style Intensity: {style_weight:.3f}**
                        - **Content Preservation: {content_weight:.0f}**
                    """)
                    time_text.markdown(f"**⏱️ Time remaining: {remaining} seconds**")
                    
                    # Perform style transfer step with user-selected weights
                    loss = st.session_state.nst.train_step(
                        image=image,
                        style_targets=style_targets,
                        content_targets=content_targets,
                        style_weight=style_weight,
                        content_weight=content_weight,
                        opt=opt
                    )
                    
                    # Small delay for UI update
                    time.sleep(0.1)

                # Show completion
                progress_bar.progress(1.0)
                status_text.markdown("### ✨ Style Transfer Complete!")
                time_text.empty()

                # Final result
                result = st.session_state.nst.finalize_image(image)
                
                # Provide download options
                st.markdown("<h2>✨ Your Artistic Creation</h2>", unsafe_allow_html=True)
                st.image(result, use_container_width=True, caption="Style Transfer Result", clamp=True)
                st.download_button(label="Download as PNG", data=result.tobytes(), file_name="styled_image.png", mime="image/png")
                st.download_button(label="Download as JPG", data=result.convert('RGB').tobytes(), file_name="styled_image.jpg", mime="image/jpeg")

                # Footer displayed after image generation
                st.markdown("""
                    <div style='text-align: center; margin-top: 2rem; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;'>
                        <h3 style='color: #333;'>Thank You for Using!</h3>
                        <p style='color: #333;'>Created with 🎨 and 🤖</p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
    else:
        st.warning("⚠️ Please upload both content and style images to begin.")

# Cleanup temp files
for file in os.listdir("temp"):
    os.remove(os.path.join("temp", file))
