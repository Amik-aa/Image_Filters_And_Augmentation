import streamlit as st
import cv2
import numpy as np
import PIL.Image
import tempfile

# Function to convert OpenCV image to PIL for display in Streamlit
def convert_cv2_to_pil(image):
    return PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Function to apply a pencil sketch effect
def apply_pencil_sketch(image):
    gray, sketch = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch

# Function to apply a sepia effect
def apply_sepia(image):
    kernel = np.array([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]])
    sepia_image = cv2.transform(image, kernel)
    return np.clip(sepia_image, 0, 255).astype(np.uint8)

# Function to apply a grayscale effect
def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to apply an emboss effect
def apply_emboss(image):
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return cv2.filter2D(image, -1, kernel)

# Function to apply a negative effect
def apply_negative(image):
    return cv2.bitwise_not(image)


st.title("Image Filters & Effects App 🎨🖼️")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        image_path = temp_file.name

    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert BGR to RGB for correct color representation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the original image
    st.image(image, caption="Original Image", use_column_width=True)

    # Select effect
    effect = st.selectbox("Choose an effect", ["None", "Pencil Sketch", "Sepia", "Grayscale", "Emboss", "Negative"])

    # Apply selected effect
    if effect == "Pencil Sketch":
        processed_image = apply_pencil_sketch(image)
    elif effect == "Sepia":
        processed_image = apply_sepia(image)
    elif effect == "Grayscale":
        processed_image = apply_grayscale(image)
    elif effect == "Emboss":
        processed_image = apply_emboss(image)
    elif effect == "Negative":
        processed_image = apply_negative(image)
    else:
        processed_image = image  # No effect applied

    # Convert OpenCV image to PIL format for display in Streamlit
    processed_image_pil = convert_cv2_to_pil(processed_image)

    # Display processed image
    st.image(processed_image_pil, caption=f"{effect} Effect", use_column_width=True)

    # download button for processed image
    st.download_button(label="Download Image",
                       data=processed_image_pil.tobytes(),
                       file_name=f"{effect.lower()}_effect.png",
                       mime="image/png")
