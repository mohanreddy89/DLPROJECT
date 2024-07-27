# import streamlit as st
# import torch
# from diffusers import StableDiffusionImg2ImgPipeline
# from PIL import Image
# from utils import preprocess_image
# import numpy as np
# import matplotlib.pyplot as plt

# st.title("Image-to-Image Translation using Stable Diffusion")

# @st.cache_resource
# def load_model():
#     model_id = "runwayml/stable-diffusion-v1-5"
#     pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
#     return pipe

# pipe = load_model()

# def generate_image(input_image, prompt):
#     with torch.no_grad():
#         image = pipe(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5).images[0]
#     return image

# def get_image_stats(image):
#     img_array = np.array(image)
#     return {
#         "Mean": np.mean(img_array, axis=(0, 1)),
#         "Std": np.std(img_array, axis=(0, 1)),
#         "Min": np.min(img_array, axis=(0, 1)),
#         "Max": np.max(img_array, axis=(0, 1))
#     }

# def plot_histograms(input_image, output_image):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
#     for ax, img, title in zip([ax1, ax2], [input_image, output_image], ['Input Image', 'Output Image']):
#         img_array = np.array(img)
#         ax.hist(img_array[:,:,0].ravel(), bins=256, color='red', alpha=0.5, label='Red')
#         ax.hist(img_array[:,:,1].ravel(), bins=256, color='green', alpha=0.5, label='Green')
#         ax.hist(img_array[:,:,2].ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
#         ax.set_xlabel('Pixel Intensity')
#         ax.set_ylabel('Count')
#         ax.set_title(f'{title} Histogram')
#         ax.legend()
    
#     plt.tight_layout()
#     return fig

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     input_image = Image.open(uploaded_file).convert('RGB')
#     st.write("Uploaded Image")
#     st.image(input_image, use_column_width=True)

#     # Preprocess the image
#     preprocessed_image = preprocess_image(input_image)

#     # Get user prompt
#     prompt = st.text_input("Enter a prompt for image translation:", "Convert ")

#     if st.button("Generate"):
#         with st.spinner("Generating image... This may take a while on CPU."):
#             generated_image = generate_image(preprocessed_image, prompt)
            
#             # Display input and output images side by side
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.write("Input Image")
#                 st.image(input_image, use_column_width=True)
#             with col2:
#                 st.write("Generated Image")
#                 st.image(generated_image, use_column_width=True)
            
#             # Display numerical comparison
#             st.write("### Numerical Comparison")
#             input_stats = get_image_stats(input_image)
#             output_stats = get_image_stats(generated_image)
            
#             comparison_data = {
#                 "Metric": ["Mean", "Std", "Min", "Max"],
#                 "Input Image (R, G, B)": [
#                     f"{input_stats['Mean'][0]:.2f}, {input_stats['Mean'][1]:.2f}, {input_stats['Mean'][2]:.2f}",
#                     f"{input_stats['Std'][0]:.2f}, {input_stats['Std'][1]:.2f}, {input_stats['Std'][2]:.2f}",
#                     f"{input_stats['Min'][0]}, {input_stats['Min'][1]}, {input_stats['Min'][2]}",
#                     f"{input_stats['Max'][0]}, {input_stats['Max'][1]}, {input_stats['Max'][2]}"
#                 ],
#                 "Output Image (R, G, B)": [
#                     f"{output_stats['Mean'][0]:.2f}, {output_stats['Mean'][1]:.2f}, {output_stats['Mean'][2]:.2f}",
#                     f"{output_stats['Std'][0]:.2f}, {output_stats['Std'][1]:.2f}, {output_stats['Std'][2]:.2f}",
#                     f"{output_stats['Min'][0]}, {output_stats['Min'][1]}, {output_stats['Min'][2]}",
#                     f"{output_stats['Max'][0]}, {output_stats['Max'][1]}, {output_stats['Max'][2]}"
#                 ]
#             }
            
#             st.table(comparison_data)
            
#             # Display graphical comparison
#             st.write("### Graphical Comparison")
#             st.pyplot(plot_histograms(input_image, generated_image))

# st.write("Note: This application uses a pre-trained Stable Diffusion model for image-to-image translation. It's running on CPU, which may be slow.")

import streamlit as st
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from utils import preprocess_image
import numpy as np
import matplotlib.pyplot as plt

# Setting the title and sidebar
st.set_page_config(page_title="Image-to-Image Translation", layout="wide")
st.title("Image-to-Image Translation using Stable Diffusion")

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    return pipe

pipe = load_model()

def generate_image(input_image, prompt):
    with torch.no_grad():
        image = pipe(prompt=prompt, image=input_image, strength=0.75, guidance_scale=7.5).images[0]
    return image

def get_image_stats(image):
    img_array = np.array(image)
    return {
        "Mean": np.mean(img_array, axis=(0, 1)),
        "Std": np.std(img_array, axis=(0, 1)),
        "Min": np.min(img_array, axis=(0, 1)),
        "Max": np.max(img_array, axis=(0, 1))
    }

def plot_histograms(input_image, output_image):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for ax, img, title in zip([ax1, ax2], [input_image, output_image], ['Input Image', 'Output Image']):
        img_array = np.array(img)
        ax.hist(img_array[:,:,0].ravel(), bins=256, color='red', alpha=0.5, label='Red')
        ax.hist(img_array[:,:,1].ravel(), bins=256, color='green', alpha=0.5, label='Green')
        ax.hist(img_array[:,:,2].ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Count')
        ax.set_title(f'{title} Histogram')
        ax.legend()
    
    plt.tight_layout()
    return fig

# Sidebar for image upload and prompt input
st.sidebar.header("Input Options")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    st.sidebar.image(input_image, caption="Uploaded Image", use_column_width=True)

    preprocessed_image = preprocess_image(input_image)

    prompt = st.sidebar.text_input("Enter a prompt for image translation:", "Convert")

    if st.sidebar.button("Generate"):
        with st.spinner("Generating image... This may take a while on CPU."):
            generated_image = generate_image(preprocessed_image, prompt)
            
            st.subheader("Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(input_image, caption="Input Image", use_column_width=True)
            with col2:
                st.image(generated_image, caption="Generated Image", use_column_width=True)

            st.subheader("Numerical Comparison")
            input_stats = get_image_stats(input_image)
            output_stats = get_image_stats(generated_image)

            comparison_data = {
                "Metric": ["Mean", "Std", "Min", "Max"],
                "Input Image (R, G, B)": [
                    f"{input_stats['Mean'][0]:.2f}, {input_stats['Mean'][1]:.2f}, {input_stats['Mean'][2]:.2f}",
                    f"{input_stats['Std'][0]:.2f}, {input_stats['Std'][1]:.2f}, {input_stats['Std'][2]:.2f}",
                    f"{input_stats['Min'][0]}, {input_stats['Min'][1]}, {input_stats['Min'][2]}",
                    f"{input_stats['Max'][0]}, {input_stats['Max'][1]}, {input_stats['Max'][2]}"
                ],
                "Output Image (R, G, B)": [
                    f"{output_stats['Mean'][0]:.2f}, {output_stats['Mean'][1]:.2f}, {output_stats['Mean'][2]:.2f}",
                    f"{output_stats['Std'][0]:.2f}, {output_stats['Std'][1]:.2f}, {output_stats['Std'][2]:.2f}",
                    f"{output_stats['Min'][0]}, {output_stats['Min'][1]}, {output_stats['Min'][2]}",
                    f"{output_stats['Max'][0]}, {output_stats['Max'][1]}, {output_stats['Max'][2]}"
                ]
            }

            st.table(comparison_data)

            st.subheader("Graphical Comparison")
            st.pyplot(plot_histograms(input_image, generated_image))

st.sidebar.write("Note: This application uses a pre-trained Stable Diffusion model for image-to-image translation. It's running on CPU, which may be slow.")
