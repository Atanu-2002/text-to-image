import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import io

@st.cache_resource
def load_model():
    model_id = "Yntec/AnimephilesAnonymous"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,  # 👈 use float32 for CPU
        safety_checker=None         # Optional: disable NSFW checker
    )
    pipe = pipe.to("cpu")
    return pipe

st.title("🎨 Text to Anime Image Generator By Atanu Mondal")
st.markdown("Generate anime-style images using prompts and AI!")

prompt = st.text_input("Enter your prompt", "cute anime girl with blue hair, smiling")

if st.button("Generate"):
    with st.spinner("Generating image... (this might take a minute on CPU)"):
        pipe = load_model()
        image = pipe(prompt).images[0]

        # Show image
        st.image(image, caption="Generated Image", use_column_width=True)

        # Download
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        st.download_button("📥 Download Image", buf.getvalue(), "anime_image.png", "image/png")
