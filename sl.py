import os
import io
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Google Generative AI SDK
import google.generativeai as genai

# ------------- Config & Secrets -------------
st.set_page_config(page_title="Gemini + Streamlit", page_icon="✨")
load_dotenv()  # loads .env for local dev

# Prefer Streamlit secrets, else environment variable
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not API_KEY:
    st.error(
        "No API key found. Set GOOGLE_API_KEY in .streamlit/secrets.toml or .env"
    )
    st.stop()

genai.configure(api_key=API_KEY)

# Choose a Gemini model
# Text/chat: "gemini-1.5-flash" (fast) or "gemini-1.5-pro" (more capable)
# Vision/multimodal: use same 1.5 models; they accept images in the prompt
MODEL_NAME = "gemini-1.5-flash"

# Model generation config
generation_config = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 2048,
}

# Safety settings (optional): see Google docs for categories and thresholds
safety_settings = {
    # Example: lower or raise as needed
    # "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE"
}

model = genai.GenerativeModel(
    model_name=MODEL_NAME,
    generation_config=generation_config,
    safety_settings=safety_settings or None,
)

# ------------- UI -------------
st.title("✨ Gemini (AI Studio) + Streamlit")
st.caption("Chat with Gemini using your AI Studio API key. Supports text and (optional) image prompt.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything or upload an image to discuss."}
    ]

# Sidebar: options
with st.sidebar:
    st.subheader("Settings")
    model_choice = st.selectbox(
        "Model",
        ["gemini-1.5-flash", "gemini-1.5-pro"],
        index=0 if MODEL_NAME.endswith("flash") else 1,
    )
    if model_choice != MODEL_NAME:
        MODEL_NAME = model_choice
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=generation_config,
            safety_settings=safety_settings or None,
        )
    streaming = st.toggle("Stream responses", value=True)
    allow_images = st.toggle("Allow image input", value=True)

    st.markdown("---")
    st.markdown("**Tip:** Keep your API key safe—don’t hardcode it in your app.")

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Image uploader (optional multimodal)
uploaded_image = None
if allow_images:
    uploaded_image = st.file_uploader(
        "Upload an image (optional)", type=["png", "jpg", "jpeg"]
    )
    if uploaded_image:
        # Preview the image
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Image to include in your prompt", use_container_width=True)

# Chat input
user_input = st.chat_input("Type your message…")

def _image_part_from_pil(pil_image: Image.Image):
    """Convert a PIL Image into a Gemini-compatible inline data part."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()
    return {
        "mime_type": "image/jpeg",
        "data": data,
    }

if user_input:
    # Append user message to UI
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Build prompt parts (text + optional image)
    parts = [user_input]
    if allow_images and uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        parts.append(_image_part_from_pil(img))

    # Generate response
    with st.chat_message("assistant"):
        if streaming:
            # Stream the tokens
            placeholder = st.empty()
            partial_text = ""
            try:
                # For chat-like behavior, start a chat and send message
                chat_session = model.start_chat(history=[])
                stream = chat_session.send_message(parts, stream=True)

                for chunk in stream:
                    if chunk and chunk.text:
                        partial_text += chunk.text
                        placeholder.markdown(partial_text)
                # Final text
                assistant_text = partial_text.strip() or "(No content received.)"
            except Exception as e:
                assistant_text = f"⚠️ Error: {e}"
                st.error(assistant_text)
        else:
            try:
                # Single-shot generation
                response = model.generate_content(parts)
                assistant_text = (response.text or "").strip() or "(No content received.)"
                st.markdown(assistant_text)
            except Exception as e:
                assistant_text = f"⚠️ Error: {e}"
                st.error(assistant_text)

    # Save assistant message to history
    st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
