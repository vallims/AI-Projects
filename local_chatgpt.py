import streamlit as st
import ollama
from PIL import Image
import base64
import io

ollama_url = "http://10.24.6.153:8000"

client = ollama.Client(host=ollama_url)

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "Hello, I'm your local ChatGPT running on Llama3.2-Vision. How can I help you?"}
        ]

def display_chat_history():
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

def get_ollama_response(prompt, image=None):
    messages = st.session_state.messages.copy()
    if image:
        image_content = image_to_base64(image)
        messages.append({"role": "user", "content": prompt, "images": [image_content]})
    else:
        messages.append({"role": "user", "content": prompt})
    
    #response = ollama.chat(model="llama3.2-vision", messages=messages)
    response = client.chat(model="llama3.2-vision", messages=messages)
    return response.message.content

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('ascii')
    

def main():
    st.title("Local ChatGPT with Llama3.2-Vision")

    initialize_chat_history()
    display_chat_history()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()
    
        # Use BytesIO to convert bytes to a file-like object
        image = Image.open(io.BytesIO(bytes_data))
        st.image(image, caption="Uploaded Image", use_container_width=True)

    prompt = st.chat_input("Enter your question:")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = get_ollama_response(prompt, image if uploaded_file else None)
                st.write(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
