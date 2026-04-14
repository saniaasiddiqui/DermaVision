import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/api/analyze"

custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0b0f2b, #1a1040) !important;
    color: white !important;
}

h1 {
    color: #ffb6c1 !important;
    text-align: center;
}
p {
   color: #ffffff

}

#gradcam, .gradcam-container {
    background-color: white !important;
}
.progress-bar {
    background-color: #9aa0b5 !important;  /* grey instead of pink */
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose strong {
    color:#ffb6c1!important;
}

.gradio-container button.gr-button {
    background-color: #ffb6c1 !important;
    color: black !important;
    border-radius: 8px !important;
    border: none !important;
}

/* Hover */
.gradio-container button.gr-button:hover {
    background-color: #e89aa9 !important;
}
input,p{
       color: white !important;
}
 .gr-panel {
    background-color: #ffffff;
    border: 1px solid #2a2f6a;
}


.gr-textbox textarea,
.gr-textbox input,
textarea,
input {
    color: black !important;
    background-color: white !important;
    border: 1px solid #2a2f6a !important;
}

.gr-textbox input:hover,
.gr-textbox textarea:hover {
    background-color: grey !important;
}

.gr-image,
.gr-image-preview,
.output-image,
.gr-box {
    background-color: white !important;
}


label {
    color: #575555 !important;
}



/* FORCE submit button color */
/* ONLY Submit button */
button.primary {
    background-color: #ffb6c1 !important;
    color: black !important;
    border-radius: 8px !important;
    border: none !important;
}

/* Hover */
button.primary:hover {
    background-color: #e89aa9 !important;
}
"""

def predict(image):
    import io
    import base64
    from PIL import Image

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    response = requests.post(
        API_URL,
        files={"file": ("image.jpg", img_bytes, "image/jpeg")}
    )

    result = response.json()

    prediction = result["prediction_name"]
    confidence = result["confidence"]
    prob_dict = result["probabilities"]

    # Decode Grad-CAM image
    gradcam_base64 = result["gradcam"]
    gradcam_img = Image.open(io.BytesIO(base64.b64decode(gradcam_base64)))

    markdown_text = f"The disease is **{prediction}** with a confidence of **{confidence*100:.2f}%**."
    plain_text = f"{prediction} ({confidence*100:.2f}%)"

    return markdown_text, plain_text, prob_dict, gradcam_img
# UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Text(label="Prediction"),
        gr.Label(label="Class Probabilities"),
         gr.Image(label="Grad-CAM Visualization")  # ✅ NEW
    ],
   title="DermaVision – AI-Powered Skin Cancer Detection",
    description="""
Analyze skin images using advanced deep learning models trained on dermatological data.

Upload a clear image of a skin lesion to receive predicted classifications along with confidence scores.

The system also provides visual insights to help understand which areas influenced the prediction.

This tool is designed for early awareness and educational purposes only and is not a substitute for professional medical advice.
""",
    css=custom_css,
    theme=gr.themes.Base() 

)

interface.launch()