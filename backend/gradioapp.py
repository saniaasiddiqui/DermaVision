import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/api/analyze"

def predict(image):
    import io
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    response = requests.post(
        API_URL,
        files={"file": ("image.jpg", img_bytes, "image/jpeg")}
    )

    result = response.json()

    prediction = result["prediction"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    classes = ['akiec','bcc','bkl','df','mel','nv','vasc']

    # Convert list → dictionary
    prob_dict = {classes[i]: probabilities[i] for i in range(len(classes))}

    return f"{prediction} ({confidence*100:.2f}%)", prob_dict

# UI
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Label(label="Class Probabilities")
    ],
    title="Dermavision - Skin Cancer Detection",
    description="Upload a skin image to analyze using AI"
)

interface.launch()