from fastapi import FastAPI, UploadFile, File
import torch
import cv2
import numpy as np
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import io
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Welcome to DermaVision API!"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create EfficientNet model
model = models.efficientnet_b3(weights=None)

# Modify classifier for 7 skin classes
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    7
)

# Load trained weights
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "model", "skin_cancer_efficientnet_b3.pth")

model.load_state_dict(
    torch.load(model_path, map_location=device)
)

model.to(device)
model.eval()

classes = ['akiec','bcc','bkl','df','mel','nv','vasc']

CLASS_NAMES = {
    'nv': 'Melanocytic Nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses and Intraepithelial Carcinoma',
    'vasc': 'Vascular Lesions',
    'df': 'Dermatofibroma'
}

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

gradients = None
activations = None

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

def save_activation(module, input, output):
    global activations
    activations = output

# Hook the last convolution layer
target_layer = model.features[-1]
target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, 1)[0]

    predicted = torch.argmax(probs).item()
    predicted_class = classes[predicted]

    # Backward pass for Grad-CAM
    model.zero_grad()
    outputs[0, predicted].backward()

    # Generate Grad-CAM
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activation = activations[0]

    for i in range(len(pooled_gradients)):
        activation[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Convert original image to numpy
    img_np = np.array(image.resize((224, 224)))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    # Convert to base64
    import base64
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "prediction": predicted_class,
        "prediction_name": CLASS_NAMES[predicted_class],
        "confidence": float(probs[predicted]),
        "probabilities": {
            CLASS_NAMES[classes[i]]: float(probs[i]) for i in range(len(classes))
        },
        "gradcam": gradcam_base64   # ✅ NEW
    }
