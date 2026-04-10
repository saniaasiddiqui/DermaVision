from fastapi import FastAPI, UploadFile, File
import torch
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

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs,1)[0]

    predicted = torch.argmax(probs).item()

    return {
        "prediction": classes[predicted],
        "confidence": float(probs[predicted]),
        "probabilities": probs.tolist()
    }