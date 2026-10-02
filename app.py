import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# -----------------------------
# Page
# -----------------------------

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠"
)

st.title("🧠 Brain Tumor Classifier")
st.write("Upload a brain MRI image for classification.")


# -----------------------------
# Normal Model Architecture
# -----------------------------

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 62 * 62, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Quantized Model Architecture
# -----------------------------

class QuantizedCNN(nn.Module):

    def __init__(self, num_classes=4):
        super(QuantizedCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.quantized.dynamic.Linear(
                32 * 62 * 62,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.quantized.dynamic.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Classes
# -----------------------------

class_names = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]


# -----------------------------
# Device
# -----------------------------

device = torch.device("cpu")


# -----------------------------
# Load Quantized Model
# -----------------------------

@st.cache_resource
def load_model():

    model = QuantizedCNN(num_classes=4)

    checkpoint = torch.load(
        "model.pth",
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    checkpoint = {
        key.replace("module.", ""): value
        for key, value in checkpoint.items()
    }

    model.load_state_dict(checkpoint)

    model.eval()

    return model


# -----------------------------
# Image Transformation
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((250, 250)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        width=400
    )

    if st.button("Predict"):

        try:

            model = load_model()

            image_tensor = transform(image)

            image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(device)

            with torch.no_grad():

                output = model(image_tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )

                confidence, predicted_class = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = predicted_class.item()

            confidence = confidence.item()

            result = class_names[predicted_class]

            st.success(
                f"Prediction: "
                f"{result.replace('_', ' ').title()}"
            )

            st.write(
                f"Confidence: {confidence * 100:.2f}%"
            )

            # -----------------------------
            # Probabilities
            # -----------------------------

            st.subheader("Class Probabilities")

            for i, class_name in enumerate(class_names):

                probability = probabilities[0][i].item()

                st.write(
                    f"{class_name.replace('_', ' ').title()}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("Model Information")

st.sidebar.write(
    "Architecture: Quantized SimpleCNN"
)

st.sidebar.write(
    "Input: 250 × 250"
)

st.sidebar.write(
    "Classes: 4"
)

st.sidebar.write(
    "Device: CPU"
)

st.sidebar.warning(
    "This application is for educational purposes "
    "and is not a medical diagnosis system."
)
