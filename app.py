import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠"
)

st.title("🧠 Brain Tumor Classifier")
st.write("Upload a brain MRI image to classify it.")


# =====================================================
# MODEL
# =====================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=4):

        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 62 * 62,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# =====================================================
# CLASS NAMES
# =====================================================

class_names = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]


# =====================================================
# DEVICE
# =====================================================

device = torch.device("cpu")


# =====================================================
# LOAD QUANTIZED MODEL
# =====================================================

@st.cache_resource
def load_model():

    # Create original float model
    model = SimpleCNN(num_classes=4)

    # Convert Linear layers to dynamic quantized Linear
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    # Load saved quantized state_dict
    checkpoint = torch.load(
        "model.pth",
        map_location="cpu",
        weights_only=False
    )

    # If checkpoint contains model_state_dict
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            checkpoint = checkpoint["model_state_dict"]

    # Remove DataParallel prefix if present
    cleaned_checkpoint = {}

    for key, value in checkpoint.items():

        new_key = key

        if new_key.startswith("module."):

            new_key = new_key.replace(
                "module.",
                "",
                1
            )

        cleaned_checkpoint[new_key] = value

    # Load quantized weights
    quantized_model.load_state_dict(
        cleaned_checkpoint,
        strict=True
    )

    quantized_model.eval()

    return quantized_model


# =====================================================
# IMAGE TRANSFORMATION
# =====================================================

transform = transforms.Compose([

    transforms.Resize(
        (250, 250)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# =====================================================
# IMAGE UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI Image",
        width=400
    )

    if st.button("Predict"):

        try:

            # Load model
            model = load_model()

            # Transform image
            image_tensor = transform(image)

            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)

            # CPU
            image_tensor = image_tensor.to(device)

            # Prediction
            with torch.no_grad():

                output = model(
                    image_tensor
                )

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

            result = class_names[
                predicted_class
            ]

            # Result
            st.success(
                "Prediction: "
                + result.replace(
                    "_",
                    " "
                ).title()
            )

            st.write(
                f"Confidence: "
                f"{confidence * 100:.2f}%"
            )

            # -----------------------------------------
            # ALL CLASS PROBABILITIES
            # -----------------------------------------

            st.subheader(
                "Class Probabilities"
            )

            for i, class_name in enumerate(
                class_names
            ):

                probability = probabilities[
                    0
                ][i].item()

                st.write(
                    f"{class_name.replace('_', ' ').title()}: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    probability
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title(
    "Model Information"
)

st.sidebar.write(
    "Architecture: SimpleCNN"
)

st.sidebar.write(
    "Quantization: Dynamic INT8"
)

st.sidebar.write(
    "Input Size: 250 × 250"
)

st.sidebar.write(
    "Classes: 4"
)

st.sidebar.write(
    "Device: CPU"
)

st.sidebar.warning(
    "This application is for educational "
    "purposes only and is not a medical "
    "diagnosis system."
)
