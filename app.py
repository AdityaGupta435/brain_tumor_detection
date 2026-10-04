import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)


# --------------------------------------------------
# Class Names
# --------------------------------------------------

class_names = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]


# --------------------------------------------------
# Compatible Model
# --------------------------------------------------

class CompatibleCNN(nn.Module):

    def __init__(self, state_dict):
        super().__init__()

        # Convolution layers
        self.register_buffer(
            "conv1_weight",
            state_dict["features.0.weight"]
        )

        self.register_buffer(
            "conv1_bias",
            state_dict["features.0.bias"]
        )

        self.register_buffer(
            "conv2_weight",
            state_dict["features.3.weight"]
        )

        self.register_buffer(
            "conv2_bias",
            state_dict["features.3.bias"]
        )

        # Quantized Linear 1
        linear1_params = state_dict[
            "classifier.1._packed_params._packed_params"
        ]

        self.register_buffer(
            "linear1_weight",
            linear1_params[0]
        )

        self.register_buffer(
            "linear1_bias",
            linear1_params[1]
        )

        # Quantized Linear 2
        linear2_params = state_dict[
            "classifier.4._packed_params._packed_params"
        ]

        self.register_buffer(
            "linear2_weight",
            linear2_params[0]
        )

        self.register_buffer(
            "linear2_bias",
            linear2_params[1]
        )


    def forward(self, x):

        # Conv 1
        x = F.conv2d(
            x,
            self.conv1_weight,
            self.conv1_bias,
            padding=1
        )

        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Conv 2
        x = F.conv2d(
            x,
            self.conv2_weight,
            self.conv2_bias,
            padding=1
        )

        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Flatten
        x = torch.flatten(x, 1)

        # Linear 1
        x = F.linear(
            x,
            self.linear1_weight.dequantize(),
            self.linear1_bias
        )

        x = F.relu(x)

        # Linear 2
        x = F.linear(
            x,
            self.linear2_weight.dequantize(),
            self.linear2_bias
        )

        return x


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():

    device = torch.device("cpu")

    state_dict = torch.load(
        "model.pth",
        map_location=device,
        weights_only=False
    )

    model = CompatibleCNN(state_dict)

    model.to(device)
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🧠 Brain Tumor Classifier")

st.write(
    "Upload a brain MRI image to classify the tumor category."
)


uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        width="stretch"
    )

    st.write("")

    if st.button("🔍 Predict", type="primary"):

        with st.spinner("Analyzing MRI image..."):

            image_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():

                output = model(image_tensor)

                probabilities = torch.softmax(
                    output,
                    dim=1
                )[0]

                predicted_index = torch.argmax(
                    probabilities
                ).item()

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = probabilities[
                    predicted_index
                ].item() * 100


        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        st.subheader("Prediction")

        st.success(
            f"Prediction: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


        # --------------------------------------------------
        # Probabilities
        # --------------------------------------------------

        st.subheader("Class Probabilities")

        for i, class_name in enumerate(class_names):

            probability = probabilities[i].item()

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(probability)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Model Information")

    st.write("**Model:** SimpleCNN")
    st.write("**Classes:** 4")
    st.write("**Input Size:** 250 × 250")
    st.write("**Inference:** CPU")
    st.write("**Model Type:** Quantized CNN")

    st.divider()

    st.warning(
        "This application is for educational and "
        "research purposes only. It should not be "
        "used as a medical diagnosis."
    )
