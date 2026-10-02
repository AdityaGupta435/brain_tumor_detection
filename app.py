import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero Section */

.hero {
    padding: 32px;
    border-radius: 22px;
    background: linear-gradient(
        135deg,
        #e8f1ff,
        #ffffff
    );
    border: 1px solid #d8e3f0;
    margin-bottom: 30px;
}

.hero h1 {
    color: #16324f;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #64748b;
    font-size: 17px;
    margin-bottom: 0;
}


/* Result Card */

.result-card {
    padding: 28px;
    border-radius: 20px;
    background: white;
    border: 1px solid #e2e8f0;
    box-shadow: 0 6px 25px rgba(0, 0, 0, 0.06);
}

.prediction-label {
    color: #64748b;
    font-size: 15px;
    margin-bottom: 5px;
}

.prediction {
    color: #17324d;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 20px;
}

.confidence-label {
    color: #64748b;
    font-size: 15px;
}

.confidence {
    color: #17324d;
    font-size: 27px;
    font-weight: 700;
}


/* Info Cards */

.info-card {
    padding: 20px;
    border-radius: 16px;
    background: white;
    border: 1px solid #e2e8f0;
    height: 100%;
}


/* Warning */

.warning-box {
    padding: 18px;
    border-radius: 14px;
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #7c2d12;
}


/* Button */

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL ARCHITECTURE
# =========================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes=4):

        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(

            # First convolution
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),


            # Second convolution
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
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

            # 250x250
            # 250 / 2 = 125
            # 125 / 2 = 62
            #
            # 32 x 62 x 62 = 123008

            nn.Linear(
                32 * 62 * 62,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            # Four classes
            nn.Linear(
                128,
                num_classes
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_PATH = "model.pth"


# IMPORTANT:
# This order MUST exactly match full_dataset.classes

CLASS_NAMES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor"
]


# Class mapping

CLASS_TO_INDEX = {
    "glioma_tumor": 0,
    "meningioma_tumor": 1,
    "no_tumor": 2,
    "pituitary_tumor": 3
}


# =========================================================
# DEVICE
# =========================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    # Create model with 4 classes

    model = SimpleCNN(
        num_classes=4
    )


    # Load trained weights

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    # -----------------------------------------------------
    # Case 1:
    # torch.save(model.state_dict(), "model.pth")
    # -----------------------------------------------------

    if isinstance(checkpoint, dict):

        # Handle checkpoint format
        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        else:

            state_dict = checkpoint


        # Remove "module." if model was trained
        # using DataParallel

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith("module."):

                new_key = new_key[
                    len("module.") :
                ]

            cleaned_state_dict[
                new_key
            ] = value


        model.load_state_dict(
            cleaned_state_dict
        )


    # -----------------------------------------------------
    # Case 2:
    # Complete model was saved
    # -----------------------------------------------------

    else:

        model = checkpoint


    # Move model to GPU/CPU

    model = model.to(
        DEVICE
    )


    # Evaluation mode

    model.eval()


    return model


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">

<h1>🧠 Brain Tumor Image Classifier</h1>

<p>
Upload a brain MRI image and the trained CNN model
will classify it into one of four categories.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Model Information")


    st.write("### Architecture")

    st.write(
        "Simple CNN"
    )


    st.write(
        "Input Size: 250 × 250"
    )


    st.write(
        "Number of Classes: 4"
    )


    st.write(
        f"Device: {DEVICE}"
    )


    st.divider()


    st.write("### 🏷️ Classes")


    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        st.write(
            f"**{index}** → {class_name}"
        )


    st.divider()


    st.write("### 🧠 Model Pipeline")

    st.write(
        "Image → Resize → Normalize → CNN → Prediction"
    )


    st.divider()


    st.warning(
        "This application is an educational "
        "machine-learning demonstration and "
        "must not be used as a medical diagnosis."
    )


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

except FileNotFoundError:

    st.error(
        "❌ model.pth was not found."
    )

    st.info(
        "Please put model.pth in the same folder "
        "as app.py."
    )

    st.stop()


except Exception as e:

    st.error(
        "❌ Failed to load the model."
    )

    st.code(
        str(e)
    )

    st.stop()


# =========================================================
# UPLOAD SECTION
# =========================================================

st.subheader(
    "📤 Upload Brain MRI Image"
)


uploaded_file = st.file_uploader(

    "Choose a JPG, JPEG or PNG image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ],

    help=(
        "The uploaded image will be resized "
        "to 250 × 250 before prediction."
    )
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:


    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # -----------------------------------------------------
    # Image preprocessing
    #
    # MUST match training preprocessing
    # -----------------------------------------------------

    test_transforms = transforms.Compose([

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


    # -----------------------------------------------------
    # Transform image
    # -----------------------------------------------------

    input_tensor = test_transforms(
        image
    )


    # Add batch dimension

    input_batch = input_tensor.unsqueeze(
        0
    )


    # Move to device

    input_batch = input_batch.to(
        DEVICE
    )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with st.spinner(
        "🔍 Analyzing MRI image..."
    ):

        with torch.no_grad():

            output = model(
                input_batch
            )


            # Convert logits to probabilities

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]


    # -----------------------------------------------------
    # Get predicted class
    # -----------------------------------------------------

    predicted_index = torch.argmax(
        probabilities
    ).item()


    predicted_class = CLASS_NAMES[
        predicted_index
    ]


    confidence = (
        probabilities[
            predicted_index
        ].item()
        * 100
    )


    # =====================================================
    # RESULT SECTION
    # =====================================================

    st.divider()


    col1, col2 = st.columns(
        [1, 1]
    )


    # -----------------------------------------------------
    # Uploaded image
    # -----------------------------------------------------

    with col1:

        st.subheader(
            "🖼️ Uploaded MRI"
        )


        st.image(
            image,
            use_container_width=True
        )


        st.caption(
            f"Original image size: "
            f"{image.size[0]} × {image.size[1]}"
        )


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with col2:

        st.subheader(
            "📊 Prediction"
        )


        st.markdown(
            f"""
            <div class="result-card">

                <div class="prediction-label">
                    Predicted Class
                </div>

                <div class="prediction">
                    {predicted_class}
                </div>

                <div class="confidence-label">
                    Model Confidence
                </div>

                <div class="confidence">
                    {confidence:.2f}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        # Confidence progress

        st.progress(
            min(
                max(
                    confidence / 100,
                    0.0
                ),
                1.0
            )
        )


        # Confidence message

        if confidence >= 80:

            st.success(
                "High model confidence"
            )

        elif confidence >= 60:

            st.warning(
                "Moderate model confidence"
            )

        else:

            st.info(
                "Low model confidence — "
                "interpret the prediction cautiously."
            )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    st.divider()


    st.subheader(
        "📈 Class Probabilities"
    )


    probability_data = {}


    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability_data[
            class_name
        ] = float(
            probabilities[index].item()
        )


    # Display chart

    st.bar_chart(
        probability_data
    )


    # Display exact percentages

    st.write(
        "### Probability Details"
    )


    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            probabilities[index].item()
            * 100
        )


        st.write(
            f"**{class_name}** — "
            f"{probability:.2f}%"
        )


    # =====================================================
    # MODEL DETAILS
    # =====================================================

    st.divider()


    st.subheader(
        "🔬 Model Details"
    )


    info1, info2, info3 = st.columns(
        3
    )


    with info1:

        st.markdown(
            """
            <div class="info-card">

            <b>Input Image</b>

            <br><br>

            RGB

            <br>

            250 × 250 pixels

            </div>
            """,
            unsafe_allow_html=True
        )


    with info2:

        st.markdown(
            """
            <div class="info-card">

            <b>Architecture</b>

            <br><br>

            2 Convolution Layers

            <br>

            2 MaxPool Layers

            </div>
            """,
            unsafe_allow_html=True
        )


    with info3:

        st.markdown(
            """
            <div class="info-card">

            <b>Output</b>

            <br><br>

            4 Classes

            <br>

            Softmax Probability

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # MEDICAL DISCLAIMER
    # =====================================================

    st.divider()


    st.markdown(
        """
        <div class="warning-box">

        ⚠️ <b>Important Medical Disclaimer</b>

        <br><br>

        This application is developed for educational,
        research, and demonstration purposes.

        <br><br>

        The prediction generated by this model is
        <b>not a medical diagnosis</b> and should not
        be used as the sole basis for any medical
        decision.

        <br><br>

        Always consult a qualified medical professional
        for clinical evaluation and diagnosis.

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# NO IMAGE UPLOADED
# =========================================================

else:

    st.info(
        "👆 Upload an MRI image above to start prediction."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Brain Tumor CNN Classifier • "
    "Educational Machine Learning Project"
)
