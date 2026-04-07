import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Underwater Vision AI", layout="wide")

# -------------------- 🌊 CUSTOM UI --------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #2c5364);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    border: none;
    font-weight: bold;
}

.card {
    background-color: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- 🌊 HERO HEADER --------------------

st.markdown("""
<div style='text-align:center; padding:20px;'>
    <h1 style='font-size:48px;'>🌊 Underwater Vision AI</h1>
    <p style='font-size:18px; color:#cce7ff;'>
        Real-time enhancement system for underwater surveillance and visibility analysis
    </p>
</div>
""", unsafe_allow_html=True)

# ✨ Soft divider
st.markdown("""
<hr style='border: 1px solid rgba(255,255,255,0.1); margin: 30px 0;'>
""", unsafe_allow_html=True)

# -------------------- 📤 UPLOAD SECTION --------------------

st.markdown("""
<div class='card'>
    <h3>📡 Upload Underwater Image</h3>
    <p style='color:#a8d8ff;'>Input raw underwater footage for enhancement and analysis</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

st.markdown(
    "<p style='text-align:center; color:#7fb3d5;'>Supported formats: JPG, PNG • Real-time processing enabled</p>",
    unsafe_allow_html=True
)

# -------------------- Enhancement Methods --------------------

def apply_clahe(image, strength):
    img = np.array(image)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=strength, tileGridSize=(8,8))
    cl = clahe.apply(l)

    enhanced_lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def white_balance(image):
    img = np.array(image).astype(np.float32)
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    avg_a = np.mean(result[:,:,1])
    avg_b = np.mean(result[:,:,2])

    result[:,:,1] -= (avg_a - 128) * (result[:,:,0] / 255.0) * 1.1
    result[:,:,2] -= (avg_b - 128) * (result[:,:,0] / 255.0) * 1.1

    result = np.clip(result, 0, 255).astype(np.uint8)

    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


def sharpen(image):
    img = np.array(image)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


# -------------------- Metrics --------------------

def calculate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray.std()


# -------------------- Auto Mode --------------------

def auto_enhance(image):
    results = {}

    clahe_img = apply_clahe(image, 3.0)
    wb_img = white_balance(image)
    sharp_img = sharpen(image)

    results["CLAHE"] = clahe_img
    results["White Balance"] = wb_img
    results["Sharpen"] = sharp_img

    best_method = None
    best_score = 0
    best_image = None

    for method, img in results.items():
        score = calculate_contrast(img)
        if score > best_score:
            best_score = score
            best_method = method
            best_image = img

    return best_image, best_method, best_score


# -------------------- UI --------------------

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("### ⚙️ Control Panel")

    method = st.selectbox(
        "Select Enhancement Mode",
        ["Manual Mode", "✨ Auto AI Mode"]
    )

    if method == "Manual Mode":
        option = st.selectbox(
            "Choose Method",
            ["CLAHE (OpenCV Contrast)", "LAB Color Correction", "Spatial Filtering (Sharpening)"]
        )
        strength = st.slider("Enhancement Strength (CLAHE only)", 1.0, 5.0, 3.0)

    process = st.button("🚀 Initiate Scan")

    st.markdown("</div>", unsafe_allow_html=True)

    if process:

        import time

        status = st.empty()

        status.info("🔍 Initializing underwater scan...")
        time.sleep(1)

        status.info("🌊 Adjusting light absorption...")
        time.sleep(1)

        status.info("🧠 Enhancing image using AI pipeline...")
        time.sleep(1)

        status.info("📡 Finalizing output...")
        time.sleep(1)

        status.success("✅ Scan Complete")

        original_contrast = calculate_contrast(img_array)

        if method == "✨ Auto AI Mode":
            enhanced, best_method, best_score = auto_enhance(image)
        else:
            if option == "CLAHE (OpenCV Contrast)":
                enhanced = apply_clahe(image, strength)
                best_method = "CLAHE"

            elif option == "LAB Color Correction":
                enhanced = white_balance(image)
                best_method = "White Balance"

            elif option == "Spatial Filtering (Sharpening)":
                enhanced = sharpen(image)
                best_method = "Sharpening"

            best_score = calculate_contrast(enhanced)

        # 🔥 RESULT SECTION

        st.markdown("## 🔍 Scan Results")
        st.markdown(
            "<p style='color:#a8d8ff;'>Enhanced visibility achieved using OpenCV-based pipeline</p>",
            unsafe_allow_html=True
        )

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📥 Original Image")
            st.image(image, width="stretch")
            st.write(f"Contrast: {original_contrast:.2f}")

        with col2:
            st.subheader("📤 Enhanced Image")
            st.image(enhanced, width="stretch")
            st.write(f"Contrast: {best_score:.2f}")

        st.markdown("---")

        st.success(f"✅ Best Method Used: {best_method}")

        improvement = best_score - original_contrast
        st.info(f"📈 Contrast Improvement: {improvement:.2f}")

        result = Image.fromarray(enhanced)
        st.download_button(
            "⬇ Download Enhanced Image",
            data=result.tobytes(),
            file_name="enhanced.png"
        )

        st.markdown("</div>", unsafe_allow_html=True)