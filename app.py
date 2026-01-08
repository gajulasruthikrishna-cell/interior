import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import numpy as np
import cv2
import torch
import glob
from PIL import Image

from transformers import (
    OneFormerProcessor,
    OneFormerForUniversalSegmentation,
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    AutoModelForDepthEstimation
)

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ONEFORMER_MODEL = "shi-labs/oneformer_ade20k_swin_large"
MASK2FORMER_MODEL = "facebook/mask2former-swin-large-coco-instance"
DEPTH_MODEL = "Intel/dpt-hybrid-midas"

ALPHA = 0.25

VALID_INTERIOR_CLASSES = {
    "wall", "floor", "door",
    "cabinet", "counter",
    "countertop", "island"
}

# 🔴 EXCLUDE ATTACHED OBJECTS (handles, fixtures, etc.)
EXCLUDE_ATTACHED_OBJECTS = {
    "door handle",
    "handle",
    "knob",
    "faucet",
    "sink",
    "microwave",
    "oven",
    "refrigerator",
    "switch",
    "outlet",
    "light",
    "vase",
    "bottle"
}

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    oneformer_processor = OneFormerProcessor.from_pretrained(ONEFORMER_MODEL)
    oneformer_model = OneFormerForUniversalSegmentation.from_pretrained(
        ONEFORMER_MODEL
    ).to(DEVICE).eval()

    mask_processor = AutoImageProcessor.from_pretrained(MASK2FORMER_MODEL)
    mask_model = Mask2FormerForUniversalSegmentation.from_pretrained(
        MASK2FORMER_MODEL
    ).to(DEVICE).eval()

    depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL
    ).to(DEVICE).eval()

    return (
        oneformer_processor,
        oneformer_model,
        mask_processor,
        mask_model,
        depth_processor,
        depth_model
    )

# ================= SEMANTIC =================
def oneformer_semantic(image, processor, model):
    inputs = processor(
        images=image,
        task_inputs=["semantic"],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    seg_map = processor.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0].cpu().numpy()

    return seg_map, model.config.id2label

# ================= INSTANCE =================
def mask2former_instances(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    return processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0]

# ================= INSTANCE SUBTRACTION (UPDATED) =================
def subtract_instances(mask, instances, remove_labels, id2label):
    cleaned = mask.copy()
    inst_map = instances["segmentation"].cpu().numpy()

    for seg in instances["segments_info"]:
        if seg.get("score", 1.0) < 0.65:
            continue

        label_name = id2label.get(seg["label_id"], "").lower()

        if (
            label_name in remove_labels
            or any(x in label_name for x in EXCLUDE_ATTACHED_OBJECTS)
        ):
            cleaned[inst_map == seg["id"]] = 0

    return cleaned

# ================= DEPTH REFINEMENT =================
def depth_refine(mask, image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth

    depth = depth.squeeze().cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    depth = cv2.resize(
        depth,
        (mask.shape[1], mask.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    grad = cv2.Laplacian(depth, cv2.CV_8U)
    refined = mask.copy()
    refined[grad > 20] = 0
    return refined

# ================= MORPH CLEANUP =================
def geometric_cleanup(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 7)
    return mask

# ================= REMOVE SMALL PARTS (HANDLES ETC.) =================
def remove_small_parts(mask, min_area=800):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    return cleaned
def remove_thin_attached_parts(mask):
    """
    Removes thin elongated structures like handles, rails, trims.
    """
    cleaned = mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 300:
            continue

        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

        # 🔥 HANDLE RULE
        if aspect_ratio > 6 and min(w, h) < 12:
            cleaned[labels == i] = 0

    return cleaned


# ================= TEXTURES =================
def load_textures(folder="textures"):
    textures = {}
    for f in glob.glob(os.path.join(folder, "*.*")):
        name = os.path.splitext(os.path.basename(f))[0]
        tex = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        textures[name] = tex
    return textures

def apply_texture(image, mask, texture, alpha=0.9):
    h, w = image.shape[:2]
    tex = cv2.resize(texture, (w, h), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    shading = np.dstack([gray, gray, gray])

    tex = tex.astype(np.float32) / 255.0
    tex = tex * shading

    result = image.astype(np.float32) / 255.0
    m = (mask > 0)[:, :, None]
    result[m] = (1 - alpha) * result[m] + alpha * tex[m]

    return (result * 255).astype(np.uint8)

# ================= STREAMLIT =================
st.set_page_config(layout="wide")
st.title("🏠 Interior Segmentation + Wall Textures")

st.info(
    "Note: Handles, knobs, fixtures, appliances, and small attached objects "
    "are automatically excluded for cleaner surface detection."
)

uploaded = st.file_uploader(
    "Upload interior image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original", use_container_width=True)

    with st.spinner("Running pipeline..."):
        (
            oneformer_processor,
            oneformer_model,
            mask_processor,
            mask_model,
            depth_processor,
            depth_model
        ) = load_models()

        seg_map, id2label = oneformer_semantic(
            image, oneformer_processor, oneformer_model
        )

        instances = mask2former_instances(
            image, mask_processor, mask_model
        )

    detected = sorted({
        id2label[int(cid)]
        for cid in np.unique(seg_map)
        if id2label.get(int(cid)) in VALID_INTERIOR_CLASSES
    })

    if not detected:
        st.warning("No supported interior objects detected.")
        st.stop()

    selected = st.selectbox("Select object", detected)

    mask = np.zeros_like(seg_map, dtype=np.uint8)
    for cid, name in id2label.items():
        if name == selected:
            mask |= (seg_map == cid).astype(np.uint8)
    mask *= 255

    if selected in {"wall", "floor"}:
        REMOVE = {
            "door", "window",
            "cabinet", "counter",
            "countertop", "island"
        }
        mask = subtract_instances(
            mask, instances, REMOVE, mask_model.config.id2label
        )

    # 🔥 FINAL REFINEMENT PIPELINE
    mask = depth_refine(mask, image, depth_processor, depth_model)
    mask = geometric_cleanup(mask)
    mask = remove_thin_attached_parts(mask)  # 🔥 NEW
    mask = remove_small_parts(mask)


    textures = load_textures("textures")

    if selected == "wall" and textures:
        tex_name = st.selectbox("Select wall texture", list(textures.keys()))
        output = apply_texture(
            np.array(image),
            mask,
            textures[tex_name]
        )
    else:
        overlay = np.array(image).astype(np.float32)
        color = np.array([0, 255, 0], dtype=np.float32)
        overlay[mask > 0] = (1 - ALPHA) * overlay[mask > 0] + ALPHA * color
        output = overlay.astype(np.uint8)

    st.image(
        output,
        caption=f"Result: {selected}",
        use_container_width=True
    )
