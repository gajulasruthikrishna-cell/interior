import cv2
import torch
import numpy as np
from PIL import Image
from functools import lru_cache

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

EXCLUDE_ATTACHED_OBJECTS = {
    "handle", "knob", "faucet", "sink",
    "microwave", "oven", "refrigerator",
    "switch", "outlet", "light"
}

# ================= LOAD MODELS =================
@lru_cache(maxsize=1)
def load_models():
    oneformer_p = OneFormerProcessor.from_pretrained(ONEFORMER_MODEL)
    oneformer_m = OneFormerForUniversalSegmentation.from_pretrained(
        ONEFORMER_MODEL
    ).to(DEVICE).eval()

    mask_p = AutoImageProcessor.from_pretrained(MASK2FORMER_MODEL)
    mask_m = Mask2FormerForUniversalSegmentation.from_pretrained(
        MASK2FORMER_MODEL
    ).to(DEVICE).eval()

    depth_p = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    depth_m = AutoModelForDepthEstimation.from_pretrained(
        DEPTH_MODEL
    ).to(DEVICE).eval()

    return oneformer_p, oneformer_m, mask_p, mask_m, depth_p, depth_m

# ================= SEMANTIC =================
def semantic_segmentation(image, processor, model):
    inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    seg_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0].cpu().numpy()

    return seg_map, model.config.id2label

# ================= INSTANCE =================
def instance_segmentation(image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    return processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

# ================= MASK REFINEMENT =================
def subtract_instances(mask, instances, id2label):
    inst_map = instances["segmentation"].cpu().numpy()

    for seg in instances["segments_info"]:
        label = id2label.get(seg["label_id"], "").lower()
        if any(x in label for x in EXCLUDE_ATTACHED_OBJECTS):
            mask[inst_map == seg["id"]] = 0

    return mask

def depth_refine(mask, image, processor, model):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth

    depth = depth.squeeze().cpu().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]))

    edges = cv2.Laplacian(depth, cv2.CV_8U)
    mask[edges > 20] = 0
    return mask

def clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 800:
            cleaned[labels == i] = 255

    return cleaned

# ================= TEXTURE =================
def apply_texture(image, mask, texture, alpha=0.9):
    h, w = image.shape[:2]
    texture = cv2.resize(texture, (w, h))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) / 255.0
    shading = np.dstack([gray, gray, gray])

    texture = (texture / 255.0) * shading
    image = image / 255.0

    m = mask[:, :, None] > 0
    image[m] = (1 - alpha) * image[m] + alpha * texture[m]

    return (image * 255).astype(np.uint8)

# ================= MAIN BACKEND FUNCTION =================
def process_image(
    image_bytes: bytes,
    target_object: str,
    texture_bytes: bytes | None = None
) -> np.ndarray:
    """
    Backend inference entry point
    """
    image = Image.open(
        np.frombuffer(image_bytes, np.uint8)
    ).convert("RGB")

    (
        one_p, one_m,
        mask_p, mask_m,
        depth_p, depth_m
    ) = load_models()

    seg_map, id2label = semantic_segmentation(image, one_p, one_m)
    instances = instance_segmentation(image, mask_p, mask_m)

    mask = np.zeros_like(seg_map, dtype=np.uint8)
    for cid, name in id2label.items():
        if name == target_object:
            mask |= (seg_map == cid).astype(np.uint8)

    mask *= 255

    mask = subtract_instances(mask, instances, mask_m.config.id2label)
    mask = depth_refine(mask, image, depth_p, depth_m)
    mask = clean_mask(mask)

    image_np = np.array(image)

    if texture_bytes:
        texture = cv2.imdecode(
            np.frombuffer(texture_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )
        texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
        return apply_texture(image_np, mask, texture)

    overlay = image_np.copy()
    overlay[mask > 0] = (1 - ALPHA) * overlay[mask > 0] + ALPHA * np.array([0, 255, 0])
    return overlay.astype(np.uint8)
