from django.shortcuts import render
import os
import cv2
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =============================
# Load ML Models Once
# =============================
classifier_model = load_model(os.path.join(settings.BASE_DIR, "predictor/ml_model/classifier.h5"))
stone_seg_model = load_model(os.path.join(settings.BASE_DIR, "predictor/ml_model/stone_segmentation.h5"))
tumor_seg_model = load_model(os.path.join(settings.BASE_DIR, "predictor/ml_model/tumor_segmentation.h5"))


# =============================
# Home & About Views
# =============================
def home(request):
    return render(request, 'index.html')   # index.html must exist in templates/

def about(request):
    return render(request, 'about.html')   # about.html must exist in templates/


# =============================
# Helper Functions
# =============================
def analyze_stones(image_path, mask):
    img = cv2.imread(image_path)
    if mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sizes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5:  # Ignore tiny noise
            diameter_px = np.sqrt(4 * area / np.pi)
            diameter_mm = diameter_px * 0.3  # Example: pixel-to-mm scaling
            sizes.append(round(diameter_mm, 2))
            cv2.drawContours(img, [cnt], -1, (0, 0, 255), 2)

    # Overlay heatmap
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, mask_color, 0.4, 0)

    results_dir = os.path.join(settings.BASE_DIR, "static", "results")
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, "heatmap.jpg")
    cv2.imwrite(out_path, overlay)

    relative_path = os.path.relpath(out_path, settings.BASE_DIR)
    return sizes, relative_path


def map_effects(stone_sizes):
    effects = []
    for size in stone_sizes:
        if size < 5:
            effects.append(f"{size} mm → Usually harmless. May pass naturally.")
        elif size < 10:
            effects.append(f"{size} mm → Moderate pain, possible blockage.")
        elif size < 15:
            effects.append(f"{size} mm → Severe pain, UTIs, hydronephrosis.")
        else:
            effects.append(f"{size} mm → Critical risk, urgent surgery required.")
    return effects


# =============================
# Prediction View
# =============================
def predict_view(request):
    if request.method == "POST" and "file" in request.FILES:
        file = request.FILES["file"]

        # Save uploaded file
        upload_dir = os.path.join(settings.BASE_DIR, "static", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, file.name)

        with open(image_path, "wb+") as dest:
            for chunk in file.chunks():
                dest.write(chunk)

        # --- Step 1: Classification ---
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        preds = classifier_model.predict(x)
        class_id = np.argmax(preds)

        # --- Step 2: Branching ---
        if class_id == 2:  # Stone
            mask = stone_seg_model.predict(x)[0]
            mask = (mask > 0.5).astype("uint8") * 255
            stone_sizes, heatmap_path = analyze_stones(image_path, mask)
            effects = map_effects(stone_sizes)
            context = {
                "prediction": "Stone",
                "stone_count": len(stone_sizes),
                "stone_sizes": stone_sizes,
                "heatmap": "/" + heatmap_path.replace("\\", "/"),
                "effects": effects,
            }

        elif class_id == 3:  # Tumor
            mask = tumor_seg_model.predict(x)[0]
            mask = (mask > 0.5).astype("uint8") * 255
            tumor_sizes, heatmap_path = analyze_stones(image_path, mask)
            context = {
                "prediction": "Tumor",
                "tumor_size": tumor_sizes,
                "heatmap": "/" + heatmap_path.replace("\\", "/"),
                "effects": ["Tumor may affect surrounding tissues/organs"],
            }

        elif class_id == 0:  # Cyst
            context = {
                "prediction": "Cyst",
                "effects": ["Cyst may cause swelling, block urine flow, risk of infection."]
            }

        elif class_id == 1:  # Normal
            context = {
                "prediction": "Normal",
                "effects": ["Healthy kidney. No abnormalities detected."]
            }

        else:
            context = {"prediction": "Unknown", "effects": ["Model could not classify the image."]}

        return render(request, "predict.html", context)

    return render(request, "predict.html")
