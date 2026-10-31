
import re
import datetime as dt
from statistics import mean

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image, ImageOps


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="eKYC Document Validation",
    page_icon="🔐",
    layout="wide",
)

st.title("🔐 eKYC Document Preprocessing & Validation")
st.caption(
    "Image preprocessing • OCR • Field validation • "
    "Face detection • Optional face verification"
)

st.info(
    "Prototype only: passing these checks does not establish that "
    "an identity document is genuine. Review results manually and "
    "use authorized verification services where available."
)


# =========================================================
# SESSION STATE AND OCR MODEL
# =========================================================

@st.cache_resource
def load_ocr_reader():
    # Downloads the EasyOCR model on the first run if necessary.
    return easyocr.Reader(["en"], gpu=False)


# =========================================================
# IMAGE UTILITIES
# =========================================================

def uploaded_to_bgr(uploaded_file):
    """Convert an uploaded image to an OpenCV BGR array."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mean_kernel(size):
    """Create the mean filter used in the notebook."""
    return np.ones((size, size), dtype=np.float32) / (size**2)


def apply_preprocessing(image, blur_size=5, threshold=120):
    """Run the notebook's principal image-processing operations."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_blur = cv2.filter2D(
        image, -1, mean_kernel(blur_size)
    )

    gaussian_blur = cv2.GaussianBlur(
        gray, (blur_size, blur_size), 0
    )

    _, binary = cv2.threshold(
        gaussian_blur,
        threshold,
        255,
        cv2.THRESH_BINARY,
    )

    _, binary_inv = cv2.threshold(
        gaussian_blur,
        threshold,
        255,
        cv2.THRESH_BINARY_INV,
    )

    otsu_value, otsu = cv2.threshold(
        gaussian_blur,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )

    adaptive = cv2.adaptiveThreshold(
        gaussian_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        21,
        2,
    )

    return {
        "gray": gray,
        "mean_blur": mean_blur,
        "gaussian_blur": gaussian_blur,
        "binary": binary,
        "binary_inv": binary_inv,
        "otsu": otsu,
        "otsu_value": otsu_value,
        "adaptive": adaptive,
    }


def crop_largest_contour(image, threshold_image):
    """
    Crop the largest external contour, following the notebook.
    Fall back to the original image if no suitable contour exists.
    """
    contours, _ = cv2.findContours(
        threshold_image.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return image.copy(), None, "No contours found; using original image."

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)

    image_h, image_w = image.shape[:2]

    # Avoid accidentally cropping a tiny noise region.
    if w < image_w * 0.15 or h < image_h * 0.15 or area < 100:
        return image.copy(), None, (
            "Largest contour was too small; using original image."
        )

    crop = image[y:y + h, x:x + w].copy()

    return crop, {
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "area": float(area),
    }, "Largest external contour selected."


# =========================================================
# OCR
# =========================================================

def run_ocr(image, reader):
    """Run EasyOCR and normalize its output."""
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
    )

    rows = []

    for line_no, item in enumerate(results, start=1):
        bbox, text, confidence = item

        text = re.sub(r"\s+", " ", str(text)).strip()
        text = re.sub(r"^[|:;,\-_.\s]+|[|:;,\-_.\s]+$", "", text)

        if not text:
            continue

        rows.append({
            "line_no": line_no,
            "text": text,
            "confidence": float(confidence),
            "bbox": [
                (float(point[0]), float(point[1]))
                for point in bbox
            ],
        })

    return rows


def bbox_rect(bbox):
    if not bbox:
        return None

    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]

    return min(xs), min(ys), max(xs), max(ys)


def mask_value(value, keep_start=2, keep_end=2):
    """Mask sensitive values for display."""
    if value is None or value == "":
        return ""

    value = str(value)

    if len(value) <= keep_start + keep_end + 2:
        return "*" * len(value)

    middle = "*" * (len(value) - keep_start - keep_end)

    return value[:keep_start] + middle + value[-keep_end:]


def draw_ocr_overlay(image, rows, threshold=0.55):
    """Green boxes mean acceptable OCR confidence; red means low."""
    overlay = image.copy()

    for row in rows:
        rect = bbox_rect(row["bbox"])

        if rect is None:
            continue

        x1, y1, x2, y2 = [int(v) for v in rect]

        color = (
            (0, 160, 0)
            if row["confidence"] >= threshold
            else (0, 0, 255)
        )

        cv2.rectangle(
            overlay, (x1, y1), (x2, y2), color, 2
        )

    return overlay


# =========================================================
# DOCUMENT FIELD EXTRACTION

# =========================================================

DOCUMENT_CONFIGS = {
    "Student ID": {
        "required": ["name", "id_number"],
        "fields": {
            "name": {
                "labels": ["student name", "name"],
                "validator": "person_name",
            },
            "id_number": {
                "labels": [
                    "student id",
                    "id number",
                    "id no",
                    "roll no",
                    "registration no",
                    "reg no",
                ],
                "patterns": [r"\b[A-Z0-9][A-Z0-9\-/]{3,}\b"],
                "validator": "id_number",
            },
            "date_of_birth": {
                "labels": ["date of birth", "birth date", "dob", "d.o.b"],
                "patterns": [
                    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
                ],
                "validator": "past_date",
            },
            "expiry_date": {
                "labels": [
                    "valid until",
                    "valid upto",
                    "expiry date",
                    "expires",
                    "exp date",
                ],
                "validator": "not_expired",
            },
        },
    },
    "Generic ID": {
        "required": ["document_number"],
        "fields": {
            "name": {
                "labels": ["full name", "name"],
                "validator": "person_name",
            },
            "document_number": {
                "labels": [
                    "document number",
                    "document no",
                    "id number",
                    "id no",
                    "number",
                ],
                "patterns": [r"\b[A-Z0-9][A-Z0-9\-/]{4,}\b"],
                "validator": "id_number",
            },
            "date_of_birth": {
                "labels": ["date of birth", "birth date", "dob"],
                "patterns": [
                    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
                ],
                "validator": "past_date",
            },
            "expiry_date": {
                "labels": ["valid until", "valid upto", "expiry", "expires"],
                "validator": "not_expired",
            },
        },
    },
}


def clean_text(text):
    text = str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" |:;,-_.")


def find_label_value(rows, labels):
    """
    Look for a field label and its value on the same line
    or on the following line.
    """
    for i, row in enumerate(rows):
        text = row["text"]

        for label in sorted(labels, key=len, reverse=True):
            pattern = (
                r"(?<!\w)" + re.escape(label)
                + r"(?!\w)\s*[:#\-]?\s*(.*)$"
            )

            match = re.search(pattern, text, flags=re.IGNORECASE)

            if match:
                value = clean_text(match.group(1))

                if value:
                    return value, [row]

                if i + 1 < len(rows):
                    return rows[i + 1]["text"], [row, rows[i + 1]]

    return None, []


def find_pattern_value(rows, patterns):
    full_text = "\n".join(row["text"] for row in rows)

    for pattern in patterns:
        match = re.search(pattern, full_text, flags=re.IGNORECASE)

        if match:
            value = match.group(1) if match.groups() else match.group(0)

            supporting_rows = [
                row for row in rows
                if value.lower() in row["text"].lower()
            ]

            return clean_text(value), supporting_rows

    return None, []


def parse_date(value):
    if not value:
        return None

    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d",
        "%d/%m/%y", "%m/%d/%y",
    ]

    for fmt in formats:
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    return None


def validate_value(value, validator):
    if not value:
        return False, "Missing field"

    if validator == "person_name":
        letters = re.sub(r"[^A-Za-z]", "", value)

        if len(letters) < 3:
            return False, "Name is too short or unclear"

        if re.search(r"\d", value):
            return False, "Name contains digits"

        return True, "Readable name format"

    if validator == "id_number":
        compact = re.sub(r"\s+", "", value)

        if len(compact) < 4:
            return False, "ID number is too short"

        if not re.search(r"[A-Za-z0-9]", compact):
            return False, "No letters or digits found"

        return True, "Plausible ID number format"

    if validator in ("past_date", "not_expired"):
        parsed = parse_date(value)

        if parsed is None:
            return False, "Date could not be parsed"

        today = dt.date.today()

        if validator == "past_date" and parsed >= today:
            return False, "Date of birth must be in the past"

        if validator == "not_expired" and parsed < today:
            return False, "Document may be expired"

        return True, f"Date parsed as {parsed.isoformat()}"

    return True, "No validator configured"


def validate_document(rows, document_type, threshold=0.55):
    config = DOCUMENT_CONFIGS[document_type]
    extracted = {}
    validation = []

    for field, field_config in config["fields"].items():
        value, support = find_label_value(
            rows, field_config.get("labels", [])
        )

        if not value:
            value, support = find_pattern_value(
                rows, field_config.get("patterns", [])
            )

        scores = [
            row["confidence"]
            for row in support
            if row.get("confidence") is not None
        ]

        confidence = mean(scores) if scores else None

        passed, message = validate_value(
            value, field_config.get("validator")
        )

        if confidence is not None and confidence < threshold:
            passed = False
            message += f"; low OCR confidence ({confidence:.2f})"

        required = field in config["required"]

        extracted[field] = {
            "value": value,
            "confidence": confidence,
            "required": required,
        }

        validation.append({
            "Field": field.replace("_", " ").title(),
            "Required": required,
            "Present": bool(value),
            "Masked value": mask_value(value),
            "Confidence": (
                round(confidence, 3)
                if confidence is not None else None
            ),
            "Passed": passed,
            "Message": message,
        })

    return extracted, validation


# =========================================================
# LAYOUT VALIDATION
# =========================================================

def layout_checks(rows, image):
    height, width = image.shape[:2]
    checks = []

    boxed_rows = [row for row in rows if row.get("bbox")]

    checks.append({
        "Check": "Bounding boxes present",
        "Passed": len(boxed_rows) > 0,
        "Details": f"{len(boxed_rows)} OCR text regions found",
    })

    invalid = []
    outside = []
    unusual_size = []

    for row in boxed_rows:
        rect = bbox_rect(row["bbox"])

        if rect is None:
            invalid.append(row["line_no"])
            continue

        x1, y1, x2, y2 = rect
        box_w, box_h = x2 - x1, y2 - y1

        if box_w <= 0 or box_h <= 0:
            invalid.append(row["line_no"])

        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            outside.append(row["line_no"])

        area_ratio = (box_w * box_h) / float(width * height)

        if area_ratio < 0.00002 or area_ratio > 0.35:
            unusual_size.append(row["line_no"])

    checks.extend([
        {
            "Check": "Valid box geometry",
            "Passed": len(invalid) == 0,
            "Details": (
                "Boxes have positive dimensions"
                if not invalid else f"Review lines {invalid}"
            ),
        },
        {
            "Check": "Boxes inside image",
            "Passed": len(outside) == 0,
            "Details": (
                "All boxes are inside the image"
                if not outside else f"Review lines {outside}"
            ),
        },
        {
            "Check": "Reasonable box size",
            "Passed": len(unusual_size) == 0,
            "Details": (
                "No unusually small or large boxes detected"
                if not unusual_size else f"Review lines {unusual_size}"
            ),
        },
        {
            "Check": "Enough text regions",
            "Passed": len(boxed_rows) >= 3,
            "Details": f"{len(boxed_rows)} text regions detected",
        },
    ])

    return checks


# =========================================================
# FACE DETECTION
# =========================================================

def detect_largest_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_file = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    cascade = cv2.CascadeClassifier(cascade_file)

    if cascade.empty():
        raise RuntimeError(
            f"Could not load Haar Cascade classifier: {cascade_file}"
        )

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    x1 = max(0, x - w // 4)
    y1 = max(0, y - h // 4)
    x2 = min(image.shape[1], x + w + w // 4)
    y2 = min(image.shape[0], y + h + h // 4)

    return image[y1:y2, x1:x2].copy(), (x, y, w, h)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Processing settings")

document_type = st.sidebar.selectbox(
    "Document profile",
    list(DOCUMENT_CONFIGS.keys()),
)

blur_size = st.sidebar.select_slider(
    "Blur kernel size",
    options=[3, 5, 7, 9, 11, 13],
    value=5,
)

threshold_value = st.sidebar.slider(
    "Binary threshold",
    min_value=0,
    max_value=255,
    value=120,
)

confidence_threshold = st.sidebar.slider(
    "OCR confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.55,
    step=0.05,
)

use_contour_crop = st.sidebar.checkbox(
    "Crop largest contour before OCR",
    value=True,
)

st.sidebar.caption(
    "Thresholding and contour cropping are image-processing "
    "heuristics. Review the preview before relying on OCR results."
)


# =========================================================
# DOCUMENT UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an ID-card image",
    type=["png", "jpg", "jpeg"],
    help="Use a clear, well-lit image with readable text.",
)

if uploaded_file is None:
    st.markdown(
        "### Getting started\n"
        "Upload an image to view preprocessing, OCR, "
        "field validation and layout checks."
    )
    st.stop()

try:
    original_bgr = uploaded_to_bgr(uploaded_file)
except Exception as exc:
    st.error(f"Could not open this image: {exc}")
    st.stop()

height, width = original_bgr.shape[:2]

m1, m2, m3 = st.columns(3)
m1.metric("Image width", f"{width}px")
m2.metric("Image height", f"{height}px")
m3.metric("File size", f"{uploaded_file.size / 1024:.1f} KB")

st.subheader("1. Original document")
st.image(bgr_to_rgb(original_bgr), use_container_width=True)


# =========================================================
# PREPROCESSING RESULTS
# =========================================================

processed = apply_preprocessing(
    original_bgr,
    blur_size=blur_size,
    threshold=threshold_value,
)

st.subheader("2. Image preprocessing")

previews = [
    ("Grayscale", processed["gray"], "gray"),
    ("Mean blur", processed["mean_blur"], None),
    ("Gaussian blur", processed["gaussian_blur"], "gray"),
    ("Binary threshold", processed["binary"], "gray"),
    ("Inverse threshold", processed["binary_inv"], "gray"),
    ("Otsu threshold", processed["otsu"], "gray"),
    ("Adaptive threshold", processed["adaptive"], "gray"),
]

cols = st.columns(3)

for i, (title, image, cmap) in enumerate(previews):
    with cols[i % 3]:
        st.markdown(f"**{title}**")

        if cmap == "gray":
            st.image(image, use_container_width=True)
        else:
            st.image(bgr_to_rgb(image), use_container_width=True)

st.caption(
    f"Otsu selected threshold: {processed['otsu_value']:.0f}"
)


# =========================================================
# HISTOGRAM
# =========================================================

with st.expander("View pixel intensity histogram"):
    gray = processed["gray"]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(gray.ravel(), bins=256, range=(0, 256))
    ax.set_title("Grayscale pixel intensity distribution")
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Pixel count")
    st.pyplot(fig)
    plt.close(fig)


# =========================================================
# CONTOUR CROPPING
# =========================================================

st.subheader("3. Contour detection and cropping")

if use_contour_crop:
    ocr_bgr, contour_info, crop_message = crop_largest_contour(
        original_bgr,
        processed["adaptive"],
    )
else:
    ocr_bgr = original_bgr.copy()
    contour_info = None
    crop_message = "Contour cropping disabled; using original image."

st.write(crop_message)

if contour_info:
    st.json(contour_info)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Image sent to OCR**")
    st.image(bgr_to_rgb(ocr_bgr), use_container_width=True)

with col2:
    st.markdown("**Adaptive threshold used for contours**")
    st.image(processed["adaptive"], use_container_width=True)


# =========================================================
# OCR AND DOCUMENT VALIDATION
# =========================================================

st.subheader("4. Text extraction and document validation")

if st.button("Run OCR and validation", type="primary"):
    try:
        with st.spinner(
            "Loading EasyOCR and processing the document..."
        ):
            reader = load_ocr_reader()
            rows = run_ocr(bgr_to_rgb(ocr_bgr), reader)

        if not rows:
            st.warning(
                "No text detected. Try a clearer image or disable "
                "contour cropping."
            )
            st.stop()

        extracted, validation = validate_document(
            rows,
            document_type,
            confidence_threshold,
        )

        layouts = layout_checks(rows, ocr_bgr)

        low_confidence = [
            row for row in rows
            if row["confidence"] < confidence_threshold
        ]

        required_issues = [
            item for item in validation
            if item["Required"]
            and (not item["Present"] or not item["Passed"])
        ]

        failed_layout = [
            item for item in layouts if not item["Passed"]
        ]

        if required_issues or failed_layout:
            status = "Needs manual review"
        elif low_confidence:
            status = "Basic checks completed with warnings"
        else:
            status = "No obvious inconsistencies found in basic checks"

        # Do not persist sensitive document values between sessions.
        st.session_state["ekyc_results"] = {
            "rows": rows,
            "validation": validation,
            "layouts": layouts,
            "extracted": extracted,
            "status": status,
            "low_confidence": low_confidence,
            "overlay": draw_ocr_overlay(
                bgr_to_rgb(ocr_bgr),
                rows,
                confidence_threshold,
            ),
        }

    except Exception as exc:
        st.error(f"OCR or validation failed: {exc}")
        st.exception(exc)


# =========================================================
# DISPLAY RESULTS
# =========================================================

result = st.session_state.get("ekyc_results")

if result:
    st.divider()
    st.subheader("5. Validation summary")

    if result["status"] == "Needs manual review":
        st.warning(result["status"])
    elif result["low_confidence"]:
        st.info(result["status"])
    else:
        st.success(result["status"])

    st.caption(
        "This status summarizes basic consistency checks, not document "
        "authenticity or identity verification."
    )

    st.markdown("**OCR text with confidence scores**")

    ocr_table = pd.DataFrame([
        {
            "Line": row["line_no"],
            "Text": mask_value(row["text"]),
            "Confidence": round(row["confidence"], 3),
            "Review": (
                "Review"
                if row["confidence"] < confidence_threshold
                else "OK"
            ),
        }
        for row in result["rows"]
    ])

    st.dataframe(ocr_table, use_container_width=True)

    st.markdown("**Extracted fields**")

    field_table = pd.DataFrame([
        {
            "Field": field.replace("_", " ").title(),
            "Masked value": mask_value(info["value"]),
            "Required": info["required"],
            "Confidence": (
                round(info["confidence"], 3)
                if info["confidence"] is not None else None
            ),
        }
        for field, info in result["extracted"].items()
    ])

    st.dataframe(field_table, use_container_width=True)

    st.markdown("**Field validation**")
    st.dataframe(
        pd.DataFrame(result["validation"]),
        use_container_width=True,
    )

    st.markdown("**OCR bounding-box overlay**")
    st.caption("Green = acceptable OCR confidence; red = low confidence.")
    st.image(result["overlay"], use_container_width=True)

    st.markdown("**Layout checks**")
    st.dataframe(
        pd.DataFrame(result["layouts"]),
        use_container_width=True,
    )

    if result["low_confidence"]:
        st.warning(
            f"{len(result['low_confidence'])} OCR line(s) need manual review."
        )

    # Export only masked results, not full document values.
    export_data = pd.DataFrame(result["validation"]).to_csv(index=False)

    st.download_button(
        "Download masked validation report",
        data=export_data,
        file_name="ekyc_validation_report.csv",
        mime="text/csv",
    )


# =========================================================
# FACE DETECTION
# =========================================================

st.divider()
st.subheader("6. Face detection")

if st.button("Detect face in document image"):
    try:
        with st.spinner("Detecting the largest face..."):
            face_crop, face_box = detect_largest_face(ocr_bgr)

        if face_crop is None:
            st.warning(
                "No face detected. Try the original document image "
                "if cropping removed the photograph."
            )
            st.session_state.pop("ekyc_face", None)
        else:
            st.session_state["ekyc_face"] = face_crop

            x, y, w, h = face_box
            marked = ocr_bgr.copy()

            cv2.rectangle(
                marked, (x, y), (x + w, y + h), (0, 255, 0), 2
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Detected face location**")
                st.image(
                    bgr_to_rgb(marked),
                    use_container_width=True,
                )

            with col2:
                st.markdown("**Extracted face crop**")
                st.image(
                    bgr_to_rgb(face_crop),
                    use_container_width=True,
                )

            st.download_button(
                "Download extracted face",
                data=cv2.imencode(".jpg", face_crop)[1].tobytes(),
                file_name="extracted_face.jpg",
                mime="image/jpeg",
            )

    except Exception as exc:
        st.error(f"Face detection failed: {exc}")


# =========================================================
# OPTIONAL DEEPFACE VERIFICATION

# =========================================================

st.divider()
st.subheader("7. Optional face verification")

st.write(
    "Compare the face in the document with a separately uploaded "
    "photograph. This is a similarity check, not liveness detection."
)

face_upload = st.file_uploader(
    "Upload comparison face image",
    type=["png", "jpg", "jpeg"],
    key="comparison_face",
)

model_name = st.selectbox(
    "Face verification model",
    ["Facenet512", "Facenet"],
)

if face_upload is not None:
    st.image(
        ImageOps.exif_transpose(Image.open(face_upload)).convert("RGB"),
        caption="Uploaded comparison photograph",
        width=250,
    )

if st.button("Compare faces with DeepFace"):
    if face_upload is None:
        st.warning("Upload a comparison face image first.")

    elif "ekyc_face" not in st.session_state:
        st.warning("Detect a face in the document image first.")

    else:
        try:
            import tempfile
            import os
            from deepface import DeepFace

            with tempfile.TemporaryDirectory() as temp_dir:
                id_path = os.path.join(temp_dir, "id_face.jpg")
                comparison_path = os.path.join(
                    temp_dir, "comparison.jpg"
                )

                cv2.imwrite(
                    id_path,
                    st.session_state["ekyc_face"],
                )

                comparison_bgr = uploaded_to_bgr(face_upload)

                cv2.imwrite(
                    comparison_path,
                    comparison_bgr,
                )

                with st.spinner("Comparing face embeddings..."):
                    verification = DeepFace.verify(
                        img1_path=id_path,
                        img2_path=comparison_path,
                        model_name=model_name,
                        detector_backend="opencv",
                        enforce_detection=True,
                    )

            if verification["verified"]:
                st.success("DeepFace reports a face match.")
            else:
                st.warning("DeepFace did not find a face match.")

            c1, c2, c3 = st.columns(3)
            c1.metric("Match", str(verification["verified"]))
            c2.metric(
                "Distance",
                f"{verification['distance']:.4f}",
            )
            c3.metric(
                "Threshold",
                f"{verification['threshold']:.4f}",
            )

            st.caption(
                f"Model: {verification.get('model', model_name)}. "
                "A face match does not prove identity, document "
                "authenticity, or that either image is live."
            )

        except ImportError:
            st.error(
                "DeepFace is not installed. Install it with "
                "`python -m pip install deepface`."
            )

        except Exception as exc:
            st.error(f"Face comparison failed: {exc}")


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption(
    "eKYC prototype | OpenCV • EasyOCR • Streamlit • DeepFace"
)
st.caption(
    "For privacy, use test images when developing. This application "
    "does not save uploaded documents to a database, but uploaded "
    "images and extracted values remain in application memory while "
    "the session is active."
)
