import flask, AI_Helper
from flask import request, jsonify, render_template, redirect, url_for
import os, tag_folder
from PIL import Image, ImageDraw, ImageFont
import hashlib
import colorsys

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def overlay_img(image, detections):

    # Load image if path
    if isinstance(image, str):
        image = Image.open(image).convert("RGBA")

    draw = ImageDraw.Draw(image, "RGBA")

    img_w, img_h = image.size

    # =========================
    # Auto font scaling
    # =========================
    base_size = int(min(img_w, img_h) * 0.035)  # scales with image
    base_size = max(18, base_size)  # minimum size safeguard

    font = ImageFont.truetype(
        "/usr/share/fonts/Adwaita/AdwaitaMono-Bold.ttf",
        base_size
    )

    # =========================
    # Dark-mode friendly color generator
    # =========================
    def label_to_color(label):
        h = int(hashlib.md5(label.encode()).hexdigest(), 16)

        hue = (h % 360) / 360.0

        # Tuned for dark UI (not too bright, not too dull)
        saturation = 0.65
        value = 0.85

        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

        return (int(r * 255), int(g * 255), int(b * 255), 255)

    # =========================
    # Count categories
    # =========================
    category_counts = {}
    for det in detections:
        label = det["class_name"]
        category_counts[label] = category_counts.get(label, 0) + 1

    # =========================
    # Panel layout (scaled)
    # =========================
    panel_padding = int(base_size * 0.6)
    line_height = int(base_size * 1.4)

    panel_items = sorted(
        category_counts.items(),
        key=lambda x: -x[1]
    )

    panel_items = [f"{k}: {v}" for k, v in panel_items]

    # Measure width
    max_width = 0
    for item in panel_items:
        bbox = draw.textbbox((0, 0), item, font=font)
        w = bbox[2] - bbox[0]
        max_width = max(max_width, w)

    panel_width = max_width + panel_padding * 2
    panel_height = len(panel_items) * line_height + panel_padding * 2

    margin = int(base_size * 0.8)

    panel_x1 = img_w - panel_width - margin
    panel_y1 = img_h - panel_height - margin
    panel_x2 = img_w - margin
    panel_y2 = img_h - margin

    # =========================
    # Panel background
    # =========================
    draw.rounded_rectangle(
        [panel_x1, panel_y1, panel_x2, panel_y2],
        radius=int(base_size * 0.5),
        fill=(15, 15, 20, 180)  # dark UI background
    )

    # =========================
    # Draw text (colored)
    # =========================
    y_offset = panel_y1 + panel_padding

    for item in panel_items:
        label_name = item.split(":")[0]
        color = label_to_color(label_name)

        text_x = panel_x1 + panel_padding
        text_y = y_offset

        # Shadow
        draw.text(
            (text_x + 2, text_y + 2),
            item,
            fill=(0, 0, 0, 255),
            font=font
        )

        # Colored text
        draw.text(
            (text_x, text_y),
            item,
            fill=color,
            font=font
        )

        y_offset += line_height

    return image

overlay_img("IMGs/test_cat.jpg",
            [{'xyxy': [345.90948486328125, 24.44744873046875, 639.9216918945312, 373.1561279296875], 'confidence': 0.9418997168540955, 'class_id': 15, 'class_name': 'cat'}, 
             {'xyxy': [40.595611572265625, 73.32191467285156, 175.53985595703125, 118.13456726074219], 'confidence': 0.9358233213424683, 'class_id': 65, 'class_name': 'remote'}]
)

@app.route('/', methods=['GET'])
def home():
    image_urls = []
    for filename in os.listdir('static'):
        if filename.endswith('.png'):
            image_urls.append(url_for('static', filename=filename))
    print(f"Serving {len(image_urls)} images: {image_urls}")
    return render_template('index.html', image_urls=image_urls)


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image_path = os.path.join("uploads", image_file.filename)
    image_file.save(image_path)

    detections = AI_Helper.Yolov.run_yolo(image=image_path)
    results = tag_folder.threshold_checker(detections)
    result = overlay_img(image_path, results['good'])
    result.save(f"static/{image_file.filename}.png")

    return redirect(url_for('static', filename=f"{image_file.filename}.png"))


if __name__ == "__main__":
    app.run(debug=True)