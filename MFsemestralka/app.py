from flask import Flask, request, render_template, send_file
import numpy as np
from PIL import Image
import io

app = Flask(__name__)


def adjust_brightness(image: Image.Image, brightness: int) -> Image.Image:
    """
    Adjust the brightness of an image.
    Brightness ranges from -100 (darken) to +100 (lighten).
    """
    img_array = np.array(image, dtype=np.int16)
    img_array = np.clip(img_array + brightness * 2.55, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def apply_solarization(image: Image.Image) -> Image.Image:
    """
    Apply solarization effect to the image.
    Pixels above 128 are inverted.
    """
    img_array = np.array(image)
    solarized_array = np.where(img_array > 128, 255 - img_array, img_array)
    return Image.fromarray(solarized_array.astype(np.uint8))


def apply_negative(image: Image.Image, negative_type: str) -> Image.Image:
    """
    Apply negative effect to an image.
    - 'color': Full color inversion
    - 'bw': Convert to grayscale before inversion
    """
    img_array = np.array(image)

    if negative_type == "bw":
        grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        negative_array = 255 - grayscale
        return Image.fromarray(negative_array, mode="L")  # "L" mode = grayscale

    negative_array = 255 - img_array
    return Image.fromarray(negative_array.astype(np.uint8))


def resize_image(image: Image.Image, resize_percent: int) -> Image.Image:
    """
    Resize the image by a given percentage (1-100% of the original size).
    """
    if resize_percent < 1 or resize_percent > 100:
        return image  # Ignore invalid resize values

    width, height = image.size
    new_size = (int(width * resize_percent / 100), int(height * resize_percent / 100))
    return image.resize(new_size, Image.Resampling.LANCZOS)  # Fixed: Using Image.Resampling.LANCZOS


def process_image(image_file, brightness, solarization, negative, negative_type, resize_percent):
    """
    Process the image by applying brightness, solarization, negative effects, and resizing.
    """
    image = Image.open(image_file).convert('RGB')

    if resize_percent != 100:  # Only resize if it's not 100%
        image = resize_image(image, resize_percent)

    if brightness != 0:
        image = adjust_brightness(image, brightness)

    if solarization:
        image = apply_solarization(image)

    if negative:
        image = apply_negative(image, negative_type)

    return image


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image provided", 400

    image_file = request.files['image']
    brightness = int(request.form.get('brightness', 0))
    solarization = 'solarization' in request.form
    negative = 'negative' in request.form
    negative_type = request.form.get('negative-type', 'color')  # Default to color negative
    resize_percent = int(request.form.get('resize', 100))  # Default to 100% (no resizing)

    if image_file.filename == '':
        return "No selected file", 400

    processed_image = process_image(image_file, brightness, solarization, negative, negative_type, resize_percent)

    img_io = io.BytesIO()
    processed_image.save(img_io, format="PNG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/png")


if __name__ == '__main__':
    app.run(debug=True)