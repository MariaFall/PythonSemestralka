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
    - 'none': No negative effect
    """
    img_array = np.array(image)

    if negative_type == "bw":
        grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        negative_array = 255 - grayscale
        return Image.fromarray(negative_array, mode="L")  # "L" mode = grayscale
    elif negative_type == "color":
        negative_array = 255 - img_array
        return Image.fromarray(negative_array.astype(np.uint8))

    return image  # Return original if no negative effect is applied


def resize_image(image: Image.Image, resize_percent: int) -> Image.Image:
    """
    Resize the image by a given percentage (0-100% of the original size).
    """
    if resize_percent < 1 or resize_percent > 100:
        return image  # Ignore invalid resize values

    width, height = image.size
    new_size = (int(width * resize_percent / 100), int(height * resize_percent / 100))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def apply_edge_detection(image: Image.Image, method: str) -> Image.Image:
    """
    Edge detection using NumPy-based convolution.
    - 'sobel': Sobel filter
    - 'laplacian': Laplacian filter
    """
    img_array = np.array(image.convert("L"), dtype=np.float32)  # Convert to grayscale

    # Define Sobel and Laplacian kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    def convolve(image, kernel):
        """Efficient 2D convolution using NumPy."""
        kernel_size = kernel.shape[0]
        pad_size = kernel_size // 2
        padded = np.pad(image, pad_size, mode='reflect')
        shape = (image.shape[0], image.shape[1], kernel_size, kernel_size)
        strides = (padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
        patches = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        return np.clip(np.einsum('ijkl,kl->ij', patches, kernel, optimize=True), 0, 255).astype(np.uint8)

    if method == "sobel":
        edges_x = convolve(img_array, sobel_x)
        edges_y = convolve(img_array, sobel_y)
        edges = np.hypot(edges_x, edges_y)
    elif method == "laplacian":
        edges = convolve(img_array, laplacian_kernel)
    else:
        return image  # Return original if no valid method is chosen

    edges = np.clip(edges / edges.max() * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(edges, mode="L")  # Return grayscale image with detected edges


def process_image(image_file, brightness, solarization, negative_type, resize_percent, edge_detection):
    """
    Process the image by applying brightness, solarization, negative effects, resizing, and edge detection.
    """
    image = Image.open(image_file).convert('RGB')

    if resize_percent != 100:
        image = resize_image(image, resize_percent)

    if brightness != 0:
        image = adjust_brightness(image, brightness)

    if solarization:
        image = apply_solarization(image)

    if negative_type != "none":
        image = apply_negative(image, negative_type)

    if edge_detection and edge_detection != "none":
        image = apply_edge_detection(image, edge_detection)

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
    negative_type = request.form.get('negative-type', 'none')  # Default to no negative
    resize_percent = int(request.form.get('resize', 100))  # Default to 100% (no resizing)
    edge_detection = request.form.get('edge-detection', 'none')

    if image_file.filename == '':
        return "No selected file", 400

    processed_image = process_image(image_file, brightness, solarization, negative_type, resize_percent, edge_detection)

    img_io = io.BytesIO()
    processed_image.save(img_io, format="PNG")
    img_io.seek(0)

    return send_file(img_io, mimetype="image/png")


if __name__ == '__main__':
    app.run(debug=True)