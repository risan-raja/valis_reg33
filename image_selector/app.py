import glob
import os
from flask import Flask, Response, render_template, request, redirect, url_for, session, send_file
from io import BytesIO
from collections import defaultdict
import json
import cv2
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Define the path where the slides (folders) are stored
SLIDES_DIR = "/storage/valis_reg/stain_sep_results"
slides = os.listdir(SLIDES_DIR)
slides = sorted(slides)
slide_images = {}

if os.path.exists('selected_slide_images.json'):
    with open('selected_slide_images.json', 'r') as f:
        selected_slide_images = json.load(f)

# Get the images for each slide
for slide in slides:
    slide_path = os.path.join(SLIDES_DIR, slide)
    if os.path.isdir(slide_path):
        slide_images[slide] = [img for img in os.listdir(slide_path) if img.endswith(('png', 'jpg', 'jpeg'))]

selected_slide_images = defaultdict(list)
for slide in slides:
    selected_slide_images[slide] = []
    for slide_img in slide_images[slide]:
        if 'proc_mask' in slide_img:
            if len(selected_slide_images[slide]) == 0:
                selected_slide_images[slide].append(slide_img)

with open('selected_slide_images.json', 'w') as f:
    json.dump(dict(selected_slide_images), f)


# Route to display the slides and images
@app.route('/')
def index():
    global selected_slide_images, slide_images, slides
    selected_slide_images
    # session['slides'] = slides
    # session['slide_images'] = slide_images
    if 'processed_slides' not in session:
        session['processed_slides'] = [False] * len(slides)  # False means unprocessed
        session['current_slide'] = 0
        for slide_idx, slide in enumerate(slides):
            if slide in selected_slide_images:
                if len(selected_slide_images[slide]) > 1:
                    session['processed_slides'][slide_idx] = True
                else:
                    session['processed_slides'][slide_idx] = False

    return render_template('grid.html', slides=slides, processed=session['processed_slides'])



# Route to display images from the selected slide folder
@app.route('/slide/<int:slide_index>')
def show_slide(slide_index):
    global slides, selected_slide_images, slide_images
    # slides = session.get('slides', [])
    current_slide = slides[slide_index]
    curr_slide_images = slide_images[current_slide]
    main_images = [img for img in curr_slide_images if any(x in img for x in ['proc_mask', 'rf_mask', 'raw'])]
    stain_images = [img for img in curr_slide_images if img not in main_images]
    curr_selected_imgs = selected_slide_images[current_slide]
    # Update session to mark this slide as the current one
    session['current_slide'] = slide_index

    return render_template('slide.html', slide=current_slide, main_images=main_images, stain_images=stain_images, selected_files=curr_selected_imgs)

@app.route('/image/<slide>/stain/<image_name>/<int:resolution>')
def get_slide_stain_image(slide, image_name, resolution) -> Response:
    image_path = os.path.join(SLIDES_DIR, slide, image_name)

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Resize the image (e.g., width = 200, height = proportional)
    target_width = resolution
    aspect_ratio = target_width / img.shape[1]
    new_dimensions = (target_width, int(img.shape[0] * aspect_ratio))
    resized_img = cv2.resize(img, new_dimensions)

    # Convert the image to RGB (because OpenCV loads as BGR)
    rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL format for encoding
    pil_image = Image.fromarray(rgb_image)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return send_file(buffer, mimetype='image/jpeg')

@app.route('/image/<slide>/main/<image_name>')
def get_slide_main_image(slide, image_name)->Response:
    image_path = os.path.join(SLIDES_DIR, slide, image_name)

    # Load the image using OpenCV
    img = cv2.imread(image_path)

    # Resize the image (e.g., width = 200, height = proportional)
    target_width = 500
    aspect_ratio = target_width / img.shape[1]
    new_dimensions = (target_width, int(img.shape[0] * aspect_ratio))
    resized_img = cv2.resize(img, new_dimensions)

    # Convert the image to RGB (because OpenCV loads as BGR)
    rgb_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Convert the image to PIL format for encoding
    pil_image = Image.fromarray(rgb_image)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return send_file(buffer, mimetype='image/jpeg')


# Route to handle image selection
@app.route('/select_images', methods=['POST'])
def select_images():
    global selected_slide_images, slides
    selected_images = request.form.getlist('selected_images')
    current_slide_index = session['current_slide']

    current_slide = slides[session['current_slide']]
    selected_slide_images[current_slide] = selected_images
    with open('selected_slide_images.json', 'w') as f:
        json.dump(dict(selected_slide_images), f)
    print("Selected Images:", selected_images)
    # You can process the selected images here (store them in a database, move to a different folder, etc.)
    session['processed_slides'][current_slide_index] = True
    # Move to the next slide if available
    if current_slide_index + 1 < len(slides):
        session['current_slide'] = current_slide_index + 1
        return redirect(url_for('show_slide', slide_index=session['current_slide']))
    else:
        # If no more slides, return to home
        return redirect(url_for('start'))

if __name__ == '__main__':
    app.run(debug=True)