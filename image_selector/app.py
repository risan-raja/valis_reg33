import asyncio
import glob
import os
from flask import (
    Flask,
    Response,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file,
)
from quart import Quart, render_template, request, session, jsonify, url_for, send_file, Response, redirect

from io import BytesIO
from collections import defaultdict
import json
import cv2
from PIL import Image

app = Quart(__name__)
app.secret_key = "3829749812374"  # Needed for session management

def initialize_processed_slides():
    if "processed_slides" not in session:
        session["processed_slides"] = [False] * len(app.slides)  # False means unprocessed
        session["current_slide"] = 0
        for slide_idx, slide in enumerate(app.slides):
            if slide in app.selected_slide_images:
                if len(app.selected_slide_images[slide]) > 1 or 'proc_mask' != app.selected_slide_images[slide][0]:
                    session["processed_slides"][slide_idx] = True
                else:
                    session["processed_slides"][slide_idx] = False

@app.before_serving
async def setup_slides():
    # Define the path where the slides (folders) are stored
    app.SLIDES_DIR =  os.environ.get("SLIDES_DIR", "/storage/valis_reg/stain_sep_results")
    slides = os.listdir(app.SLIDES_DIR)
    slides = sorted(slides)
    slide_images = {}


    # Get the images for each slide
    for slide in slides:
        slide_path = os.path.join(app.SLIDES_DIR, slide)
        if os.path.isdir(slide_path):
            slide_images[slide] = [
                img
                for img in os.listdir(slide_path)
                if img.endswith(("png", "jpg", "jpeg"))
            ]

    app.slides = slides
    app.slide_images = slide_images
    # Load the selected images from the JSON file
    # If the file does not exist, initialize with empty default dict
    if os.path.exists("selected_slide_images.json"):
        with open("selected_slide_images.json", "r") as f:
            selected_slide_images = json.load(f)

    else:
        selected_slide_images = defaultdict(list)
        for slide in slides:
            selected_slide_images[slide] = []
            for slide_img in slide_images[slide]:
                if "proc_mask" in slide_img:
                    if len(selected_slide_images[slide]) == 0:
                        selected_slide_images[slide].append(slide_img)
    
    app.selected_slide_images = selected_slide_images
    initialize_processed_slides()



def save_selected_images():
    with open("selected_slide_images.json", "w") as f:
        json.dump(dict(selected_slide_images), f)

@app.before_request
def make_session_permanent():
    global selected_slide_images
    session.permanent = True
    save_selected_images()


# Route to display the slides and images
@app.route("/")
async def index():
    initialize_processed_slides()

    return await render_template(
        "grid.html", slides=app.slides, processed=session["processed_slides"]
    )

async def load_and_resize_image(image_path, target_width):
    loop = asyncio.get_event_loop()
    # Run blocking image loading and resizing in the thread pool
    img = await loop.run_in_executor(None, cv2.imread, image_path)

    if img is None:
        return None

    aspect_ratio = target_width / img.shape[1]
    new_dimensions = (target_width, int(img.shape[0] * aspect_ratio))

    resized_img = await loop.run_in_executor(None, cv2.resize, img, new_dimensions)
    rgb_image = await loop.run_in_executor(None, cv2.cvtColor, resized_img, cv2.COLOR_BGR2RGB)

    # Convert the image to a format that can be sent back
    pil_image = Image.fromarray(rgb_image)
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return buffer

# Route to display images from the selected slide folder
@app.route("/slide/<int:slide_index>")
async def show_slide(slide_index):
    # slides = session.get('slides', [])
    current_slide = app.slides[slide_index]
    curr_slide_images = app.slide_images[current_slide]
    main_images = [
        img
        for img in curr_slide_images
        if any(x in img for x in ["proc_mask", "rf_mask", "raw"])
    ]
    stain_images = [img for img in curr_slide_images if img not in main_images]
    curr_selected_imgs = app.selected_slide_images[current_slide]
    # Update session to mark this slide as the current one
    session["current_slide"] = slide_index

    return await render_template(
        "slide.html",
        slide=current_slide,
        main_images=main_images,
        stain_images=stain_images,
        selected_files=curr_selected_imgs,
    )




@app.route("/image/<slide>/stain/<image_name>/<int:resolution>")
async def get_slide_stain_image(slide, image_name, resolution) -> Response:
    image_path = os.path.join(app.SLIDES_DIR, slide, image_name)
    buffer = await load_and_resize_image(image_path, resolution)
    if buffer is None:
        return 'Image not found', 404
    return await send_file(buffer, mimetype="image/jpeg")

@app.route("/image/<slide>/main/<image_name>")
def get_slide_main_image(slide, image_name) -> Response:
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

    return send_file(buffer, mimetype="image/jpeg")

# Route to handle image selection for confirmation
@app.route('/confirm_images', methods=['POST'])
def confirm_images():
    global slides
    selected_images = request.form.getlist('selected_images')
    current_slide_index = session['current_slide']
    current_slide = slides[current_slide_index]

    # Render the confirmation page with selected images
    return render_template('confirm.html', slide=current_slide, selected_images=selected_images)

# Route to submit images after confirmation
@app.route('/submit_images', methods=['POST'])
def submit_images():
    global slides, selected_slide_images
    selected_images = request.form.getlist('selected_images')
    current_slide_index = session['current_slide']
    current_slide = slides[current_slide_index]
    selected_slide_images[current_slide] = selected_images

    with open("selected_slide_images.json", "w") as f:
        json.dump(dict(selected_slide_images), f)
    print("Selected Images:", selected_images)
    # Store confirmed images with folder name
    # session['selected_images'].extend([f"{current_slide}/{image}" for image in selected_images])

    # Mark the slide as processed
    initialize_processed_slides()
    session['processed_slides'][current_slide_index] = True

    # Move to the next slide if available
    if current_slide_index + 1 < len(slides):
        session['current_slide'] = current_slide_index + 1
        return redirect(url_for('show_slide', slide_index=session['current_slide']))
    else:
        # If no more slides, return to home
        return redirect(url_for('start'))



if __name__ == "__main__":
    app.run(debug=True)
