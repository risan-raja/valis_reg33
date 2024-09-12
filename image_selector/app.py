import asyncio
from datetime import datetime
import glob
import os

from quart import (
    Quart,
    render_template,
    request,
    session,
    jsonify,
    url_for,
    send_file,
    Response,
    redirect,
)

from io import BytesIO
from collections import defaultdict
import json
import cv2
from PIL import Image
from motor.motor_asyncio import AsyncIOMotorClient
test=True
app = Quart(__name__)
app.secret_key = "3829749812374"  # Needed for session management
DB_PATH = "selected_images.db"
MONGO_URI = "mongodb://root:password@qd2.humanbrain.in:27017"  # Your MongoDB URI
DATABASE_NAME = "slide_selection_db"
COLLECTION_NAME = "selected_images"
SELECTION_COUNT_COLLECTION = "selection_counts"  # Collection to store image selection counts

# Initialize MongoDB client
mongo_client = AsyncIOMotorClient(MONGO_URI)
db = mongo_client[DATABASE_NAME]
selected_images_collection = db[COLLECTION_NAME]
selection_count_collection = db[SELECTION_COUNT_COLLECTION]


async def load_selected_images():
    """Loads previously selected images from MongoDB."""
    selected_slide_images = {}
    cursor = selected_images_collection.find()
    async for document in cursor:
        slide = document['slide']
        selected_slide_images[slide] =  {
            "images": document.get('images', []),
            "last_updated": document.get('last_updated', None)
        }
    return selected_slide_images


async def update_selection_count(selected_images):
    """Updates the selection count for each selected image."""
    for image in selected_images:
        await selection_count_collection.update_one(
            {"image": image},
            {"$inc": {"selection_count": 1}},  # Increment the selection count by 1
            upsert=True  # Insert the document if it doesn't exist
        )

async def get_image_selection_probabilities():
    """Fetches the selection count for all images."""
    cursor = selection_count_collection.find({})
    image_probabilities = {}
    async for document in cursor:
        image = document['image']
        count = document.get('selection_count', 0)
        image_probabilities[image] = count
    return image_probabilities

async def save_selected_images():
    """Saves the selected images to MongoDB."""
    # print("Saving selected images to MongoDB...")
    # print(app.selected_slide_images)    
    for slide, data in app.selected_slide_images.items():

        images = data["images"]
        last_updated = data.get("last_updated", None)

        # Check if the document for the slide exists in the database
        existing_doc = await selected_images_collection.find_one({"slide": slide})

        # Determine if we need to update the document
        if existing_doc:
            # Compare last_updated to ensure we only update if something has changed
            if last_updated and last_updated <= existing_doc["last_updated"]:
                continue  # Skip saving since there are no new changes

        # Update MongoDB with the new images and update the last_updated timestamp
        await selected_images_collection.update_one(
            {"slide": slide},  # Match the document by slide
            {
                "$set": {
                    "images": images,
                    "last_updated": datetime.utcnow()
                }
            },
            upsert=True  # Insert the document if it doesn't exist
        )

async def periodic_save_selected_images(interval=10):
    """Periodically saves selected images to MongoDB every `interval` seconds."""
    while True:
        await asyncio.sleep(interval)
        await save_selected_images()
        # print("Selected images saved to MongoDB.")
        app.logger.info("Selected images saved to MongoDB.")

def initialize_processed_slides():
    if "processed_slides" not in session:
        session["processed_slides"] = [False] * len(
            app.slides
        )  # False means unprocessed
        session["current_slide"] = 0
        for slide_idx, slide in enumerate(app.slides):
            if slide in app.selected_slide_images:
                if (
                    len(app.selected_slide_images[slide]) > 1
                    or "proc_mask" != app.selected_slide_images[slide]["images"][0]
                ):
                    session["processed_slides"][slide_idx] = True
                else:
                    session["processed_slides"][slide_idx] = False
    else:
        return

async def load_selected_images_from_mongo():
    """Loads selected images along with the last updated time from MongoDB."""
    selected_slide_images = {}

    # Fetch all documents from the MongoDB collection
    cursor = selected_images_collection.find({})
    async for document in cursor:
        slide = document['slide']
        images = document.get('images', [])
        last_updated = document.get('last_updated', None)  # Load the last_updated field

        # Store the images and last_updated time for each slide
        selected_slide_images[slide] = {
            "images": images,
            "last_updated": last_updated
        }

    # Return the selected images as a dictionary
    return selected_slide_images



@app.before_serving
async def setup_slides():
    # Define the path where the slides (folders) are stored
    app.SLIDES_DIR = os.environ.get(
        "SLIDES_DIR", "/storage/valis_reg/stain_sep_results"
    )
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
    # Load previously selected images from MongoDB
    app.selected_slide_images = await load_selected_images_from_mongo()

    app.save_task = asyncio.create_task(periodic_save_selected_images())



@app.before_request
def make_session_permanent():
    global selected_slide_images
    session.permanent = True
    # save_selected_images(app)


# Route to display the slides and images
@app.route("/")
async def index():
    initialize_processed_slides()
    # Update processed_slides based on whether each slide has selected images
    processed_slides = []
    for slide in app.slides:
        if slide in app.selected_slide_images and len(app.selected_slide_images[slide]["images"]) >=1:
            processed_slides.append(True)  # Mark as processed if there are selected images
        else:
            processed_slides.append(False)  # Mark as not processed if no images are selected

    # Update the session with the latest processed status
    session["processed_slides"] = processed_slides

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
    rgb_image = await loop.run_in_executor(
        None, cv2.cvtColor, resized_img, cv2.COLOR_BGR2RGB
    )

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
    # Fetch the selection probabilities (i.e., how often each image was selected)
    image_probabilities = await get_image_selection_probabilities()    
    # Sort images by selection probability (higher count images appear first)
    curr_slide_images.sort(key=lambda img: image_probabilities.get(img, 0), reverse=True)
    main_images = [
        img
        for img in curr_slide_images
        if any(x in img for x in ["proc_mask", "rf_mask", "raw"])
    ]
    stain_images = [img for img in curr_slide_images if img not in main_images]
    curr_selected_imgs = app.selected_slide_images.get(current_slide, {}).get("images", [])
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
        return "Image not found", 404
    return await send_file(buffer, mimetype="image/jpeg")


# Route to handle image selection for confirmation
@app.route("/confirm_images", methods=["POST"])
async def confirm_images():
    # selected_images = request.form.getlist("selected_images")
    selected_images_form = await request.form
    selected_images =selected_images_form.getlist("selected_images")
    current_slide_index = session["current_slide"]
    current_slide = app.slides[current_slide_index]

    # Render the confirmation page with selected images
    return await render_template(
        "confirm.html", slide=current_slide, selected_images=selected_images
    )


# Route to submit images after confirmation
@app.route("/submit_images", methods=["POST"])
async def submit_images():
    selected_images_form = await request.form
    selected_images =selected_images_form.getlist("selected_images")
    current_slide_index = session["current_slide"]
    current_slide = app.slides[current_slide_index]
     # Update selected images in app
    app.selected_slide_images[current_slide] = {
        "images": selected_images,
        "last_updated": datetime.utcnow()
    }
    # app.selected_slide_images[current_slide] = selected_images

    # with open("selected_slide_images.json", "w") as f:
    #     json.dump(app.selected_slide_images, f)
    app.logger.info("Selected Images:", selected_images)
    # Store confirmed images with folder name
    # session['selected_images'].extend([f"{current_slide}/{image}" for image in selected_images])

    # Mark the slide as processed
        # Update the selection count for each selected image
    await update_selection_count(selected_images)
    initialize_processed_slides()
    session["processed_slides"][current_slide_index] = True
    await save_selected_images()

    # await save_selected_images(app)

    # Move to the next slide if available
    if current_slide_index + 1 < len(app.slides):
        session["current_slide"] = current_slide_index + 1
        return redirect(url_for("show_slide", slide_index=session["current_slide"]))
    else:
        # If no more slides, return to home
        return redirect(url_for("start"))

@app.teardown_request
async def teardown_request(exception=None):
    """Sync all selected images to the database before the request ends or session ends."""
    app.logger.info("Tearing down request, syncing data to MongoDB...")
    await save_selected_images()
    app.logger.info("Data synced successfully!")

if __name__ == "__main__":
    app.run(debug=True)
