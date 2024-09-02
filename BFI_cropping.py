{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b5105e1-9d58-4954-9255-1e60beb72e8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "from PIL import Image\n",
    "\n",
    "thresh = 150\n",
    "\n",
    "def process_BFI_images(input_folder, output_folder):\n",
    "    \"\"\"Processes all images in the input folder and saves the results to the output folder.\n",
    "\n",
    "    Args:\n",
    "        input_folder (str): The path to the input folder containing the images.\n",
    "        output_folder (str): The path to the output folder where the   \n",
    " processed images will be saved.\n",
    "    \"\"\"\n",
    "\n",
    "    if not os.path.exists(output_folder):\n",
    "        os.makedirs(output_folder)\n",
    "\n",
    "    for filename in os.listdir(input_folder):\n",
    "        if filename.endswith(('.jpg')):\n",
    "            image_path = os.path.join(input_folder, filename)   \n",
    "\n",
    "            output_path = os.path.join(output_folder, filename)\n",
    "\n",
    "            # Load the image using Pillow\n",
    "            with Image.open(image_path) as img:\n",
    "                # Perform your image processing operations here\n",
    "                # For example, you could resize, rotate, apply filters, etc.\n",
    "                # Replace this with your specific processing logic\n",
    "                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "                gray=img[:,:,1]\n",
    "                bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]\n",
    "                contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
    "                largest_contour = max(contours, key=cv2.contourArea)\n",
    "                # Get the bounding rectangle of the brain contour\n",
    "                x, y, w, h = cv2.boundingRect(largest_contour)\n",
    "\n",
    "                # Crop the image using the bounding rectangle\n",
    "                cropped_image = img[y:y+h, x:x+w]\n",
    "                processed_img = cropped_image\n",
    "\n",
    "                # Save the processed image\n",
    "                processed_img.save(output_path)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    input_folder = \"/home/projects/imgseg/keerthi_data/BFIW_original/\"  # Replace with your input folder path\n",
    "    output_folder = \"/home/projects/imgseg/valis_reg/cropped_imgs_new/\"  # Replace with your output folder path\n",
    "    process_images(input_folder, output_folder)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
