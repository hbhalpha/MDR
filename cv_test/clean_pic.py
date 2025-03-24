import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
# Set the directory containing the images and the output directory
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Split images into sub-images based on contours")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing full-page images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save cropped sub-images')
    parser.add_argument('--ext', type=str, default='.png', help='Image file extension to process (e.g., .png, .jpg)')
    parser.add_argument('--min_area', type=int, default=5000, help='Minimum contour area to be considered')
    parser.add_argument('--max_splits', type=int, default=10, help='Maximum number of cropped images to save per input image')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(args.input_dir, f"*{args.ext}"))

    for image_path in tqdm(image_paths):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image_output_dir = os.path.join(args.output_dir, image_name)

        if os.path.exists(image_output_dir):
            continue
        os.makedirs(image_output_dir)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 170], dtype=np.uint8)
        upper_white = np.array([180, 60, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        mask_nonwhite = cv2.bitwise_not(mask_white)
        image_no_white = cv2.bitwise_and(image, image, mask=mask_nonwhite)

        gray = cv2.cvtColor(image_no_white, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 120)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        saved_images = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < args.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.3 < aspect_ratio < 2:
                cropped = image[y:y + h, x:x + w]
                saved_images.append(cropped)

        for idx, cropped_image in enumerate(saved_images[:args.max_splits]):
            output_image_path = os.path.join(image_output_dir, f"{idx + 1}.jpg")
            cv2.imwrite(output_image_path, cropped_image)

        # Optional log:
        # print(f"Processed {image_name}, extracted {len(saved_images[:args.max_splits])} images.")
if __name__ == '__main__':
    main()
