import cv2
import os
import sys
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description="ColorThresholder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--name",
        help="name of the file for thresholding",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-pc",
        "--pointcloud",
        help="Specified File is a pointcloud.",
        action="store_true",
        required=False,
    )

    return parser.parse_args()


def create_binary_pic(file_name, mask, green_pixels, img):
    green_pixels[mask > 0] = (255, 255, 255)
    img[mask == 0] = (0, 0, 0)

    result = cv2.bitwise_or(img, green_pixels)
    cv2.imwrite(f"{file_name.split('.')[0]}_binary.tiff", result)
    cv2.imwrite(
        f"{file_name.split('.')[0]}_binary.jpeg",
        result,
        [int(cv2.IMWRITE_JPEG_QUALITY), 200],
    )


def create_segmented_color_pic(file_name, mask, green_pixels, img):
    green_pixels[mask > 0] = (0, 0, 0)
    img[mask == 0] = (255, 255, 255)

    result = cv2.bitwise_or(img, green_pixels)
    cv2.imwrite(
        f"{file_name.split('.')[0]}_thresholded.jpeg",
        result,
        [int(cv2.IMWRITE_JPEG_QUALITY), 200],
    )
    cv2.imwrite(f"{file_name.split('.')[0]}_thresholded.tiff", result)


def main():
    args = get_args()
    actual_name = args.name
    img = cv2.imread(actual_name)
    os.mkdir(actual_name.split(".")[0])
    os.chdir(actual_name.split(".")[0])
    # make a jpeg copy of the actual file for easy comparison
    cv2.imwrite(
        actual_name.split(".")[0] + ".jpeg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 200]
    )

    # Need to shift from BGR to HSV for more accurate filtering
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color Range. MODIFY IF REQUIRED
    lower_green = np.array([34, 22, 22])
    upper_green = np.array([86, 255, 255])

    # Code to isolate green
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    green_pixels = cv2.bitwise_and(img, img, mask=mask)

    create_binary_pic(actual_name, mask, green_pixels, img)
    create_segmented_color_pic(actual_name, mask, green_pixels, img)

    os.chdir("../")


# --------------------------------------------------
if __name__ == "__main__":
    main()
