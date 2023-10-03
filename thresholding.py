import cv2
import os
import sys
import argparse
import numpy as np
import pylas


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


def tiff_thresholding(actual_name):
    img = cv2.imread(actual_name)
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


def laz_thresholding(actual_name):
    las = laspy.read(actual_name)
    os.chdir(actual_name.split(".")[0])

    # Extract the color values into numpy arrays and normalize them
    R = las.red / 65535.0
    G = las.green / 65535.0
    B = las.blue / 65535.0

    # Stack the normalized RGB values to create an image
    RGB = np.dstack((R, G, B))
    HSV = cv2.cvtColor((RGB * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)

    # Color Range. MODIFY IF REQUIRED
    lower_val = np.array([34, 22, 22])
    upper_val = np.array([86, 255, 255])

    # Create a mask for points that are "green"
    mask = (
        (HSV[..., 0] >= lower_val[0])
        & (HSV[..., 0] <= upper_val[0])
        & (HSV[..., 1] >= lower_val[1])
        & (HSV[..., 1] <= upper_val[1])
        & (HSV[..., 2] >= lower_val[2])
        & (HSV[..., 2] <= upper_val[2])
    )

    # Apply the mask to get the points that are "green"
    green_points = las.points[mask.flatten()]

    # Create a new .las file with only the green points
    new_las = laspy.create(point_format=las.header.point_format_id)
    new_las.points = green_points

    # Write the new .las file
    new_las.write(f"{actual_name.split('.')[0]}_green_only.las")
    las = pylas.read(actual_name)


def main():
    args = get_args()
    actual_name = args.name
    os.mkdir(actual_name.split(".")[0])
    if not args.pointcloud:
        tiff_thresholding(actual_name)
    else:
        laz_thresholding(actual_name)


# --------------------------------------------------
if __name__ == "__main__":
    main()
