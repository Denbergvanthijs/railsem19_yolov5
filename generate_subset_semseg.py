import os
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from generate_subset import copy_images, generate_folders, train_val_test_split


def get_relevant_data_semseg(fp_folder, class_number):
    """Reads all Semantic Segmentation label images in a folder. Returns list of all filenames containing `class_number`."""
    return [file.split(".")[0] for file in tqdm(os.listdir(fp_folder), desc="Checking labels") if class_number in cv2.imread(os.path.join(fp_folder, file), cv2.IMREAD_GRAYSCALE)]


def semseg2bbox(contour, classnumber: int = 11, img_width: int = 1920, img_height: int = 1080) -> Tuple[int, int, int, int, int]:
    x_min, y_min, width, height = cv2.boundingRect(contour)
    x_center = (width / 2 + x_min) / img_width
    y_center = (height / 2 + y_min) / img_height

    return [classnumber, x_center, y_center, width/img_width, height/img_height]


def generate_all_labels_semseg(filenames_train_val_test: Tuple[List[str], List[str], List[str]], fp_in: str = "./rs19_val",
                               fp_out: str = "./rs19_person", class_number_semseg: int = 11, class_number_bbox: int = 0, min_pixel_area: int = 400) -> None:
    folders_out = (os.path.join(fp_out, "labels/train"), os.path.join(fp_out, "labels/val"), os.path.join(fp_out, "labels/test"))

    for dataset, folder in zip(filenames_train_val_test, folders_out):
        for filename in tqdm(dataset, desc=f"Generating labels: {folder}"):
            path_in = os.path.join(fp_in, "uint8", "rs19_val", filename + ".png")  # Filepath to the label in the RailSem19 dataset
            path_out = os.path.join(folder, filename + ".txt")  # Filepath to copy the label to

            img = cv2.imread(path_in, cv2.IMREAD_GRAYSCALE)  # Read the Semantic Segmentation label
            img_height, img_width = img.shape[:2]  # Shape should always be of length 2 since img is grayscale

            binary_image = (img == class_number_semseg).astype(np.uint8)
            contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            data_txt = []  # List of all objects in a single image
            for contour in contours:
                # if cv2.contourArea(contour) > min_pixel_area:  # TODO: Only save big enough instances
                bbox = semseg2bbox(contour, class_number_bbox, img_width, img_height)
                data_txt.append(" ".join(str(i) for i in bbox))  # Add instance to list of instances of this image

            data_txt = "\n".join(data_txt)  # Combine all objects into single string
            with open(path_out, "w") as file_out:
                file_out.write(data_txt)


if __name__ == "__main__":
    fp_railsem19_dataset = "./rs19_val"
    fp_new_dataset = "./rs19_person_semseg"

    generate_folders(fp_rootdir=fp_new_dataset)  # Generate empty folders

    # List of all filenames with the class `Human`
    relevant_data = get_relevant_data_semseg(fp_folder=os.path.join(fp_railsem19_dataset, "uint8", "rs19_val"), class_number=11)

    filenames_train_val_test = train_val_test_split(relevant_data, seed=42)  # Split filenames in train / val / test split
    # Copy relevant images to corresponding folders
    copy_images(filenames_train_val_test, fp_in=fp_railsem19_dataset, fp_out=fp_new_dataset)

    generate_all_labels_semseg(filenames_train_val_test, fp_in=fp_railsem19_dataset, fp_out=fp_new_dataset,
                               class_number_semseg=11, class_number_bbox=0, min_pixel_area=400)
    print([len(l) for l in filenames_train_val_test])
