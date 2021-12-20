import json
import os
import shutil
import warnings
from typing import List, Tuple

from sklearn.model_selection import train_test_split


def read_json(fp: str) -> dict:
    """Helper. Reads a JSON file and returns content.

    :param fp: The filepath to the JSON file.
    :dtype fp: str
    """
    with open(fp) as f:
        f_data = json.load(f)
    return f_data


def generate_folders(fp_rootdir: str = "./rs19_person") -> None:
    """"Generates folders to save the new dataset in. Follows YOLOv5 standard.

    :param fp_rootdir: Foldername of the new dataset.
    :dtype fp_rootdir: str
    """
    for folder in (os.path.join(fp_rootdir, "images/train"), os.path.join(fp_rootdir, "images/val"), os.path.join(fp_rootdir, "images/test"),
                   os.path.join(fp_rootdir, "labels/train"), os.path.join(fp_rootdir, "labels/val"), os.path.join(fp_rootdir, "labels/test")):
        if not os.path.exists(folder):  # Only makes new a folder if it doesn't exist already
            os.makedirs(folder)
        else:
            warnings.warn(f"Folder {folder} already exists. Make sure there's no train/val/test contamination"
                          " from a previously generated subset with a (possible) different seed.", Warning, stacklevel=2)


def read_all_labels(fp_folder: str = "./rs19_val/jsons/rs19_val") -> List[dict]:
    """Loads the data of all JSON files from a provided folder and returns the contents in a list.

    :param fp_folder: Filepath to the folder containing JSON files.
    :dtype fp_folder: str
    """
    return [read_json(os.path.join(fp_folder, file)) for file in os.listdir(fp_folder)]


def get_relevant_data(json_data: List[dict], classname: str = "person") -> List[str]:
    """Generates a list of all filenames with at least one instance of the provided class.

    :param json_data: List of dicts, each dict is JSON data in RailSem19 format.
    :dtype json_data: List[dict], generaly the output of `read_all_labels()`.
    """
    return [label["frame"] for label in json_data if classname in (l["label"] for l in label["objects"])]


def train_val_test_split(relevant_data: List[str], seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Splits a list in seperate train, validate and test datasets.

    TODO: add params for train / val / test sizes

    :param relevant_data: The list to be divided, generaly a list of filenames.
    :dtype relevant_data: List[str]
    """
    relevant_data = sorted(relevant_data)  # Ensures the input to the split is always the same
    train, rest = train_test_split(relevant_data, test_size=0.3, shuffle=True, random_state=seed)  # 70% to train
    val, test = train_test_split(rest, test_size=0.5, shuffle=True, random_state=seed)  # Divide the remaining 30% equally over val and test

    return train, val, test


def copy_images(filenames_train_val_test: Tuple[List[str], List[str], List[str]], fp_in: str = "./rs19_val", fp_out: str = "./rs19_person") -> None:
    """Copies images from RailSem19 dataset to the correct folders. In YOLOv5 folder format.

    :param filenames_train_val_test: Tuple of lists containing the filenames (without extension) of the images,
                                     generaly the output of `train_val_test_split()`.
    :dtype filenamse_train_val_test: Tuple[List[str], List[str], List[str]]
    :param fp_in: Filepath of the folder where the images are currently stored.
    :dtype fp_in: str
    :param fp_out: Root directory to place the contents of the three datasets in.
                   Do not forget to first make these folders with `generate_folders()`.
    :dtype fp_out: str
    """
    folders_out = (os.path.join(fp_out, "images/train"), os.path.join(fp_out, "images/val"), os.path.join(fp_out, "images/test"))

    for dataset, folder in zip(filenames_train_val_test, folders_out):
        for filename in dataset:
            path_in = os.path.join(fp_in, "jpgs", "rs19_val", filename + ".jpg")  # Filepath to the image in the RailSem19 dataset
            path_out = os.path.join(folder, filename + ".jpg")  # Filepath to copy the image to

            if not os.path.isfile(path_out):  # Skip copying if file already exists, performance improvement
                shutil.copyfile(path_in, path_out)


def poly2bbox(points: List[Tuple[int, int]], classnumber: int = 0,
              img_width: int = 1920, img_height: int = 1080) -> Tuple[int, int, int, int, int]:
    """Converts N by 2 list to 4 coordinate bbox in YOLOv5 format.
    Format: [classnumber, x_center, y_center, width, height]
    Based on: https://stackoverflow.com/a/46336730

    :param points: Polygon points of an object, in RailSem19 format.
    :dtype points: List[Tuple[int, int]]
    :param classnumber: The number of the class, to directly output as first variable in the output-list
    :dtype classnumber: int
    :param img_width: The width of the image containing the bbox
    :dtype img_width: int
    :param img_height: The height of the image containing the bbox
    :dtype img_height: int
    """
    x_coords, y_coords = zip(*points)  # Transpose N by 2 list to a 2 by N list
    x_min = min(x_coords)  # Leftbound of bbox
    y_min = min(y_coords)  # Lowerbound of bbox
    width = (max(x_coords) - x_min) / img_width  # Width of bbox
    height = (max(y_coords) - y_min) / img_height  # Height of bbox
    x_center = width / 2 + x_min / img_width
    y_center = height / 2 + y_min / img_height

    return [classnumber, x_center, y_center, width, height]


def generate_all_labels(filenames_train_val_test: Tuple[List[str], List[str], List[str]], fp_in: str = "./rs19_val",
                        fp_out: str = "./rs19_person", relevant_classes: dict = {"person": 0}) -> None:
    """Converts all JSON labels in RailSem19 format to txt labels in YOLOv5 format. Writes txts to corresponding folders.

    :param filenames_train_val_test: Tuple of lists containing the filenames (without extension) of the images,
                                     generaly the output of `train_val_test_split()`.
    :dtype filenamse_train_val_test: Tuple[List[str], List[str], List[str]]
    :param fp_in: Filepath of the folder where the images are currently stored.
    :dtype fp_in: str
    :param fp_out: Root directory to place the contents of the three datasets in.
                   Do not forget to first make these folders with `generate_folders()`.
    :dtype fp_out: str
    :param relevant_classes: Dict of all classes that are worth saving in YOLOv5 format
    :dtype relevant_classes: dict, keys are the classnames in RailSem19 format, values are the classnumbers in YOLOv5 format.
    """
    folders_out = (os.path.join(fp_out, "labels/train"), os.path.join(fp_out, "labels/val"), os.path.join(fp_out, "labels/test"))

    for dataset, folder in zip(filenames_train_val_test, folders_out):
        for filename in dataset:
            path_in = os.path.join(fp_in, "jsons", "rs19_val", filename + ".json")  # Filepath to the label in the RailSem19 dataset
            path_out = os.path.join(folder, filename + ".txt")  # Filepath to copy the label to

            data = read_json(path_in)  # Read the JSON annotation
            img_width, img_height = data["imgWidth"], data["imgHeight"]

            data_txt = []  # List of all objects in a single image
            for obj in data["objects"]:
                if obj["label"] in relevant_classes.keys():  # Only save relevant classes to the txt
                    polygon = obj["polygon"]  # TODO: Add support if bbox is known instead of polygon
                    bbox = poly2bbox(polygon, relevant_classes[obj["label"]], img_width, img_height)

                    data_txt.append(" ".join(str(i) for i in bbox))  # Add instance to list of instances of this image

            data_txt = "\n".join(data_txt)  # Combine all objects into single string
            if not os.path.isfile(path_out):  # Skip writing if file already exists, performance improvement
                with open(path_out, "w") as file_out:
                    file_out.write(data_txt)


if __name__ == "__main__":
    fp_railsem19_dataset = "./data/rs19_val"
    fp_new_dataset = "./data/rs19_person"

    generate_folders(fp_rootdir=fp_new_dataset)  # Generate empty folders
    json_data = read_all_labels(fp_folder=os.path.join(fp_railsem19_dataset, "jsons", "rs19_val"))  # Load all JSON labels

    # List of all filenames with the class `person`
    data_person = get_relevant_data(json_data=json_data, classname="person")
    # List of all filenames with the class `person-group`
    data_person_group = get_relevant_data(json_data=json_data, classname="person-group")

    relevant_data = set(data_person) - set(data_person_group)  # Only keep images where `person` is present and `person-group` is not

    filenames_train_val_test = train_val_test_split(relevant_data, seed=42)  # Split filenames in train / val / test split
    copy_images(filenames_train_val_test, fp_in=fp_railsem19_dataset,
                fp_out=fp_new_dataset)  # Copy relevant images to corresponding folders

    generate_all_labels(filenames_train_val_test, fp_in=fp_railsem19_dataset, fp_out=fp_new_dataset, relevant_classes={"person": 0})
    print([len(l) for l in filenames_train_val_test])
