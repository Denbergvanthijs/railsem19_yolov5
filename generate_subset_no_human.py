import glob
import os
import shutil

from tqdm import tqdm

from generate_subset_semseg import get_relevant_data_semseg

if __name__ == "__main__":
    """This script generates a subset containing all images without instances of the Semantic Segmentation Human class."""

    fp_railsem19_dataset = "./data/rs19_val"
    fp_out = "./data/rs19_no_human"
    class_human = 11

    if not os.path.exists(fp_out):  # Only makes new a folder if it doesn't exist already
        os.makedirs(fp_out)

    images_all = {os.path.splitext(os.path.basename(file))[0] for file in glob.glob(fp_railsem19_dataset + "/jpgs/rs19_val/*.jpg")}
    images_with_human = set(get_relevant_data_semseg(fp_folder=os.path.join(
        fp_railsem19_dataset, "uint8", "rs19_val"), class_number=class_human))
    relevant_filenames = images_all - images_with_human

    for filename in tqdm(relevant_filenames, desc="Copying images"):
        # Filepath to the image in the RailSem19 dataset
        path_in = os.path.join(fp_railsem19_dataset, "jpgs", "rs19_val", filename + ".jpg")
        path_out = os.path.join(fp_out, filename + ".jpg")  # Filepath to copy the image to

        if not os.path.isfile(path_out):  # Skip copying if file already exists, performance improvement
            shutil.copyfile(path_in, path_out)
