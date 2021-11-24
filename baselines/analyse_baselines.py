import glob

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns


def analyse_dataset(dataset: str, n_images: int):
    """Analyse the text files generated from running a study on a single of the three datasets."""
    files = glob.glob(f"./baselines/{dataset}/*.txt")
    column_names = ["precision", "recall", "mAP@0.50", "mAP@0.50;0.05;0.95", "time preprocessing", "total inference time", "NMS time"]

    data = []
    for fp in files:  # One model for each file
        file_data = pd.read_csv(fp, sep="\s+", usecols=(0, 1, 2, 3, 7, 8, 9), names=column_names)
        file_data["model"] = fp.split("_")[-1].split(".")[0]  # Remove substring and file extension
        file_data["input_size"] = ["1280x736", "1920x1088"]
        data.append(file_data)

    df = pd.concat(data, axis=0, ignore_index=True)  # Combine the results of all models
    df["average inference time"] = df["total inference time"] / n_images  # Divide total inference time by amount of images
    df = df[["model", "input_size", "mAP@0.50;0.05;0.95", "average inference time"]]  # Select relevant data

    return df


if __name__ == "__main__":
    data_train = analyse_dataset("train", 100)
    data_train["Dataset"] = "train"
    data_val = analyse_dataset("val", 21)
    data_val["Dataset"] = "val"
    data_test = analyse_dataset("test", 22)
    data_test["Dataset"] = "test"

    order_dict = {"yolov5n": 0, "yolov5s": 1, "yolov5m": 2, "yolov5l": 3, "yolov5x": 4}  # Order the labels on the x axis

    data = pd.concat([data_train, data_test, data_val], axis=0, ignore_index=True)  # Combine the results of all models
    data = data.sort_values(by="model", key=lambda i: [order_dict[j] for j in i.values])  # Order the labels on the x axis
    print(data)

    sns.set(style="whitegrid", font="serif")

    plot_mAP = sns.relplot(data=data, x="model", y="mAP@0.50;0.05;0.95", hue="Dataset",
                           col="input_size", kind="line", hue_order=["train", "val", "test"])
    plot_mAP.set_titles("Input-formaat: {col_name}")
    plot_mAP.set_ylabels("mAP @ 0.50;0.05;0.95 (hoger is beter)", clear_inner=False)
    plot_mAP.set_xlabels("YOLOv5 baseline modellen", clear_inner=False)
    plot_mAP.set(ylim=(0, 0.25))
    plt.savefig("./baselines/baselines_mAP.png")

    plot_time = sns.relplot(data=data, x="model", y="average inference time", hue="Dataset",
                            col="input_size", kind="line", hue_order=["train", "val", "test"])
    plot_time.set_ylabels("Inferentie-tijd per afbeelding (ms, lager is beter)", clear_inner=False)
    plot_time.set_xlabels("YOLOv5 baseline modellen", clear_inner=False)
    plot_time.set(ylim=(0, 40))

    plot_time.map(plt.axhline, y=33, ls='--', c='red')
    plt.gca().text(-4.75, 33.5, "Realtime\n(30 fps = 33.3 ms)", color="red")  # TODO: Submit bugreport due to order of execution

    plot_time.set_titles("Input-formaat: {col_name}")
    plt.savefig("./baselines/baselines_time.png")
