import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    data = pd.read_csv("./results/trained/trained_best.csv")  # Generated with `generate_baseline_results.py`
    data["FPS"] = 1000 / data["Inference time"]
    data["Model"] = data["Model"].str[:-3]
    data["Model"] = data["Model"].str.replace("trained_best", "Getraind model")  # Change text for legend
    data["Model"] = data["Model"].str.replace("yolov5s", "YOLOv5s baseline")
    print(data)

    sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=1.5, rc={"lines.linewidth": 2})

    plot_mAP = sns.barplot(data=data, x="Dataset", y="mAP@0.50;0.05;0.95", hue="Model", hue_order=["YOLOv5s baseline", "Getraind model"])
    plot_mAP.set(ylabel="mAP @ 0.50;0.05;0.95 (hoger is beter)")
    plot_mAP.legend(loc="upper right")
    plt.savefig("./results/trained/trained_mAP.png")

    plt.clf()

    plot_time = sns.barplot(data=data, x="Dataset", y="FPS", hue="Model", hue_order=["YOLOv5s baseline", "Getraind model"])
    plot_time.set(ylabel="Aantal beelden per seconde (hoger is beter)", ylim=(0, 60))
    plot_time.legend(loc="upper right")
    plt.axhline(y=30, ls="--", c="red")

    plt.savefig("./results/trained/trained_time.png")
