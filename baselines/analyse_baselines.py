import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    data_gpu = pd.read_csv("./baselines/results.csv")  # Generated with `generate_baseline_results.py`
    data_cpu = pd.read_csv("./baselines/results_cpu.csv")
    data = pd.concat((data_gpu, data_cpu), ignore_index=True)
    data["FPS"] = 1000 / data["Inference time"]
    data["Model"] = data["Model"].str[:-3]
    print(data)

    sns.set(style="whitegrid", font="serif")

    plot_mAP = sns.relplot(data=data, x="Model", y="mAP@0.50;0.05;0.95", col="Dataset", kind="line")
    plot_mAP.set_titles("Dataset: {col_name}")
    plot_mAP.set_ylabels("mAP @ 0.50;0.05;0.95 (hoger is beter)", clear_inner=False)
    plot_mAP.set_xlabels("YOLOv5 baseline modellen", clear_inner=False)
    plot_mAP.set(ylim=(0, 0.30))
    plt.savefig("./baselines/baselines_mAP.png")

    plot_time = sns.relplot(data=data, x="Model", y="FPS", col="Dataset", kind="line")
    plot_time.set_titles("Dataset: {col_name}")
    plot_time.set_ylabels("Aantal beelden per seconde (hoger is beter)", clear_inner=False)
    plot_time.set_xlabels("YOLOv5 baseline modellen", clear_inner=False)
    plot_time.set(ylim=(0, 60))

    plot_time.map(plt.axhline, y=30, ls='--', c='red')
    plt.gca().text(-6.33, 30.5, "Realtime\n(30 fps)", color="red")

    plt.savefig("./baselines/baselines_time.png")
