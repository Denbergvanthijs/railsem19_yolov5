import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    data_gpu = pd.read_csv("./results/baselines/baselines_gpu.csv")  # Generated with `generate_results.py`
    data_cpu = pd.read_csv("./results/baselines/baselines_cpu.csv")
    data = pd.concat((data_gpu, data_cpu), ignore_index=True)
    data["FPS"] = 1000 / data["Inference time"]
    data["Model"] = data["Model"].str[:-3]
    print(data)

    sns.set_theme(context="paper", style="whitegrid", font="serif", font_scale=1.5, rc={"lines.linewidth": 2})

    plot_mAP = sns.relplot(data=data, x="Model", y="mAP@0.50;0.05;0.95", col="Dataset", kind="line")
    plot_mAP.set_titles("Dataset: {col_name}")
    plot_mAP.set_ylabels("mAP @ 0.50;0.05;0.95 (hoger is beter)", clear_inner=False)
    plot_mAP.set_xlabels("", clear_inner=False)
    plot_mAP.set(ylim=(0, None))
    plt.savefig("./results/baselines/baselines_mAP.png")

    plot_time = sns.relplot(data=data, x="Model", y="FPS", col="Dataset", kind="line")
    plot_time.set_titles("Dataset: {col_name}")
    plot_time.set_ylabels("Aantal beelden per seconde (hoger is beter)", clear_inner=False)
    plot_time.set_xlabels("", clear_inner=False)
    plot_time.set(ylim=(0, None))

    plot_time.map(plt.axhline, y=30, ls='--', c='red')
    plt.gca().text(-6.9, 30.5, "Realtime\n(30 fps)", color="red")

    plt.savefig("./results/baselines/baselines_time.png")
