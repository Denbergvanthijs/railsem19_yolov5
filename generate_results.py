import csv

path_to_yolov5 = "./yolov5"
# fp_output_csv = "./results/baselines/baselines_gpu.csv"
# fp_output_csv = "./results/baselines/baselines_cpu.csv"
fp_output_csv = "./results/trained/trained_best.csv"
# models = ("./models/yolov5n.pt", "./models/yolov5s.pt", "./models/yolov5m.pt", "./models/yolov5l.pt")
# models = ("./models/yolov5x.pt", )
models = ("./models/yolov5s.pt", "./models/trained_best.pt")
datasets = ("train", "val", "test")
yaml_data = "./data/rs19_person.yaml"
cpu = False

# python ./yolov5/val.py --data "./yolov5/data/coco128.yaml" --weights "yolov5n.pt" --batch-size 1 --task val --workers 1
if __name__ == "__main__":
    from yolov5 import val

    rows = [("Model", "Dataset", "Precision", "Recall", "mAP@0.50", "mAP@0.50;0.05;0.95",
             "Pre-process time", "Inference time", "NMS time per image")]
    for model in models:
        for dataset in datasets:
            results, _, timings = val.run(data=yaml_data,
                                          weights=model,
                                          batch_size=1,
                                          task=dataset,
                                          workers=1,
                                          single_cls=True)

            rows.append((model.split("/")[-1], dataset) + results[:-3] + timings)

    with open(fp_output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)
