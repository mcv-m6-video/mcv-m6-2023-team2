import sys
import time
import argparse
from ultralytics import YOLO

sys.path.append('../')
from utils import load_config


def finetune(cfg):

    params = {
        'batch_size': 64,
        'image_size' : 512,
        'learning_rate': 0.0005,
    }

    batch_size = int(params['batch_size'])
    learning_rate = float(params['learning_rate'])
    image_size = int(params['image_size'])

    run_name = f"bs-{batch_size}_lr-{learning_rate}_imgsz-{image_size}"

    model = YOLO(cfg["model_weights"])

    start_time = time.time()

    model.train(
        data=cfg["finetune_yolo_config"],
        batch=batch_size, imgsz=image_size, lr0=learning_rate, cos_lr=False,
        optimizer="Adam", patience=10, epochs=cfg["epochs"],
        device=0, workers=0, val=True, cache=True,
        pretrained=True, name=run_name, save_period=10,
    )

    final_time = time.time() - start_time
    h = final_time // 3600
    s = final_time % 3600
    min = s // 60
    s %= 60
    print('Finetuning time: {:02d}hours {:02d}min {:02d}s'.format(int(h), int(min), int(s)))
    val_results = model.val()
    print("Metrics (validation): ", val_results)

    success = model.export(format="-")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../config/finetune_detector.yaml")
    args = parser.parse_args(sys.argv[1:])

    cfg = load_config(args.config)

    finetune(cfg)
