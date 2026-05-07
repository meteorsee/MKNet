from ultralytics import YOLO

def main():
    model = YOLO("MobileNetv3_w_SPD-DW.yaml")

    # Train
    model.train(
        # Used for RGB training
        data = "./coco.yaml",
        amp=False,
        batch = 64,
        epochs=200,
        imgsz=640,
        device="cuda:0",
        optimizer="SGD",   # better convergence on small/custom datasets
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        cos_lr=True,
        warmup_epochs=3, # modified from 10 to 3
        cache=False,

        # ✅ augmentations (heavy → light via close_mosaic)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.2, scale=0.4, shear=0.0,  # NOTE: scale is a single float in v8
        mosaic=0.7, mixup=0.05,
        # need to try change the close_mosaic value to 20~40, default value is 10
        close_mosaic=20,       # turn mosaic off in the last 80 epochs
        label_smoothing=0.05,

        # ✅ loss weights (rebalance toward recall)
        box=7.5, cls=0.4, dfl=1.5,

        # Number of epochs to wait without training improvement
        patience = 10,

        save = True,
        save_period = 20, #save a checkpoint every 20 epochs

    )

    # ## Validate
    model.val(
        # Used for RGB validation
        data = "./coco.yaml",
        device="cuda:0"
    )



if __name__ == "__main__":
    main()
