
# MKNet: Adaptive Kernel Mixing Depthwise Convolution for Lightweight Object Detection

MKNet is a lightweight YOLO-based object detection architecture designed for real-time assistive vision applications. The model focuses on improving the trade-off between detection accuracy, computational cost, and deployment efficiency by integrating MobileNetV3-based feature extraction, spatial-preserving downsampling, adaptive kernel mixing, and lightweight feature fusion.

## Main Contributions

- A lightweight MobileNetV3-based YOLO architecture for efficient object detection.
- An SPD-DW module for spatial information preservation during downsampling.
- A CSPCAM_MBV3 module for lightweight contextual feature aggregation.
- An AKMixDW-Partial SC module for adaptive lightweight convolution with improved receptive field flexibility.
- A GSConv-based neck for efficient multi-scale feature fusion.
- A real-time object detection design suitable for assistive vision and resource-constrained deployment scenarios.

## Results

### Model Comparison

| Model            | Params (M) | GFLOPs | mAP@0.5 | mAP@0.5:0.95 |
| ---------------- | ---------: | -----: | ------: | -----------: |
| YOLOv8n Baseline |          3.2 |      8.7 |       - |            37.3 |
| MKNet            |          1.99 |      7.8 |       46.9 |            31.6 |

### Ablation Study

| Variant              | MBV3 | SPD-DW | AKMixDW-Partial SC | GSConv | CSPCAM_MBV3 | Params (M) | GFLOPs | mAP@0.5 | mAP@0.5:0.95 |
| -------------------- | ---- | ------ | ----------- | ------------------ | ------ | ---------: | -----: | ------: | -----------: |
| YOLOv8n Baseline     |      |        |             |                    |        |          3.2 |      8.7 |       - |            37.3 |
| + MBV3               | ✓    |        |             |                    |        |          1.69 |      25.5 |       38.1 |            23.7 |
| + SPD-DW             | ✓    | ✓      |            |                    |        |          1.69 |      6.8 |       43.9 |            29.3 |
| + AKMixDW-Partial SC | ✓    | ✓      | ✓           |                   |        |          2.18 |      8.1 |       47.3 |            32 |
| + GSConv Neck        | ✓    | ✓      | ✓           | ✓                  |       |          2.12 |      7.9 |       47.2 |            31.8 |
| + CSPCAM_MBV3        | ✓    | ✓      | ✓           | ✓                  | ✓       |          1.99 |      7.8 |       46.9 |            31.6 |

## Architecture

<img width="2914" height="3644" alt="proposed structure" src="https://github.com/user-attachments/assets/c471385d-4847-4d80-bd83-fa613d0b20c8" />

## Repository Structure

```text
MKNet/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── configs/
│   ├── mknet.yaml
│   ├── mknet-nano.yaml
│   ├── mknet-small.yaml
│   └── dataset.yaml
│
├── models/
│   ├── MobileNetV3.py
│   ├── GSConv.py
│   └── common.py
│
├── scripts/
│   ├── train.py
│   ├── val.py
│   ├── predict.py
│   └── coco_eval.py
│
├── results/
│   ├── ablation_results.csv
│   ├── comparison_results.csv
│   └── figures/
│
├── weights/
│   └── MKNet.pt
│
├── docs/
│   ├── model_architecture.md
│   ├── training_guide.md
│   ├── evaluation_guide.md
│   └── deployment_guide.md
│
└── assets/
    ├── demo_images/
    ├── demo_outputs/
    └── architecture.png
````

## Installation

This project follows the standard Ultralytics YOLO installation procedure.

### 1. Create a Python environment
```bash
python -m venv mknet-env
source mknet-env/bin/activate
```


### 2. Clone this repository
```bash
git clone https://github.com/meteorsee/MKNet.git
cd MKNet
```

### 3. Install Ultralytics and required dependencies

```bash
pip install ultralytics
```

## Dataset Preparation

There are two ways to get the dataset.

First, directly use the yaml provided by Ultralytics, which will be located in the path below

```bash
lib/python3.11/site-packages/ultralytics/cfg/datasets/coco.yaml
```
NOTE: The version of python might be different time to time.

OR

You may download the Microsoft Common Object in Context from the link below,

```link
https://cocodataset.org/#download
```

The dataset should follow the standard YOLO format.

```text
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── dataset.yaml
```

Example `dataset.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 80
names:
  0: person
  1: bicycle
  2: car
  ...
```

For COCO-based experiments, please prepare the MS COCO 2017 dataset following the official YOLO/Ultralytics dataset format.

## Training

Training can be performed in two ways:

1. Using the standard Ultralytics training command.
2. Using the provided custom training script.

### Option 1: Train with Ultralytics CLI

```bash
yolo detect train model=configs/mknet.yaml data=configs/dataset.yaml imgsz=640 epochs=300 batch=32 device=0
```

Example for nano or small variants:

```bash
yolo detect train model=configs/mknet-nano.yaml data=configs/dataset.yaml imgsz=640 epochs=300 batch=32 device=0
```

```bash
yolo detect train model=configs/mknet-small.yaml data=configs/dataset.yaml imgsz=640 epochs=300 batch=32 device=0
```

### Option 2: Train with custom setup

```bash
python scripts/train.py
```

The custom training script is optional and mainly provided for users who want to reproduce the exact experimental setup used in this work.

## Validation

Validation is performed using COCO-style evaluation with `pycocotools`.

### Standard Ultralytics validation

```bash
yolo detect val model=weights/mknet.pt data=configs/dataset.yaml imgsz=640 device=0
```

### COCO-style validation using pycocotools

```bash
python scripts/coco_eval.py \
  --weights weights/mknet.pt \
  --data configs/dataset.yaml \
  --imgsz 640 \
  --ann /path/to/instances_val2017.json
```

The `pycocotools` evaluation reports standard COCO metrics such as:

```text
AP@[0.50:0.95]
AP@0.50
AP@0.75
AP_small
AP_medium
AP_large
AR@[0.50:0.95]
```

## Inference

Run inference on an image:

```bash
yolo detect predict model=weights/mknet.pt source=assets/demo_images/example.jpg imgsz=640
```

Run inference on a video:

```bash
yolo detect predict model=weights/mknet.pt source=demo.mp4 imgsz=640
```

Or using the custom script:

```bash
python scripts/predict.py \
  --weights weights/mknet.pt \
  --source assets/demo_images/
```

## Pretrained Weights

Pretrained weights will be released through GitHub Releases, Google Drive, or Hugging Face.

| Model       | Dataset   | Weights     |
| ----------- | --------- | ----------- |
| MKNet       | COCO 2017 | Coming soon |

Large weight files are not directly stored in this repository to keep the repository lightweight.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{meteorsee2026mknet,
  title={MKNet: Adaptive Kernel Mixing Depthwise Convolution for Lightweight Object Detection},
  author={Keng Lek, See, Wai Kong, Lee, Soung Yue, Liew, Shen Khang, Teoh, Hock Guan, Goh, Ramanchandra, Achar},
  journal={To be updated},
  year={2026}
}
```

## Acknowledgement

This work is developed based on the YOLO object detection framework and follows the Ultralytics training and deployment workflow. We also acknowledge the MS COCO dataset and related lightweight object detection studies that inspired the design of MKNet.

## License

This project is released under the selected license. Please refer to the `LICENSE` file for more details.

```

One important suggestion: since your title is **“Adaptive Kernel Mixing Depthwise Convolution for Lightweight Object Detection”**, your README should make **AKMixDW** look like the main contribution, while MBV3, SPD-DW, CSPCAM, and GSConv are supporting architectural components. This makes the novelty clearer.
```
