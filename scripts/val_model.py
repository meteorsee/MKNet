# REMOVE: from faster_coco_eval import COCO, COCOeval_faster
# ADD:
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train12/weights/last.pt")

    # Validate using pycocotools
    model.val(
        data="coco.yaml",
        device="cuda:0"
    )

if __name__ == "__main__":
    main()

