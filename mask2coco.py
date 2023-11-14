# Script to save mask prediction in COCO format (Instance segmentation)
# Adjusted from https://github.com/Mortyzhang/Mask2polygon_tool/blob/main/Mask2polygon.py

import json
from pathlib import Path

# import pandas as pd
import mmcv
import numpy as np
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules


def init_coco(model):
    # prepare coco format
    dataset = {
        "info": {},
        "license": {},
        "images": [],
        "annotations": [],
        "categories": [],
    }

    labels = model.dataset_meta.get("classes")
    for i, label in enumerate(labels):
        class_index = {i: label}

    for _, k in enumerate(list(class_index.keys())):
        dataset["categories"].append(
            {"id": k, "name": class_index[k], "supercategory": None}
        )

    return dataset


def create_sub_mask_annotation(
    sub_mask, image_id, category_id, annotation_id, is_crowd, vol
):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(
        np.array(sub_mask), 0.5, positive_orientation="low"
    )

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords)
        segmentation = np.maximum(segmentation, 0).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    if multi_poly.bounds == ():
        return "skip"
    x, y, max_x, max_y = multi_poly.bounds
    # x = max(0, x)
    # y = max(0, y)
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    area = multi_poly.area

    annotation = {
        "segmentation": segmentations,
        "iscrowd": is_crowd,
        "image_id": image_id,
        "category_id": category_id,
        "id": annotation_id,
        "bbox": bbox,
        "area": area,
        "volume": vol.item()
    }

    return annotation


# Choose to use a config and initialize the detector
config_file = "projects/HiFive/configs/rtmdet-ins_s_hifive_reg_coco.py"

# Setup a checkpoint file to load
checkpoint_file = "work_dirs/rtmdet-ins_s_hifive_reg_coco/20230825_123656/best_hifive_volume_mse_epoch_279.pth"

# register all modules in mmdet into the registries
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_detector(
    config_file, checkpoint_file, device="cuda:0"
)  # or device='cuda:0'
thres = 0.4

# image path for inference
data_path = "/data/datasets/HiFive-E3-7/Without-Big-Leak/HIFIVE_Data_train_t01_val_t05_Split_20230830-1415/val/PNGImages"
data_path = Path(data_path)

# save path
out = "results"

# init coco format ouput
dataset = init_coco(model)

# by default no crowd
is_crowd = 0
# These ids will be automatically increased as we go
annotation_id = 0
image_id = 0
category_id = 0

for i, img_file in enumerate(data_path.glob("*.png")):
    print(f"Processing {img_file}...")
    img = mmcv.imread(img_file)
    width, height, _ = img.shape
    dataset["images"].append(
        {
            "license": 0,
            "file_name": img_file.name,
            "id": i,
            "width": width,
            "height": height,
        }
    )

    # get model predictions
    pred = inference_detector(model, img)
    # convert to numpy for post-processing
    masks = pred.pred_instances.masks.cpu().numpy()

    # get volume
    vols = pred.pred_instances.vols.cpu().numpy()

    for mask, vol in zip(masks, vols):
        annotation = create_sub_mask_annotation(
            mask, image_id, category_id, annotation_id, is_crowd, vol
        )
        annotation_id += 1
        if annotation == "skip":
            continue
        dataset["annotations"].append(annotation)
    image_id += 1

# save coco output json
with open(f"{out}/{data_path.stem}-coco-annotations.json", "w") as f:
    json.dump(dataset, f)

print("Done!")
