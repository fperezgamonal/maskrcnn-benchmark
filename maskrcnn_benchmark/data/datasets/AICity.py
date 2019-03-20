# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

from maskrcnn_benchmark.data.datasets.AICity_detection import AICityDetection


class AICityDataset(AICityDetection):  # torch.utils.data.Dataset
    CLASSES = (
        "__background__ ",
        "car"
    )

    def __init__(self, ann_file, root, remove_images_without_annotations, transforms=None):
        super(AICityDetection, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            new_ids = []
            for img_id in self.ids:
                anns = self.AICity.loadAnns(self.AICity.getAnnIds(imgIds=img_id, iscrowd=None))
                is_filled = True
                for ann in anns:
                    if not ann["bbox"]:  # or not ann["segmentation"]:
                        is_filled = False
                if is_filled and len(anns) > 0:
                    new_ids.append(img_id)
            self.ids = new_ids
            # self.ids = [
            #    img_id
            #    for img_id in self.ids
            #    if len(self.AICity.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            # ]

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.AICity.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.transforms = transforms

    def __getitem__(self, index):
        import sys
        sys.stdout.flush()
        print('getting item', index)
        img, anno = super(AICityDataset, self).__getitem__(index)
        # filter crowd annotations
        # TODO might be better to add an extra field
        # anno = [obj for obj in anno if obj["iscrowd"] == 0]
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        # In AICity we do not have segmentations, right?
        # masks = [obj["segmentation"] for obj in anno]
        # masks = SegmentationMask(masks, img.size)
        # target.add_field("masks", masks)

        # was remove_empty=True
        target = target.clip_to_image(remove_empty=False)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_img_info(self, index):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        img_id = self.id_to_img_map[index]
        img_data = self.aicity.imgs[img_id]

        return img_data
