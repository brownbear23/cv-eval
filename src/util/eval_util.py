def has_quality(filename, threshold=0.4):
    score_str = filename.rsplit("_", 1)[-1].split(".")[0]  # '5-3'
    score = float(score_str.replace("-", "."))  # convert '5-3' → '5.3' → float
    if score >= threshold:
        return True
    else:
        return False


COCO_KEYWORDS = {"couch": [57, 56],  # couch: chair
                 "chair": [56, 57],  # chair: couch
                 "potted": [58, 75, 45],  # potted plant: vase, bowl
                 "dining": [60],  # dining table:
                 "backpack": [24, 26, 28], # backpack: handbag, suitcase
                 "door": [],
                 "ladder": [],
                 "rolled": [],
                 "shoes": [],
                 "trash": []
                 }


def get_coco_lst(target_obj):
    target_obj_lower = target_obj.lower()
    for keyword, class_idx_lst in COCO_KEYWORDS.items():
        if keyword in target_obj_lower:
            return class_idx_lst
    return None
