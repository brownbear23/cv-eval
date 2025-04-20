def has_quality(filename, threshold=0.4):
    score_str = filename.rsplit("_", 1)[-1].split(".")[0]  # '5-3'
    score = float(score_str.replace("-", "."))  # convert '5-3' → '5.3' → float
    if score >= threshold:
        return True
    else:
        return False