def is_mouth_open(landmarks, img_h):
    # lip top (point 13), lip bottom (point 14)
    upper = landmarks[13]
    lower = landmarks[14]
    mouth_dist = (lower.y - upper.y) * img_h
    return mouth_dist > 15  # tweak as per face distance
