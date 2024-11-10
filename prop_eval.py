import numpy as np

def mabo(rect, ground_truth, coord = None):
    box = rect.copy()
    
    if coord == 'xywh':                 # If coordinates are given as (x, y, w, h) instead of (xmin, ymin, xmax, ymax)
        box[:, 2] += box[:, 0]
        box[:, 3] += box[:, 1]

    average = 0.0
    for i in range(0, np.shape(ground_truth)[0]):
        best_cand = []
        for j in range(0, np.shape(box)[0]):
            overlap = iou(ground_truth[i], box[j])
            best_cand.append(overlap)
        
        best = max(best_cand)
        average += best
    average /= np.shape(ground_truth)[0]

    return average

def detection_rate(rect, ground_truth, k = 0.8, coord = None):
    box = rect.copy()

    if coord == 'xywh':                 # If coordinates are given as (x, y, w, h) instead of (xmin, ymin, xmax, ymax)
        box[:, 2] += box[:, 0]
        box[:, 3] += box[:, 1]

    good_prop = 0
    for i in range(0, np.shape(ground_truth)[0]):
        for j in range(0, np.shape(box)[0]):
            iou_k = iou(ground_truth[i], box[j])
            if iou_k > k:
                good_prop += 1
                break

    per = good_prop / np.shape(ground_truth)[0]

    return per

def iou(boxA, boxB):
    inter_xmin = max(boxA[0], boxB[0])
    inter_ymin = max(boxA[1], boxB[1])
    inter_xmax = min(boxA[2], boxB[2])
    inter_ymax = min(boxA[3], boxB[3])

    if inter_xmin < inter_xmax and inter_ymin < inter_ymax:
        inter = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        union = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) + (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) - inter
        overlap = inter / union
    else:
        overlap = 0
    
    return overlap