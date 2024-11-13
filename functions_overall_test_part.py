# Function to use on 1 image at a time, boxes and scores are tensors and iou_threshold is a float
def nms(boxes, scores, iou_threshold):  
    # Sort boxes by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    keep_indices = []

    while len(sorted_indices) > 0:
        # Get the index of the box with the highest score
        box_index = sorted_indices[0]
        keep_indices.append(box_index)

        # Calculate IoU between the current box and all remaining boxes
        ious = ops.box_iou(boxes[box_index].unsqueeze(0), boxes[sorted_indices[1:]])

        # Remove boxes with IoU above the threshold
        sorted_indices = sorted_indices[1:][ious < iou_threshold]

    keep_boxes = boxes[keep_indices]
    keep_scores = scores[keep_indices]

    # Return the kept boxes and scores associated
    return keep_boxes, keep_scores


# Function to run on the entire test set, with boxes, boxes_img, true_boxes and true_boxes_img tensors 
# where a box is at the same position than its image (where it comes from) in the corresponding img tensor. 
# Score is also a tensor and iou_threshold is a float.
# Output is : AP, a list of the precision evolution and a list of the recall evolution (to eventually plot them)
def ap(boxes, boxes_img, true_boxes, true_boxes_img, scores, iou_threshold=0.5):
    # Sort boxes by scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    number_boxes=len(true_boxes)
    true_positives = 0
    false_positives = 0
    precision=[]
    recall=[]
    sum_precision=0

    for i, box_index in enumerate(sorted_indices):
        box = boxes[box_index]
        img = boxes_img[box_index]
        img_true_boxes = true_boxes[true_boxes_img == img]
        iou = ops.box_iou(box.unsqueeze(0), img_true_boxes)
        max_iou, max_idx = iou.max(dim=1)
        if max_iou >= iou_threshold:
            true_positives += 1
            true_boxes=torch.cat((true_boxes[:max_idx], true_boxes[max_idx+1:]))
            sum_precision+=true_positives / (true_positives + false_positives)
        else:
            false_positives += 1

        precision.append(true_positives / (true_positives + false_positives))
        recall.append(true_positives / number_boxes)
        
    return sum_precision/number_boxes, precision, recall


# Function to discard all of the background boxes. 
# Boxes and scores are tensors, threshold is a float. 
# Score correspond to sigmoid(output_of_the_model).
def real_boxes(boxes, score, threshold=0.5):
    labels=score>threshold
    boxes=boxes[labels]
    score=score[labels]
    return boxes, score
