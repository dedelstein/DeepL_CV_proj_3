import cv2
import json
import os
import torch
import pickle
import pandas as pd
import numpy as np
import torchvision.ops as ops
from datetime import datetime
from torchvision import tv_tensors
from xml.etree import ElementTree as ET

def edge_boxes(image_path, num_rects, model):

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)

    image = cv2.imread(image_path)
    rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(num_rects)
    boxes, scores = edge_boxes.getBoundingBoxes(edges, orimap)

    return boxes, scores

def selective_search(image_path, num_rects, quality=True):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    image = cv2.imread(image_path)
    ss.setBaseImage(image)
    
    if quality:
        ss.switchToSelectiveSearchQuality()
    else:
        ss.switchToSelectiveSearchFast()
        
    rects = ss.process()

    return rects[:num_rects]

def abo(boxes, ground_truths):

    abo = torch.max(ops.box_iou(boxes, ground_truths), dim=0).values.sum().item() / ground_truths.numel()

    return abo

def recall(boxes, ground_truths, k = 0.7, coord = None):

    ious = ops.box_iou(boxes, ground_truths)
    recall = torch.gt(ious, k).sum().item() / boxes.shape[0]

    return recall

def read_xml(path: str):  

    tree = ET.parse(path)
    root = tree.getroot()

    obj_list = []

    size = (int(root.find("size/width").text), int(root.find("size/height").text))

    for obj in root.iter('object'):

        ymin = int(obj.find("bndbox/ymin").text)
        xmin = int(obj.find("bndbox/xmin").text)
        ymax = int(obj.find("bndbox/ymax").text)
        xmax = int(obj.find("bndbox/xmax").text)

        bbox = (xmin, ymin, xmax, ymax)
        obj_list.append(bbox)
    
    return obj_list, size

path = 'Potholes/annotated-images/'
splits = 'Potholes/splits.json'
model = 'model.yml'

device = "cuda"

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

train_mask_list = [path + f for f in json.load(open(splits))['train']]
train_img_list = [filename.replace('xml', 'jpg') for filename in train_mask_list]

dataset = []
for idx, mask in enumerate(train_mask_list):
    ground_truths, size = read_xml(mask)
    ground_truths = tv_tensors.BoundingBoxes(ground_truths, format="XYXY", canvas_size=size)
    image = train_img_list[idx]
    dataset.append([image, ground_truths])

num_rects = 10
mabo_edge = []
mabo_ss_fast = []
mabo_ss_qual = []
total_recalls = []
print("here")

while num_rects < 2500:
    edge_abo = 0
    ss_fast_abo = 0
    ss_qual_abo = 0
    recalls = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) # [edge, ss_fast, ss_qual], .5, .7, .9
   
    for datum in dataset:
        image = datum[0]
        ground_truths = datum[1].to(device)
        size = ground_truths.canvas_size

        edgeboxes, _ = edge_boxes(image, num_rects=num_rects, model=model)
        ss_fast = selective_search(image, num_rects=num_rects, quality=False)
        ss_qual = selective_search(image, num_rects=num_rects, quality=True)

        edgeboxes = ops.box_convert(torch.tensor(edgeboxes), "xywh", "xyxy")
        edgeboxes = tv_tensors.BoundingBoxes(edgeboxes, format="XYXY", canvas_size=size).to(device)
        ss_fast = ops.box_convert(torch.tensor(ss_fast), "xywh", "xyxy")
        ss_fast = tv_tensors.BoundingBoxes(ss_fast, format="XYXY", canvas_size=size).to(device)
        ss_qual = ops.box_convert(torch.tensor(ss_qual), "xywh", "xyxy")
        ss_qual = tv_tensors.BoundingBoxes(ss_qual, format="XYXY", canvas_size=size).to(device)

        edge_abo += abo(edgeboxes, ground_truths)
        ss_fast_abo += abo(ss_fast, ground_truths)
        ss_qual_abo += abo(ss_qual, ground_truths)

        for idx, k in enumerate([.5, .7, .9]):
            recall_edge = recall(edgeboxes, ground_truths, k=k)
            recall_ss_fast = recall(ss_fast, ground_truths, k=k)
            recall_ss_qual = recall(ss_qual, ground_truths, k=k)
            recalls[idx] += np.array([recall_edge, recall_ss_fast, recall_ss_qual])

    recalls = recalls / len(dataset)
    total_recalls.append(recalls)
    mabo_edge.append(edge_abo / len(dataset))
    mabo_ss_fast.append(ss_fast_abo / len(dataset))
    mabo_ss_qual.append(ss_qual_abo / len(dataset))

    num_rects += 20

#timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

results_dict = {
        'mabo_edge': mabo_edge,
        'mabo_ss_fast': mabo_ss_fast,
        'mabo_ss_qual': mabo_ss_qual,
        'total_recalls': total_recalls
    }

#pickle_path = os.path.join('results', f'results_{timestamp}.pkl')
#with open(pickle_path, 'wb') as f:
#    pickle.dump(results_dict, f)

df = pd.DataFrame(results_dict)
df.to_csv('results.csv', index=False)