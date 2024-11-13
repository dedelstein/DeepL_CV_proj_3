import cv2
import json
import os
import torch
import pandas as pd
import numpy as np
import torchvision.ops as ops
from torchvision import tv_tensors
from xml.etree import ElementTree as ET
from tqdm import tqdm

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
    abo = torch.max(ops.box_iou(boxes, ground_truths), dim=0).values.sum().item() / len(ground_truths)
    return abo

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

def save_results(num_rects, mabo_scores, output_file='results_final.csv'):
    row_dict = {
        'num_rects': num_rects,
        'mabo_edge': mabo_scores[0],
        'mabo_ss_fast': mabo_scores[1],
        'mabo_ss_qual': mabo_scores[2]
    }
      
    df = pd.DataFrame([row_dict])
    
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode='a', header=False, index=False)

def main():
    # Configuration
    path = 'Potholes/annotated-images/'
    splits = 'Potholes/splits.json'
    model = 'model.yml'
    device = "cpu"
    output_file = 'results_final.csv'
    
    # Optimization parameters
    subsample = 50  # Number of images to sample each iteration
    rect_step = 20   # Steps between number of rectangles
    max_rects = 2000  # Maximum number of rectangles
    
    cv2.setUseOptimized(True)
    cv2.setNumThreads(6)

    # Load full dataset list
    train_mask_list = [path + f for f in json.load(open(splits))['train']]

    # Process with increasing number of rectangles
    num_rects = 10  # Start with more rectangles
    with tqdm(total=(max_rects-num_rects)//rect_step) as pbar:
        while num_rects <= max_rects:
            # Random sampling for this iteration
            sampled_masks = train_mask_list[:subsample]
            train_img_list = [filename.replace('xml', 'jpg') for filename in sampled_masks]

            # Build dataset for this iteration
            dataset = []
            for idx, mask in enumerate(sampled_masks):
                ground_truths, size = read_xml(mask)
                ground_truths = tv_tensors.BoundingBoxes(ground_truths, format="XYXY", canvas_size=size)
                image = train_img_list[idx]
                dataset.append([image, ground_truths])

            edge_abo = 0
            ss_fast_abo = 0
            ss_qual_abo = 0
            
            for datum in dataset:
                image = datum[0]
                ground_truths = datum[1].to(device)
                size = ground_truths.canvas_size

                try:
                    # Get proposals from each method
                    edgeboxes, _ = edge_boxes(image, num_rects=num_rects, model=model)
                    ss_fast = selective_search(image, num_rects=num_rects, quality=False)
                    ss_qual = selective_search(image, num_rects=num_rects, quality=True)

                    # Convert to correct format
                    edgeboxes = ops.box_convert(torch.tensor(edgeboxes), "xywh", "xyxy")
                    edgeboxes = tv_tensors.BoundingBoxes(edgeboxes, format="XYXY", canvas_size=size).to(device)
                    ss_fast = ops.box_convert(torch.tensor(ss_fast), "xywh", "xyxy")
                    ss_fast = tv_tensors.BoundingBoxes(ss_fast, format="XYXY", canvas_size=size).to(device)
                    ss_qual = ops.box_convert(torch.tensor(ss_qual), "xywh", "xyxy")
                    ss_qual = tv_tensors.BoundingBoxes(ss_qual, format="XYXY", canvas_size=size).to(device)

                    edge_abo += abo(edgeboxes, ground_truths)
                    ss_fast_abo += abo(ss_fast, ground_truths)
                    ss_qual_abo += abo(ss_qual, ground_truths)

                except Exception as e:
                    print(f"Error processing image {image}: {str(e)}")
                    continue

            mabo_scores = [
                edge_abo / len(dataset),
                ss_fast_abo / len(dataset),
                ss_qual_abo / len(dataset)
            ]

            save_results(num_rects, mabo_scores, output_file)
            
            num_rects += rect_step
            pbar.update(1)
"""
# Runtime Analysis
def main():
    import cv2
    import time
    import matplotlib.pyplot as plt
    import numpy as np

    def measure_runtime(image_path, num_trials=5):
        # Load model once for edge boxes
        model = 'model.yml'

        # Lists to store timing results
        edge_times = []
        ss_fast_times = []
        ss_qual_times = []

        for _ in range(num_trials):
            # Time Edge Boxes
            start = time.time()
            edge_boxes(image_path, num_rects=750, model=model)
            edge_times.append((time.time() - start) * 1000)  # Convert to milliseconds

            # Time Selective Search Fast
            start = time.time()
            selective_search(image_path, num_rects=750, quality=False)
            ss_fast_times.append((time.time() - start) * 1000)

            # Time Selective Search Quality
            start = time.time()
            selective_search(image_path, num_rects=750, quality=True)
            ss_qual_times.append((time.time() - start) * 1000)

        # Calculate averages
        avg_times = [
            np.mean(edge_times),
            np.mean(ss_fast_times),
            np.mean(ss_qual_times)
        ]

        # Create bar plot
        methods = ['Edge Boxes', 'SS Fast', 'SS Quality']
        plt.figure(figsize=(8, 5))
        plt.bar(methods, avg_times)

        # Customize plot
        plt.title('Object Detection Methods: Runtime Comparison')
        plt.ylabel('Runtime (milliseconds)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of each bar
        for i, v in enumerate(avg_times):
            plt.text(i, v + 30, f'{v:.0f}ms', ha='center')

        plt.tight_layout()
        plt.savefig('runtime_chart.png', bbox_inches='tight', dpi=300)
        plt.close()

        return avg_times

    # Usage:
    # Choose a representative image from your dataset
    image_path = 'Potholes/annotated-images/img-13.jpg'  # Replace with actual image path
    runtimes = measure_runtime(image_path)
    print(f"Average runtimes: {runtimes}")
"""
if __name__ == "__main__":
    main()