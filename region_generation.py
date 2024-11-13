import cv2
import json
import pickle
import torch
import torchvision.ops as ops
from torchvision import tv_tensors
from xml.etree import ElementTree as ET
from tqdm import tqdm

def selective_search(image_path, num_rects):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    image = cv2.imread(image_path)
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    return rects[:num_rects]

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

def save_results(results, output_file):
    # Function to save results as a pickle file or any desired format
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

def main():
    # Configuration
    path = 'Potholes/annotated-images/'
    splits = 'Potholes/splits.json'
    num_rects = 750
    output_file = 'proposals.pkl'
    
    cv2.setUseOptimized(True)
    cv2.setNumThreads(8)

    # Load full dataset list
    train_mask_list = [path + f for f in json.load(open(splits))['train']]
    train_img_list = [filename.replace('xml', 'jpg') for filename in train_mask_list]
    val_mask_list = [path + f for f in json.load(open(splits))['test']]
    val_img_list = [filename.replace('xml', 'jpg') for filename in val_mask_list]

    mask_list = train_mask_list + val_mask_list
    img_list = train_img_list + val_img_list

    dataset = []
    for idx, mask in enumerate(mask_list):
        ground_truths, size = read_xml(mask)
        ground_truths = tv_tensors.BoundingBoxes(ground_truths, format="XYXY", canvas_size=size)
        image = img_list[idx]
        dataset.append([image, ground_truths])
    
    results = []
    for datum in tqdm(dataset, desc="Processing images"):
        image = datum[0]
        ground_truths = datum[1]
        size = ground_truths.canvas_size

        try:
            # Get proposals
            ss_fast = selective_search(image, num_rects=num_rects)
            ss_fast = ops.box_convert(torch.tensor(ss_fast), "xywh", "xyxy")
            ss_fast = tv_tensors.BoundingBoxes(ss_fast, format="XYXY", canvas_size=size)

        except Exception as e:
            print(f"Error processing image {image}: {str(e)}")
            continue
        
        results.append([image, ground_truths, ss_fast])

    save_results(results, output_file)

if __name__ == "__main__":
    main()