"""
reads on args the path for a coco dataset image folder and its correspondending annotation json file and converts it to the YOLO format:
- a folder with the images
- a txt file for each image with the annotations in the format: class x_center y_center width heigh
it will also create a yaml file with the classes names at the output folder with the following format:

yaml file example:
```
names:
- sedan
- minibus
- truck
- pickup
- bus
- cement_truck
- trailer
nc: 7
```
"""

import os
import sys
import json
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', filename='main.log', filemode='w')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert COCO dataset to YOLO format')
    parser.add_argument('--images_folder', type=str, help='path to the dataset folder', required=True)
    parser.add_argument('--annotations_path', type=str, help='path to the annotations json file', required=True)
    parser.add_argument('--output_path', type=str, help='path to the output folder (both images and annotations folder will be created inside it)', required=True)
    
    return parser.parse_args()



def main(args):
    
    images_folder = args.images_folder.replace('\\', '/')
    annotations_path = args.annotations_path.replace('\\', '/')
    output_path = args.output_path.replace('\\', '/')
    logging.info(f'Dataset path: {images_folder}')
    logging.info(f'Annotations path: {annotations_path}')
    logging.info(f'Output path: {output_path}')
    
    # create output folders
    images_output_path = os.path.join(output_path, 'images')
    annotations_output_path = os.path.join(output_path, 'labels')
    os.makedirs(images_output_path, exist_ok=True)
    os.makedirs(annotations_output_path, exist_ok=True)
    
    # read annotations
    logging.info('Opening annotations')
    with open(annotations_path, 'r') as f:
        logging.info('Reading annotations')
        annotations = json.load(f)
    logging.info('Annotations read successfully')
    
    # get all categories
    logging.info('Getting all unique categories')
    categories = {}
    for category in tqdm(annotations['categories']):
        categories[category['id']] = category['name']
    
    # write classes to yaml file
    classes = []
    for category in categories.values():
        classes.append(category)
    logging.info('Writing classes to yaml file')
    with open(os.path.join(output_path, 'classes.yaml'), 'w') as f:
        f.write('names:\n')
        for class_name in classes:
            f.write(f'- {class_name}\n')
        f.write(f'nc: {len(classes)}\n')
    logging.info('Classes written to yaml file')
    
    # write annotations to txt files
    logging.info('Initializing writing annotations to txt files')
    logging.info(f'Unique keys of an annotation: {annotations["annotations"][0].keys()}')
    for annotation in tqdm(annotations['annotations']):
        image_id = annotation['image_id']
        image_name = f'{image_id}.jpg'
        image_path = os.path.join(images_folder, image_name)
        logging.info(f'Processing image {image_name}')
                
        # write annotation (it can happen to have multiple annotations for the same image, so it will append to the file)
        annotation_path = os.path.join(annotations_output_path, f'{image_id}.txt')
        with open(annotation_path, 'a') as f:
            category_id = annotation['category_id']
            category = categories[category_id]
            bbox = annotation['bbox']
            x_center = bbox[0] + bbox[2] / 2
            y_center = bbox[1] + bbox[3] / 2
            width = bbox[2]
            height = bbox[3]
            f.write(f'{category} {x_center} {y_center} {width} {height}\n')
            
    logging.info('Finished writing annotations to txt files')
    
    # Copy images to output folder
    logging.info('Copying images to output folder')
    # check os
    if sys.platform == 'win32':
        os.system(f'copy {images_folder}/*.jpg {images_output_path}')
    else:
        os.system(f'cp {images_folder}/*.jpg {images_output_path}')
        
    logging.info('Images copied to output folder')
    
    logging.info('Finished')
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
    