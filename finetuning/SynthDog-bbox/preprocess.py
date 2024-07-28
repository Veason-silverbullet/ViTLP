import os
import pickle
import json
from PIL import Image, ImageDraw
import shutil
DATA_DIR = 'synthetic_images'
PREPROCESSED_DATA_DIR = 'preprocessed_data'
IMAGE_DIR = os.path.join(PREPROCESSED_DATA_DIR, 'images')
ANNOTATION_DIR = os.path.join(PREPROCESSED_DATA_DIR, 'data')
VISUALIZATION_DIR = os.path.join(PREPROCESSED_DATA_DIR, 'visualization')
VISUALIZATION_FLAG = True
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
if not os.path.exists(ANNOTATION_DIR):
    os.makedirs(ANNOTATION_DIR)
if not os.path.exists(VISUALIZATION_DIR) and VISUALIZATION_FLAG:
    os.makedirs(VISUALIZATION_DIR)


def process(metadata, width, height):
    data = []
    for (region_data, (region_x1, region_y1, region_x2, region_y2)) in metadata:
        # print('Region-level bbox: [%.2f, %.2f, %.2f, %.2f]' % (region_x1, region_y1, region_x2, region_y2))
        for (line_data, (line_x1, line_x2, line_y1, line_y2)) in region_data:
            # print('Line-level bbox: [%.2f, %.2f, %.2f, %.2f]' % (line_x1, line_x2, line_y1, line_y2))
            for (word, (word_x1, word_y1, word_x2, word_y2)) in line_data:
                # print('Word: %s\tBbox: [%.2f, %.2f, %.2f, %.2f]' % (word, word_x1, word_x2, word_y1, word_y2))
                data.append({
                    'word': word.strip(),
                    'bbox': [
                        max(min(int(word_x1 * 1000 / width), 1000), 0),
                        max(min(int(word_y1 * 1000 / height), 1000), 0),
                        max(min(int(word_x2 * 1000 / width), 1000), 0),
                        max(min(int(word_y2 * 1000 / height), 1000), 0)
                    ]
                })
    return data


if __name__ == '__main__':
    for image in filter(lambda x: x.startswith('image_') and (x.endswith('.jpg') or x.endswith('.png')), os.listdir(DATA_DIR)):
        metadata_file = os.path.join('ocr_' + image[6: -4] + '.pkl')
        with open(os.path.join(DATA_DIR, metadata_file), 'rb') as f:
            metadata = pickle.load(f)
        with Image.open(os.path.join(DATA_DIR, image)) as img:
            # 1. Get image
            shutil.copy(os.path.join(DATA_DIR, image), os.path.join(IMAGE_DIR, image))
            width, height = img.size

            # 2. Preprocess data into [[word, [x1, y1, x2, y2]]...]
            data = process(metadata, width, height)
            annotation_file = os.path.join(ANNOTATION_DIR, 'ocr_' + image[6:-4] + '.json')
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            # 3. Visualization (optional)
            if VISUALIZATION_FLAG:
                img = img.convert('RGB')
                draw = ImageDraw.Draw(img)
                for item in data:
                    word = item['word']
                    x1, y1, x2, y2 = item['bbox']
                    x1 = x1 / 1000 * width
                    y1 = y1 / 1000 * height
                    x2 = x2 / 1000 * width
                    y2 = y2 / 1000 * height
                    draw.polygon([x1, y1, x2, y1, x2, y2, x1, y2], outline=(255, 0, 0))
                    try:
                        draw.text(((x1 + x2) / 2 - len(word) * 2.4, y1) , word, fill=(0, 0, 255, 128))
                    except:
                        pass
                img.save(os.path.join(VISUALIZATION_DIR, image))
