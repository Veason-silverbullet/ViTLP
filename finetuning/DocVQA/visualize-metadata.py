# This visualization is for metadata debug, making sure it is as in desired format.
import os
import json
import random
from PIL import Image, ImageDraw
if os.path.exists('visualization'):
    os.system('rm -r visualization')
if not os.path.exists('visualization'):
    os.mkdir('visualization')
BIN_SIZE = 1000
NUM = 32


if __name__ == '__main__':
    with open('train-metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    random.shuffle(metadata)
    cnt = 0
    for item in metadata:
        if 'bboxes' not in item:
            continue
        img = item['image']
        image = Image.open(img).convert('RGB')
        width, height = image.size
        question = item['question']
        draw = ImageDraw.Draw(image)
        draw.text(((width / 2 - 256), 64), question, fill=(255, 0, 0, 2048))
        for bbox, word in item['bboxes']:
            w1, h1, w2, h2 = bbox
            w1 = w1 * width / BIN_SIZE
            w2 = w2 * width / BIN_SIZE
            h1 = h1 * height / BIN_SIZE
            h2 = h2 * height / BIN_SIZE
            draw.text(((w1 + w2) / 2 - len(word) * 2, h1), word, fill=(0, 0, 255, 2048))
            draw.polygon([w1, h1, w2, h1, w2, h2, w1, h2], outline=(255, 0, 0))
        image.save(os.path.join('visualization', os.path.basename(img)))
        cnt += 1
        if cnt == NUM:
            break
