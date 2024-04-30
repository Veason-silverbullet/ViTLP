import os
import json
from PIL import Image, ImageDraw, ImageFont
ttf = ImageFont.truetype('misc/Arial.ttf', 9)


def visualize(file, visualization_dir):
    image_file = os.path.join(visualization_dir, file + '.jpg')
    if not os.path.exists(image_file):
        image_file = os.path.join(visualization_dir, file + '.png')
    if not os.path.exists(image_file):
        image_file = os.path.join(visualization_dir, file + '.tif')
    with open(os.path.join(visualization_dir, file + '.json'), 'r', encoding='utf-8') as f:
        results = json.load(f)
    image = Image.open(image_file).convert('RGB')
    width, height = image.size
    for bbox, word in results:
        draw = ImageDraw.Draw(image)
        w1, h1, w2, h2 = bbox
        w1 = w1 / 1000 * width
        w2 = w2 / 1000 * width
        h1 = h1 / 1000 * height
        h2 = h2 / 1000 * height
        draw.polygon([w1, h1, w2, h1, w2, h2, w1, h2], outline=(255, 0, 0))
        try:
            draw.text(((w1 + w2) / 2 - len(word) * 2.4, h1 - 3) , word.strip(), fill=(0, 0, 255, 128), font=ttf)
        except:
            pass
    image.save(os.path.join(visualization_dir, file + '-visualization.png'))


def visualize_box(file, visualization_dir):
    image_file = os.path.join(visualization_dir, file + '.png')
    result_file = os.path.join(visualization_dir, file + '.json')
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    image = Image.open(image_file).convert('RGB')
    width, height = image.size
    draw = ImageDraw.Draw(image)
    for bbox, word in results:
        w1, h1, w2, h2 = bbox
        w1 = w1 / 1000 * width
        w2 = w2 / 1000 * width
        h1 = h1 / 1000 * height
        h2 = h2 / 1000 * height
        draw.polygon([w1, h1, w2, h1, w2, h2, w1, h2], outline=(255, 0, 0))
    image.save(os.path.join(visualization_dir, file + '-visualization-box.png'))


if __name__ == '__main__':
    for visualization_dir in filter(lambda x: x.startswith('ViTLP'), os.listdir('decode_output')):
        if not os.path.exists(os.path.join('decode_output', visualization_dir)):
            os.mkdir(os.path.join('decode_output', visualization_dir))
        visualization_dir_ = os.path.join('decode_output', visualization_dir)
        for result_file in filter(lambda x: x.endswith('.json'), os.listdir(visualization_dir_)):
            visualize(result_file[:-5], visualization_dir_)
