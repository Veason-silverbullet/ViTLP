"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import pickle
import os
import re
from typing import List
import numpy as np
from elements import Background, Document
from PIL import Image
from synthtiger import components, layers, templates


class SynthDoG(templates.Template):
    def __init__(self, config=None, split_ratio: List[float] = [0.8, 0.1, 0.1]):
        super().__init__(config)
        if config is None:
            config = {}
        self.quality = config.get('quality', [50, 95])
        self.landscape = config.get('landscape', 0.5)
        self.short_size = config.get('short_size', [720, 1024])
        self.aspect_ratio = config.get('aspect_ratio', [1, 2])
        self.background = Background(config.get('background', {}))
        self.document = Document(config.get('document', {}))
        self.effect = components.Iterator(
            [
                components.Switch(components.RGB()),
                components.Switch(components.Shadow()),
                components.Switch(components.Contrast()),
                components.Switch(components.Brightness()),
                components.Switch(components.MotionBlur()),
                components.Switch(components.GaussianBlur())
            ],
            **config.get('effect', {})
        )

    def generate(self):
        landscape = np.random.rand() < self.landscape
        short_size = np.random.randint(self.short_size[0], self.short_size[1] + 1)
        aspect_ratio = np.random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
        long_size = int(short_size * aspect_ratio)
        size = (long_size, short_size) if landscape else (short_size, long_size)
        bg_layer = self.background.generate(size)
        paper_layer, text_layers, texts, ocr_data = self.document.generate(size)
        document_group = layers.Group([*text_layers, paper_layer])
        document_space = np.clip(size - document_group.size, 0, None)
        document_group.left = np.random.randint(document_space[0] + 1)
        document_group.top = np.random.randint(document_space[1] + 1)
        roi = np.array(paper_layer.quad, dtype=int)
        layer = layers.Group([*document_group.layers, bg_layer]).merge()
        self.effect.apply([layer])
        image = layer.output(bbox=[0, 0, *size])
        label = ' '.join(texts)
        label = label.strip()
        label = re.sub(r'\s+', ' ', label)
        quality = np.random.randint(self.quality[0], self.quality[1] + 1)

        data = {
            'image': image,
            'label': label,
            'quality': quality,
            'roi': roi,
        }
        for i in range(len(ocr_data)):
            for j in range(len(ocr_data[i][0])):
                for k in range(len(ocr_data[i][0][j][0])):
                    ocr_data[i][0][j][0][k][1][0] += document_group.left
                    ocr_data[i][0][j][0][k][1][1] += document_group.top
                    ocr_data[i][0][j][0][k][1][2] += document_group.left
                    ocr_data[i][0][j][0][k][1][3] += document_group.top
                ocr_data[i][0][j][1][0] += document_group.left
                ocr_data[i][0][j][1][1] += document_group.top
                ocr_data[i][0][j][1][2] += document_group.left
                ocr_data[i][0][j][1][3] += document_group.top
            ocr_data[i][1][0] += document_group.left
            ocr_data[i][1][1] += document_group.top
            ocr_data[i][1][2] += document_group.left
            ocr_data[i][1][3] += document_group.top
        return data, ocr_data

    def init_save(self, root):
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)

    def save(self, root, data_, idx):
        data, ocr_data = data_
        image = data['image']
        quality = data['quality']

        # save image
        image_filename = f'image_{idx}.jpg'
        image_filepath = os.path.join(root, image_filename)
        os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
        image = Image.fromarray(image[..., :3].astype(np.uint8))
        image.save(image_filepath, quality=quality)

        # save metadata
        metadata_filename = f'ocr_{idx}.pkl'
        metadata_filepath = os.path.join(root, metadata_filename)
        with open(metadata_filepath, 'wb') as f:
            pickle.dump(ocr_data, f)

    def end_save(self, root):
        pass
