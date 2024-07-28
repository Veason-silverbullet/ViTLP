"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import os
from collections import OrderedDict

import numpy as np
from synthtiger import components

from elements.textbox import TextBox
from layouts import GridStack
MAX_GEN_NUM = 3
MIN_SHORT_SIZE = 320


class TextReader:
    def __init__(self, path, cache_size=2 ** 28, block_size=2 ** 20):
        self.fp = open(path, "r", encoding="utf-8")
        self.length = 0
        self.offsets = [0]
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.block_size = block_size
        self.bucket_size = cache_size // block_size
        self.idx = 0

        while True:
            text = self.fp.read(self.block_size)
            if not text:
                break
            self.length += len(text)
            self.offsets.append(self.fp.tell())

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        char = self.get()
        self.next()
        return char

    def move(self, idx):
        self.idx = idx

    def next(self):
        self.idx = (self.idx + 1) % self.length

    def prev(self):
        self.idx = (self.idx - 1) % self.length

    def get(self):
        key = self.idx // self.block_size

        if key in self.cache:
            text = self.cache[key]
        else:
            if len(self.cache) >= self.bucket_size:
                self.cache.popitem(last=False)

            offset = self.offsets[key]
            self.fp.seek(offset, 0)
            text = self.fp.read(self.block_size)
            self.cache[key] = text

        self.cache.move_to_end(key)
        char = text[self.idx % self.block_size]
        return char


class Content:
    def __init__(self, config):
        self.margin = config.get("margin", [0, 0.1])
        self.reader = TextReader(**config.get("text", {}))
        self.font = components.BaseFont(**config.get("font", {}))
        self.layout = GridStack(config.get("layout", {}))
        self.textbox = TextBox(config.get("textbox", {}))
        self.textbox_color = components.Switch(components.Gray(), **config.get("textbox_color", {}))
        self.content_color = components.Switch(components.Gray(), **config.get("content_color", {}))
        self.last_idx = None

    def generate(self, size):
        if self.last_idx is not None and self.last_idx > self.reader.idx:
            raise Exception('Reader indexing error')
        self.last_idx = self.reader.idx
        width, height = size

        layout_left = width * np.random.uniform(self.margin[0], self.margin[1])
        layout_top = height * np.random.uniform(self.margin[0], self.margin[1])
        layout_width = max(width - layout_left * 2, 0)
        layout_height = max(height - layout_top * 2, 0)
        layout_bbox = [layout_left, layout_top, layout_width, layout_height]

        text_layers, texts = [], []
        while True:
            layouts = self.layout.generate(layout_bbox)
            flag = True
            for layout in layouts:
                for bbox, align in layout:
                    x, y, w, h = bbox
                    if w <= MIN_SHORT_SIZE:
                        flag = False
                        break
            if flag:
                break
        ocr_data = []

        region_flag = True
        for layout in layouts:
            font = self.font.sample()
            if region_flag:
                ocr_data.append([[], None])
                region_x1, region_y1, region_x2, region_y2 = None, None, None, None
            assert len(layout) > 0

            generate_cnt = 0
            while True:
                previous_reader_idx = self.reader.idx
                previous_text_layers_size = len(text_layers)
                previous_texts_size = len(texts)
                previous_ocr_data_size = len(ocr_data[-1][0])
                previous_region_x1, previous_region_y1, previous_region_x2, previous_region_y2 = region_x1, region_y1, region_x2, region_y2
                layout_flag = True
                for bbox, align in layout:
                    x, y, w, h = bbox
                    text_layer, text, offset, info, textbox_flag = self.textbox.generate((w, h), self.reader, font, generate_cnt)
                    if not textbox_flag:
                        layout_flag = False
                        if generate_cnt != MAX_GEN_NUM:
                            break
                    for i in range(offset):
                        self.reader.prev()

                    if text_layer is None:
                        continue
                    x_offset = (w - info[-1][1] - info[-1][2]) / 2
                    assert x_offset >= 0, 'offset error'
                    cache_chars = ''
                    W = 0
                    ocr_data[-1][0].append([[], None])
                    line_x1, line_y1, line_x2, line_y2 = None, None, None, None
                    for char, char_x, char_w in info:
                        if char == ' ':
                            if cache_chars != '':
                                x1, y1, x2, y2 = X, y + int(h) // 8 + 1, W, max(h - int(h) // 8 - 1, 0)
                                ocr_data[-1][0][-1][0].append([cache_chars, [x1, y1, x1 + x2, y1 + y2]])
                                if line_x1 is None:
                                    line_x1 = x1
                                else:
                                    line_x1 = min(line_x1, x1)
                                if line_y1 is None:
                                    line_y1 = y1
                                else:
                                    line_y1 = min(line_y1, y1)
                                if line_x2 is None:
                                    line_x2 = x1 + x2
                                else:
                                    line_x2 = max(line_x2, x1 + x2)
                                if line_y2 is None:
                                    line_y2 = y1 + y2
                                else:
                                    line_y2 = max(line_y2, y1 + y2)
                                if region_x1 is None:
                                    region_x1 = x1
                                else:
                                    region_x1 = min(region_x1, x1)
                                if region_y1 is None:
                                    region_y1 = y1
                                else:
                                    region_y1 = min(region_y1, y1)
                                if region_x2 is None:
                                    region_x2 = x1 + x2
                                else:
                                    region_x2 = max(region_x2, x1 + x2)
                                if region_y2 is None:
                                    region_y2 = y1 + y2
                                else:
                                    region_y2 = max(region_y2, y1 + y2)
                                cache_chars = ''
                                W = 0
                        else:
                            if cache_chars == '':
                                X = x + char_x + x_offset
                            W += char_w
                            cache_chars += char
                    if cache_chars != '':
                        x1, y1, x2, y2 = X, y + int(h) // 8 + 1, W, max(h - int(h) // 8 - 1, 0)
                        ocr_data[-1][0][-1][0].append([cache_chars, [x1, y1, x1 + x2, y1 + y2]])
                        if line_x1 is None:
                            line_x1 = x1
                        else:
                            line_x1 = min(line_x1, x1)
                        if line_y1 is None:
                            line_y1 = y1
                        else:
                            line_y1 = min(line_y1, y1)
                        if line_x2 is None:
                            line_x2 = x1 + x2
                        else:
                            line_x2 = max(line_x2, x1 + x2)
                        if line_y2 is None:
                            line_y2 = y1 + y2
                        else:
                            line_y2 = max(line_y2, y1 + y2)
                        if region_x1 is None:
                            region_x1 = x1
                        else:
                            region_x1 = min(region_x1, x1)
                        if region_y1 is None:
                            region_y1 = y1
                        else:
                            region_y1 = min(region_y1, y1)
                        if region_x2 is None:
                            region_x2 = x1 + x2
                        else:
                            region_x2 = max(region_x2, x1 + x2)
                        if region_y2 is None:
                            region_y2 = y1 + y2
                        else:
                            region_y2 = max(region_y2, y1 + y2)

                    text_layer.center = (x + w / 2, y + h / 2)

                    self.textbox_color.apply([text_layer])
                    text_layers.append(text_layer)
                    texts.append(text)

                    assert line_x1 is not None and line_y1 is not None and line_x2 is not None and line_y2 is not None
                    ocr_data[-1][0][-1][1] = [line_x1, line_y1, line_x2, line_y2]

                if layout_flag or generate_cnt == MAX_GEN_NUM:
                    if region_x1 is not None and region_y1 is not None and region_x2 is not None and region_y2 is not None:
                        ocr_data[-1][1] = [region_x1, region_y1, region_x2, region_y2]
                        region_flag = True
                    else:
                        region_flag = False
                    break
                generate_cnt += 1
                self.reader.move(previous_reader_idx)
                text_layers = text_layers[:previous_text_layers_size]
                texts = texts[:previous_texts_size]
                ocr_data[-1][0] = ocr_data[-1][0][:previous_ocr_data_size]
                region_x1, region_y1, region_x2, region_y2 = previous_region_x1, previous_region_y1, previous_region_x2, previous_region_y2
                font['size'] = font['size'] // 1.2

        if not region_flag:
            ocr_data = ocr_data[:-1]

        self.content_color.apply(text_layers)

        return text_layers, texts, ocr_data
