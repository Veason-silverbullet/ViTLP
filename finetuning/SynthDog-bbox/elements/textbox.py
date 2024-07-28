"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import numpy as np
from synthtiger import layers


class TextBox:
    def __init__(self, config):
        self.fill = config.get("fill", [1, 1])

    def generate(self, size, text, font, generate_cnt=0):
        width, height = size

        char_layers, chars = [], []
        width = np.clip(width, height, width)
        if generate_cnt > 0:
            height /= 1.2 ** generate_cnt
        font = {**font, "size": int(height)}
        left, top = 0, 0
        pos = None

        for index, char in enumerate(text):
            if char in "\r\n":
                pos = index
                continue
            if char == ' ':
                pos = index

            char_layer = layers.TextLayer(char, **font)
            char_scale = height / char_layer.height
            char_layer.bbox = [left, top, *(char_layer.size * char_scale)]
            if char_layer.right > width:
                if pos is None:
                    offset = 1
                else:
                    offset = index - pos
                    if len(''.join(chars[:-offset]).strip()) == 0:
                        offset = 1
                    else:
                        char_layers = char_layers[:-offset]
                        chars = chars[:-offset]
                break

            char_layers.append(char_layer)
            chars.append(char)
            left = char_layer.right

        text = "".join(chars).strip()
        if len(char_layers) == 0 or len(text) == 0:
            return None, None, offset, None, (pos is not None)

        text_layer = layers.Group(char_layers).merge()
        info = [[chars[i], char_layers[i].bbox[0], char_layers[i].bbox[2]] for i in range(len(char_layers))]
        while len(info) > 0 and info[0][0] == ' ':
            info = info[1:]
        return text_layer, text, offset, info, (pos is not None)
