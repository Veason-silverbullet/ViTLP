import os
import json
import numpy as np
import shutil
from transformers import BartTokenizer
BIN_SIZE = 1000
DECODER_START_TOKEN_ID = 2
LOCATE_TOKEN_ID = 50265
CONTINUE_DECODE_ID = 50266
PREFIX_RATIO = 0.25
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MAX_LENGTH = 1280
PAD_TOKEN_TYPE = 0
WORD_TOKEN_TYPE = 1
LOCATE_TOKEN_TYPE = 2
tokenizer = BartTokenizer.from_pretrained('../configs/ViTLP-1920-1600')


def process_text_bbox_data(metadata_dir, image_dir, preprocessed_data_dir):
    if not os.path.exists(preprocessed_data_dir):
        os.mkdir(preprocessed_data_dir)
    preprocessed_image_dir = os.path.join(preprocessed_data_dir, 'images')
    if not os.path.exists(preprocessed_image_dir):
        os.mkdir(preprocessed_image_dir)
    images = os.listdir(image_dir)
    N = len(images)
    index = 0
    tokens = np.full([N * 2, MAX_LENGTH + 1], PAD_TOKEN_ID, dtype=np.int32)
    locations = np.full([N * 2, MAX_LENGTH + 1, 4], BIN_SIZE + 1, dtype=np.int16)
    token_types = np.zeros([N * 2, MAX_LENGTH + 1], dtype=np.int8)
    segments = np.zeros([N * 2, MAX_LENGTH + 1], dtype=bool) # 0 for input ids, 1 for prefix
    tokens[:, 0] = DECODER_START_TOKEN_ID
    with open(os.path.join(preprocessed_data_dir, 'mapping.txt'), 'w', encoding='utf-8') as f:
        for image in images:
            assert image.startswith('image_') and (image.endswith('.jpg') or image.endswith('.png'))
            with open(os.path.join(metadata_dir, 'ocr_' + image[6: -4] + '.json'), 'r', encoding='utf-8') as f_:
                metadata = json.load(f_)
            bboxes_ = [[]]
            texts_ = [[]]
            prefix_bboxes_ = []
            prefix_texts_ = []
            budget = MAX_LENGTH - 1
            for idx, item in enumerate(metadata):
                if idx == 0:
                    bbox, ids = item['bbox'], tokenizer.encode(item['word'], add_special_tokens=False)
                else:
                    bbox, ids = item['bbox'], tokenizer.encode(' ' + item['word'], add_special_tokens=False)
                N = len(ids) + 1
                if N > budget:
                    M = int(len(bboxes_[-1]) * PREFIX_RATIO)
                    assert M > 0, image
                    prefix_bboxes_.append(bboxes_[-1][-M:])
                    prefix_texts_.append(texts_[-1][-M:])
                    bboxes_.append([])
                    texts_.append([])
                    budget = MAX_LENGTH - 1
                    for prefix_ids in prefix_texts_[-1]:
                        budget -= len(prefix_ids) + 1
                    assert budget > 2
                bboxes_[-1].append(bbox)
                texts_[-1].append(ids)
                budget -= N
            for j, bboxes in enumerate(bboxes_):
                pos = 1
                if j > 0:
                    tokens[index][0] = CONTINUE_DECODE_ID
                    prefix_bboxes = prefix_bboxes_[j - 1]
                    prefix_texts = prefix_texts_[j - 1]
                    for i, bbox in enumerate(prefix_bboxes):
                        ids = prefix_texts[i]
                        N = len(ids)
                        assert pos + N + 1 <= MAX_LENGTH, 'Exceed max length'
                        for offset in range(N):
                            tokens[index][pos + offset] = ids[offset]
                            token_types[index][pos + offset] = WORD_TOKEN_TYPE
                        pos += N
                        tokens[index][pos] = LOCATE_TOKEN_ID
                        token_types[index][pos] = LOCATE_TOKEN_TYPE
                        locations[index][pos] = bbox
                        pos += 1
                    segments[index][:pos] = True
                texts = texts_[j]
                for i, bbox in enumerate(bboxes):
                    ids = texts[i]
                    N = len(ids)
                    assert pos + N + 1 <= MAX_LENGTH, 'Exceed max length'
                    for offset in range(N):
                        tokens[index][pos + offset] = ids[offset]
                        token_types[index][pos + offset] = WORD_TOKEN_TYPE
                    pos += N
                    tokens[index][pos] = LOCATE_TOKEN_ID
                    token_types[index][pos] = LOCATE_TOKEN_TYPE
                    locations[index][pos] = bbox
                    pos += 1
                if j == len(bboxes_) - 1:
                    tokens[index][pos] = EOS_TOKEN_ID
                elif pos <= MAX_LENGTH:
                    suffix_bboxes_, suffix_texts_ = bboxes_[j + 1][0], texts_[j + 1][0]
                    offset = 0
                    while pos <= MAX_LENGTH and offset < len(suffix_texts_):
                        tokens[index][pos] = suffix_texts_[offset]
                        token_types[index][pos] = WORD_TOKEN_TYPE
                        pos += 1
                        offset += 1
                    if pos <= MAX_LENGTH:
                        tokens[index][pos] = LOCATE_TOKEN_ID
                        token_types[index][pos] = LOCATE_TOKEN_TYPE
                        for k in range(4):
                            locations[index][pos][k] = suffix_bboxes_[k]
                        pos += 1
                index += 1
                f.write(image + '\n')
            shutil.copy(os.path.join(image_dir, image), os.path.join(preprocessed_image_dir, image))
    np.save(os.path.join(preprocessed_data_dir, 'localization-tokens-%d.npy' % MAX_LENGTH), tokens[:index])
    np.save(os.path.join(preprocessed_data_dir, 'localization-bboxes-%d.npy' % MAX_LENGTH), locations[:index])
    np.save(os.path.join(preprocessed_data_dir, 'localization-token_types-%d.npy' % MAX_LENGTH), token_types[:index])
    np.save(os.path.join(preprocessed_data_dir, 'localization-segments-%d.npy' % MAX_LENGTH), segments[:index])
    np.save(os.path.join(preprocessed_data_dir, 'localization-ignore_tokens-%d.npy' % MAX_LENGTH), np.zeros_like(tokens[:index], dtype=bool)) # A workaround to filter poor-quality pre-training OCR data, i.e., the rubbish Huawei OCR which could even misrecognize CJK characters in a pure English document. For the rubbish token, set the corresponding `ignore_tokens` as True to remove it from loss computation.


if __name__ == '__main__':
    process_text_bbox_data(
        metadata_dir = './SynthDog-bbox/preprocessed_data/data',
        image_dir = './SynthDog-bbox/preprocessed_data/images',
        preprocessed_data_dir = './text_bbox_data'
    )
