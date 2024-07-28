import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ImageFeatureExtractionMixin


# from https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/vit/feature_extraction_vit.py
class ViTFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    def __init__(
        self,
        do_resize=True,
        size=[1600, 1920],
        resample=Image.BILINEAR,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def __call__(self, image) -> torch.FloatTensor:
        if self.do_resize and self.size is not None:
            image = self.resize(image=image, size=self.size, resample=self.resample)
        if self.do_normalize:
            image = self.normalize(image=image, mean=self.image_mean, std=self.image_std)
        return torch.from_numpy(image).half()


class PretrainDataset(Dataset):
    def __init__(self, dataset_path, image_dir, config, mode, rank=None):
        self.rank = rank
        self.mode = mode
        assert self.mode in ['train', 'validation']
        self.image_dir = image_dir
        self.PAD_TOKEN_ID = config.pad_token_id  # 1
        self.PAD_BBOX_TOKEN_ID = config.bin_size # 1001
        self.IGNORE_INDEX = -100
        self.vitFeatureExtractor = ViTFeatureExtractor(do_resize=True, size=[config.image_width, config.image_height], resample=config.resample, do_normalize=True)
        self.image_map = []
        self.LOCATE_ID = config.vocab_size - 2
        assert self.LOCATE_ID == 50265 # hard-code confirmation
        if self.rank is not None:
            with open(os.path.join(dataset_path, 'mapping-%d.txt' % self.rank), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 0:
                        image, prefix = line.split('\t')
                        self.image_map.append((os.path.join(prefix + '_' + str(self.rank), image)))
                self.num = len(self.image_map)
            with open(os.path.join(dataset_path, 'localization-tokens-%d-%d.npy' % (self.rank, config.seq_length)), 'rb') as f:
                self.tokens = np.load(f).astype(np.int32)
                assert self.num == self.tokens.shape[0], 'Sizes mismatch in localization-tokens-%d-%d.npy, %d vs. %d' % (self.rank, config.seq_length, self.num, self.tokens.shape[0]) # sanity check
            with open(os.path.join(dataset_path, 'localization-bboxes-%d-%d.npy' % (self.rank, config.seq_length)), 'rb') as f:
                self.bboxes = np.load(f) # np.int16
                assert self.bboxes.dtype == np.int16
                assert self.num == self.bboxes.shape[0], 'Sizes mismatch in localization-bboxes-%d-%d.npy, %d vs. %d' % (self.rank, config.seq_length, self.num, self.bboxes.shape[0]) # sanity check
            with open(os.path.join(dataset_path, 'localization-token_types-%d-%d.npy' % (self.rank, config.seq_length)), 'rb') as f:
                token_types = np.load(f).astype(np.int8)
                # self.decoder_input_types = token_types[:, :-1]
                self.token_types = token_types[:, 1:]
                assert self.num == token_types.shape[0], 'Sizes mismatch in localization-token_types-%d-%d.npy, %d vs. %d' % (self.rank, config.seq_length, self.num, token_types.shape[0]) # sanity check
            with open(os.path.join(dataset_path, 'localization-segments-%d-%d.npy' % (self.rank, config.seq_length)), 'rb') as f:
                self.segments = ~np.load(f)[:, 1:]
                assert self.num == self.segments.shape[0], 'Sizes mismatch in localization-segments-%d-%d.npy, %d vs. %d' % (self.rank, config.seq_length, self.num, self.segments.shape[0]) # sanity check
            with open(os.path.join(dataset_path, 'localization-ignore_tokens-%d-%d.npy' % (self.rank, config.seq_length)), 'rb') as f:
                self.ignore_tokens = np.load(f)[:, 1:]
                assert self.num == self.ignore_tokens.shape[0], 'Sizes mismatch in localization-ignore_tokens-%d-%d.npy, %d vs. %d' % (self.rank, config.seq_length, self.num, self.ignore_tokens.shape[0]) # sanity check
        else:
            with open(os.path.join(dataset_path, 'mapping.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    image = line.strip()
                    if len(image) > 0:
                        self.image_map.append(image)
                self.num = len(self.image_map)
            with open(os.path.join(dataset_path, 'localization-tokens-%d.npy' % config.seq_length), 'rb') as f:
                self.tokens = np.load(f).astype(np.int32)
                assert self.num == self.tokens.shape[0], 'Sizes mismatch in localization-tokens-%d.npy, %d vs. %d' % (config.seq_length, self.num, self.tokens.shape[0]) # sanity check
            with open(os.path.join(dataset_path, 'localization-bboxes-%d.npy' % config.seq_length), 'rb') as f:
                self.bboxes = np.load(f) # np.int16
                assert self.bboxes.dtype == np.int16
                assert self.num == self.bboxes.shape[0], 'Sizes mismatch in localization-bboxes-%d.npy, %d vs. %d' % (config.seq_length, self.num, self.bboxes.shape[0]) # sanity check
            with open(os.path.join(dataset_path, 'localization-token_types-%d.npy' % config.seq_length), 'rb') as f:
                token_types = np.load(f).astype(np.int8)
                # self.decoder_input_types = token_types[:, :-1]
                self.token_types = token_types[:, 1:]
                assert self.num == token_types.shape[0], 'Sizes mismatch in localization-token_types-%d.npy, %d vs. %d' % (config.seq_length, self.num, token_types.shape[0]) # sanity check
            with open(os.path.join(dataset_path, 'localization-segments-%d.npy' % config.seq_length), 'rb') as f:
                self.segments = ~np.load(f)[:, 1:]
                assert self.num == self.segments.shape[0], 'Sizes mismatch in localization-segments-%d.npy, %d vs. %d' % (config.seq_length, self.num, self.segments.shape[0]) # sanity check
            with open(os.path.join(dataset_path, 'localization-ignore_tokens-%d.npy' % config.seq_length), 'rb') as f:
                self.ignore_tokens = np.load(f)[:, 1:]
                assert self.num == self.ignore_tokens.shape[0], 'Sizes mismatch in localization-ignore_tokens-%d.npy, %d vs. %d' % (config.seq_length, self.num, self.ignore_tokens.shape[0]) # sanity check

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        decoder_input_ids = tokens[:-1]
        segments = self.segments[idx]
        ignore_tokens = self.ignore_tokens[idx]
        labels = tokens[1:]
        labels = np.where((labels != self.PAD_TOKEN_ID) & segments, labels, self.IGNORE_INDEX)
        labels = np.where(ignore_tokens, self.IGNORE_INDEX, labels).astype(np.int64)
        bboxes = self.bboxes[idx].astype(np.int32)
        decoder_input_bboxes = bboxes[:-1]
        bboxes = bboxes[1:]
        bboxes = np.where((bboxes != self.PAD_BBOX_TOKEN_ID) & np.expand_dims(segments, axis=1), bboxes, self.IGNORE_INDEX).astype(np.int64) # this is very crucial for a trick `bbox_input_ids = bboxes[:, :, :3].to(torch.int32)` in modeling_ViTLP.py
        n1 = (labels == self.LOCATE_ID).astype(np.float32).sum()
        n2 = (labels != self.IGNORE_INDEX).astype(np.float32).sum()
        if self.mode == 'train':
            sample = {
                'image': self.vitFeatureExtractor(Image.open(os.path.join(self.image_dir, self.image_map[idx])).convert('RGB')),
                'decoder_input_ids': decoder_input_ids,
                'decoder_input_bboxes': decoder_input_bboxes,
                'labels': labels,
                'bboxes': bboxes,
                'n1': n1,
                'n2': n2
            }
        else:
            sample = {
                'image': self.vitFeatureExtractor(Image.open(os.path.join(self.image_dir, self.image_map[idx])).convert('RGB')),
                'decoder_input_ids': decoder_input_ids,
                'decoder_input_bboxes': decoder_input_bboxes,
                'labels': labels,
                'bboxes': bboxes,
                'token_types': self.token_types[idx]
            }
        return sample
