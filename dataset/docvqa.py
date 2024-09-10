import os
import torch
import json
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
        fp16=True,
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
        self.fp16 = fp16

    def __call__(self, image) -> torch.FloatTensor:
        if self.do_resize and self.size is not None:
            image = self.resize(image=image, size=self.size, resample=self.resample)
        if self.do_normalize:
            image = self.normalize(image=image, mean=self.image_mean, std=self.image_std)
        return torch.from_numpy(image).half() if self.fp16 else image


class DocVQATrainDataset(Dataset):
    def __init__(self, dataset_path, config, image_width, image_height, fp16=True):
        self.PAD_TOKEN_ID = config.pad_token_id  # 1
        self.PAD_BBOX_TOKEN_ID = config.bin_size # 1001
        self.IGNORE_INDEX = -100
        self.vitFeatureExtractor = ViTFeatureExtractor(do_resize=True, size=[image_width, image_height], resample=config.resample, do_normalize=True, fp16=fp16)
        self.LOCATE_ID = config.vocab_size - 2
        assert self.LOCATE_ID == 50265 # hard-code confirmation
        self.ANSWER_SPAN_TYPE = 2
        self.image_map = []
        with open(os.path.join(dataset_path, 'train-mapping.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                image = line.strip()
                if len(image) > 0:
                    self.image_map.append(os.path.join(dataset_path, image))
        self.num = len(self.image_map)
        with open(os.path.join(dataset_path, 'tokens-train-%d.npy' % config.docvqa_seq_length), 'rb') as f:
            tokens = np.load(f).astype(np.int32)
            self.decoder_input_ids = tokens[:, :-1]
            self.labels = tokens[:, 1:]
            self.labels = np.where(self.labels != self.PAD_TOKEN_ID, self.labels, self.IGNORE_INDEX).astype(np.int64)
            assert self.num == tokens.shape[0], 'Sizes mismatch in tokens-train-%d.npy, %d vs. %d' % (config.docvqa_seq_length, self.num, tokens.shape[0]) # sanity check
            with open(os.path.join(dataset_path, 'qa_span_types-train-%d.npy' % config.docvqa_seq_length), 'rb') as f_:
                qa_span_types = np.load(f_)[:, 1:]
                self.labels = np.where(qa_span_types == self.ANSWER_SPAN_TYPE, self.labels, self.IGNORE_INDEX)
                assert self.num == qa_span_types.shape[0], 'Sizes mismatch in qa_span_types-train-%d.npy, %d vs. %d' % (config.docvqa_seq_length, self.num, tokens.shape[0]) # sanity check
        with open(os.path.join(dataset_path, 'bboxes-train-%d.npy' % config.docvqa_seq_length), 'rb') as f:
            bboxes = np.load(f).astype(np.int32)
            self.decoder_input_bboxes = bboxes[:, :-1]
            self.bboxes = bboxes[:, 1:]
            self.bboxes = np.where(self.bboxes != self.PAD_BBOX_TOKEN_ID, self.bboxes, self.IGNORE_INDEX)
            self.bboxes = np.where(np.expand_dims(qa_span_types == self.ANSWER_SPAN_TYPE, axis=2), self.bboxes, self.IGNORE_INDEX).astype(np.int64) # this is very crucial for a trick `bbox_input_ids = bboxes[:, :, :3].to(torch.int32)` in modeling_ViTLP.py
            assert self.num == bboxes.shape[0], 'Sizes mismatch in bboxes-train-%d.npy, %d vs. %d' % (config.docvqa_seq_length, self.num, self.bboxes.shape[0]) # sanity check
            assert np.all((np.all(self.bboxes == self.IGNORE_INDEX, axis=2)) | (qa_span_types == self.ANSWER_SPAN_TYPE)) # make sure all bbox tokens appear in answer spans

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        labels = self.labels[idx]
        bboxes = self.bboxes[idx]
        n1 = (bboxes[:, 0] != self.IGNORE_INDEX).astype(np.float32).sum()
        n2 = (labels != self.IGNORE_INDEX).astype(np.float32).sum()
        sample = {
            'image': self.vitFeatureExtractor(Image.open(self.image_map[idx]).convert('RGB')),
            'decoder_input_ids': self.decoder_input_ids[idx],
            'decoder_input_bboxes': self.decoder_input_bboxes[idx],
            'labels': labels,
            'bboxes': bboxes,
            'n1': n1,
            'n2': n2
        }
        return sample


class DocVQAInferDataset(Dataset):
    def __init__(self, dataset_path, config, image_width, image_height, mode, rank, world_size):
        assert mode in ['val', 'test']
        self.mode = mode
        self.rank = rank
        self.world_size = world_size
        self.vitFeatureExtractor = ViTFeatureExtractor(do_resize=True, size=[image_width, image_height], resample=config.resample, do_normalize=True, fp16=False)
        self.vqa_info = []
        self.image_map = []
        indices = []
        with open(os.path.join(dataset_path, 'meta.json'), 'r', encoding='utf-8') as f:
            self.meta_data = json.load(f)
            for index in range(len(self.meta_data)):
                assert index == self.meta_data[index]['index']
                if index % self.world_size == self.rank:
                    image_path = os.path.join(dataset_path, self.meta_data[index]['image'])
                    self.image_map.append(image_path)
                    questionId = self.meta_data[index]['questionId']
                    self.vqa_info.append([questionId, image_path])
                    indices.append(index)
            self.num = len(self.image_map)
        with open(os.path.join(dataset_path, 'tokens-%s-%d.npy' % (self.mode, config.docvqa_seq_length)), 'rb') as f:
            decoder_input_ids = np.load(f)
            for i in range(len(decoder_input_ids)):
                decoder_input_ids[i] = decoder_input_ids[i].astype(np.int32)
            self.decoder_input_ids = [decoder_input_ids[index] for index in indices]
            assert self.num == len(self.decoder_input_ids), 'Sizes mismatch in tokens-%s-%d.npy, %d vs. %d' % (self.mode, config.docvqa_seq_length, self.num, len(self.decoder_input_ids)) # sanity check
        with open(os.path.join(dataset_path, 'bboxes-%s-%d.npy' % (self.mode, config.docvqa_seq_length)), 'rb') as f:
            decoder_input_bboxes = np.load(f)
            for i in range(len(decoder_input_bboxes)):
                decoder_input_bboxes[i] = decoder_input_bboxes[i].astype(np.int32)
            self.decoder_input_bboxes = [decoder_input_bboxes[index] for index in indices]
            assert self.num == len(self.decoder_input_bboxes), 'Sizes mismatch in bboxes-%s-%d.npy, %d vs. %d' % (self.mode, config.docvqa_seq_length, self.num, len(self.decoder_input_bboxes)) # sanity check
        self.ground_truth = os.path.join(dataset_path, mode + '_v1.0.json')
        assert os.path.exists(self.ground_truth), 'Ground truth file does not exist: ' + self.ground_truth

    def __len__(self):
        return self.num

    # assert batch_size == 1
    def __getitem__(self, idx):
        sample = {
            'image': self.vitFeatureExtractor(Image.open(self.image_map[idx]).convert('RGB')),
            'decoder_input_ids': self.decoder_input_ids[idx],
            'decoder_input_bboxes': self.decoder_input_bboxes[idx]
        }
        return sample
