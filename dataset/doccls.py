import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ImageFeatureExtractionMixin


# from https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/vit/feature_extraction_vit.py
class ViTFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    def __init__(
        self,
        do_resize=True,
        size=[1536, 1920],
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
        return image


class DocClsDataset(Dataset):
    def __init__(self, dataset_path, config):
        self.vitFeatureExtractor = ViTFeatureExtractor(do_resize=True, size=[config.image_width, config.image_height], resample=config.resample, do_normalize=True)
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        with open(os.path.join(dataset_path, 'mapping.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                if len(line.strip()) > 0:
                    image, label = line.strip().split('\t')
                    self.images.append(os.path.join(self.dataset_path, image))
                    self.labels.append(int(label))
        self.num = len(self.images)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        sample = {
            'image': self.vitFeatureExtractor(Image.open(self.images[idx]).convert('RGB')),
            'labels': self.labels[idx]
        }
        return sample
