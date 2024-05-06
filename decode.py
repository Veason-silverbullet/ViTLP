import os
import json
import shutil
import torch
from torch.nn.functional import log_softmax
from PIL import Image
from argparse import ArgumentParser
from models.ViTLP.configuration_ViTLP import ViTLPConfig
from models.ViTLP.modeling_ViTLP import ViTLPModel
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, ImageFeatureExtractionMixin
from transformers import BartTokenizer
BBOX_SEARCH_SIZE = 4
BBOX_SEARCH_SIZES = [BBOX_SEARCH_SIZE, BBOX_SEARCH_SIZE, BBOX_SEARCH_SIZE, BBOX_SEARCH_SIZE]
DECODER_START_TOKEN_ID = 2
LOCATE_TOKEN_ID = 50265
CONTINUE_DECODE_ID = 50266
PREFIX_RATIO = 0.25
MAX_SEGMENT_NUM = 4
EOS_TOKEN_ID = 2
MAX_LENGTH = 1280
IOU_THRESHOLD = 0.5
IOU_UPPERBOUND = 0.8
RETRY_NUM = 2
parser = ArgumentParser(description='ViTLP OCR')
parser.add_argument('--pretrained_model', default='ckpts/ViTLP-medium', type=str, help='Pretrained ViTLP model')
parser.add_argument('--images', nargs='+', required=True, help='Decode image paths')
args = parser.parse_args()
tokenizer_config = 'configs/ViTLP-1920-1600'
tokenizer = BartTokenizer.from_pretrained(tokenizer_config)
config = ViTLPConfig.from_pretrained(args.pretrained_model)
config.gradient_checkpointing = False
config.LOCATE_TOKEN_ID = LOCATE_TOKEN_ID
assert config.decoder_start_token_id == DECODER_START_TOKEN_ID and config.bin_size == 1001
assert all([image_file[-4:] in ['.jpg', '.png', '.tif'] for image_file in args.images]), 'Image format must be in [\'.jpg\', \'.png\', \'.tif\'].'
ViTLP = ViTLPModel.from_pretrained(args.pretrained_model, config=config)
ViTLP = ViTLP.cuda()
ViTLP.eval()
lm_decoder = ViTLP.decoder.lm_decoder
bbox_output_embeddings = lm_decoder.bbox_output_embeddings
bbox_decoder_start_embedding = lm_decoder.bbox_decoder_start_embedding
bbox_decoder = lm_decoder.bbox_decoder
bbox_head = lm_decoder.bbox_head
hidden_size = config.hidden_size
device = torch.device('cuda')
PAD_BBOXES = torch.full([1, 1, 4], config.bin_size, dtype=torch.int32, device=device)
decode_output_dir = 'decode_output/' + os.path.basename(args.pretrained_model)
if not os.path.exists(decode_output_dir):
    os.makedirs(decode_output_dir)


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
        return torch.from_numpy(image)
vitFeatureExtractor = ViTFeatureExtractor(do_resize=True, size=[config.image_width, config.image_height], resample=config.resample, do_normalize=True)


# bboxes     : [num, 4]
# anchor_box : [4]
def IOU(bboxes, anchor_box):
    num = bboxes.size(0)
    anchor_box = anchor_box.float()
    anchor_box = anchor_box.repeat(num).view([num, 4])
    bboxes[:, 0] = torch.minimum(bboxes[:, 0], bboxes[:, 2])
    bboxes[:, 1] = torch.minimum(bboxes[:, 1], bboxes[:, 3])
    x_left = torch.maximum(bboxes[:, 0], anchor_box[:, 0])
    y_top = torch.maximum(bboxes[:, 1], anchor_box[:, 1])
    x_right = torch.minimum(bboxes[:, 2], anchor_box[:, 2])
    y_bottom = torch.minimum(bboxes[:, 3], anchor_box[:, 3])

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    anchor_box_area = (anchor_box[:, 2] - anchor_box[:, 0]) * (anchor_box[:, 3] - anchor_box[:, 1])

    iou = intersection_area / (bboxes_area + anchor_box_area - intersection_area)
    iou.masked_fill_((x_right < x_left) | (y_bottom < y_top), 0)
    return iou


# hidden_states : [batch_size, hidden_dim]
def bbox_decode(hidden_states, return_list):
    batch_size, hidden_dim = hidden_states.size()
    sample_num = BBOX_SEARCH_SIZES[0] * BBOX_SEARCH_SIZES[1] * BBOX_SEARCH_SIZES[2] * BBOX_SEARCH_SIZES[3]
    N = sample_num
    bbox_decode_index_table = torch.zeros([batch_size * sample_num, 4], dtype=torch.int64, device=device)
    for i in range(4):
        if i == 0:
            h = bbox_decoder(bbox_decoder_start_embedding.repeat(batch_size, 1), hidden_states, 0)
        else:
            h = bbox_decoder(bbox_output_embeddings(indices.flatten()), h.repeat(1, BBOX_SEARCH_SIZES[i - 1]).view([-1, hidden_dim]), i)
        probs = log_softmax(bbox_head[i % 2](h), dim=1)
        probs, indices = torch.topk(probs, k=BBOX_SEARCH_SIZES[i], dim=1) # [batch_size, BBOX_SEARCH_SIZE]
        #### update search table ####
        if i == 0:
            N //= BBOX_SEARCH_SIZES[i]
            bbox_logprobs = probs.unsqueeze(dim=2).repeat(1, 1, N).flatten()
            bbox_decode_index_table[:, 0] = indices.unsqueeze(dim=2).repeat(1, 1, N).flatten()
        elif i < 3:
            N //= BBOX_SEARCH_SIZES[i]
            bbox_logprobs += probs.unsqueeze(dim=2).repeat(1, 1, N).flatten()
            bbox_decode_index_table[:, i] = indices.unsqueeze(dim=2).repeat(1, 1, N).flatten()
        else:
            bbox_logprobs += probs.flatten()
            bbox_decode_index_table[:, 3] = indices.flatten()
        #### update search table ####
    indices = torch.argmax(bbox_logprobs.view([batch_size, sample_num]), dim=1) + torch.arange(start=0, end=batch_size, dtype=torch.int64, device=device) * sample_num
    decode_bboxes = bbox_decode_index_table.index_select(dim=0, index=indices)
    if return_list:
        return decode_bboxes, decode_bboxes.tolist()
    return decode_bboxes


def greedy_search(image):
    for _ in range(RETRY_NUM):
        encoder_outputs = ViTLP.encoder(image).last_hidden_state
        decoder_input_ids = torch.full([1, 1], config.decoder_start_token_id, dtype=torch.int32, device=device)
        decoder_input_bboxes = PAD_BBOXES
        i = 0
        decode_ids = []
        word_flag = True
        bboxes = None
        words = []
        pre_i = i
        repeat_cnt = {}
        # Greedy search without repetition
        while i < MAX_LENGTH:
            if i == 0:
                hidden_states, past_key_values = lm_decoder.forward_(encoder_outputs, decoder_input_ids, decoder_input_bboxes, past_key_values=None, use_cache=True)
            else:
                decoder_input_ids_ = decoder_input_ids[:, -1].unsqueeze(dim=1)
                decoder_input_bboxes_ = decoder_input_bboxes[:, -1, :].unsqueeze(dim=1)
                hidden_states, past_key_values = lm_decoder.forward_(encoder_outputs, decoder_input_ids_, decoder_input_bboxes_, past_key_values=past_key_values, use_cache=True)
            hidden_states = hidden_states.select(dim=1, index=0)
            if i == 0:
                index = torch.argmax(lm_decoder.lm_head(hidden_states)[:, 4:-3], dim=1) + 4
            elif word_flag:
                index = torch.argmax(lm_decoder.lm_head(hidden_states)[:, 2:-3], dim=1) + 2
            else:
                index = torch.argmax(lm_decoder.lm_head(hidden_states)[:, 2:-1], dim=1) + 2
            index_ = index.item()
            if index_ == LOCATE_TOKEN_ID:
                decode_bbox = bbox_decode(hidden_states, return_list=False)[0, :].unsqueeze(dim=0)
                word = tokenizer.decode(decode_ids).strip()
                if bboxes is None:
                    decode_flag = True
                else:
                    ious = IOU(bboxes, decode_bbox.squeeze(dim=0)).tolist()
                    decode_flag = all([iou < IOU_THRESHOLD or (iou <= IOU_UPPERBOUND and word not in words[iou_index][0] and words[iou_index][0] not in word) for iou_index, iou in enumerate(ious)])
                if decode_flag:
                    decoder_input_bboxes = torch.cat([decoder_input_bboxes, decode_bbox.unsqueeze(dim=1)], dim=1)
                    bboxes = decode_bbox if bboxes is None else torch.cat([bboxes, decode_bbox], dim=0)
                    words.append((word, decode_ids))
                    # print(bboxes[-1, :].tolist(), '\t', word)
                    pre_i = i + 1
                    repeat_cnt[pre_i] = 1
                    decode_ids = []
                    word_flag = True
                else:
                    i = pre_i
                    repeat_cnt[pre_i] += 1
                    decoder_input_ids = decoder_input_ids[:, :i + 1]
                    decoder_input_bboxes = decoder_input_bboxes[:, :i + 1, :]
                    hidden_states, past_key_values = lm_decoder.forward_(encoder_outputs, decoder_input_ids, decoder_input_bboxes, past_key_values=None, use_cache=True)
                    hidden_states = hidden_states.select(dim=1, index=i)
                    topk_values, topk_indices = torch.topk(lm_decoder.lm_head(hidden_states)[:, 2:-1], k=repeat_cnt[i], dim=1)
                    index = topk_indices[:, -1] + 2
                    index_ = index.item()
                    if index_ == EOS_TOKEN_ID:
                        results = []
                        bboxes = bboxes.tolist()
                        for i, (word, decode_ids) in enumerate(words):
                            results.append([bboxes[i], word])
                        return results
                    decode_ids = [index_]
                    decoder_input_bboxes = torch.cat([decoder_input_bboxes, PAD_BBOXES], dim=1)
                    word_flag = False
            elif index_ == EOS_TOKEN_ID:
                results = []
                bboxes = bboxes.tolist()
                for i, (word, decode_ids) in enumerate(words):
                    results.append([bboxes[i], word])
                return results
            else:
                decode_ids.append(index_)
                decoder_input_bboxes = torch.cat([decoder_input_bboxes, PAD_BBOXES], dim=1)
                word_flag = False
            decoder_input_ids = torch.cat([decoder_input_ids, index.unsqueeze(dim=0)], dim=1)
            i += 1
        if bboxes is None:
            image = torch.clamp(image * 2, -1, 1) # A workaround of improving image contrast to try to avoid decoding repetition. Mostly, this case would not happen.
            continue
        for i in range(MAX_SEGMENT_NUM):
            flag, bboxes, words = greedy_search_continue(encoder_outputs, bboxes, words) # multi-segment decoding
            if flag:
                break
        results = []
        bboxes = bboxes.tolist()
        for i, (word, decode_ids) in enumerate(words):
            results.append([bboxes[i], word])
        return results
    return None


def greedy_search_continue(encoder_outputs, bboxes, words):
    n = len(words)
    decoder_input_ids = torch.zeros([MAX_LENGTH], dtype=torch.int32)
    decoder_input_bboxes = torch.full([MAX_LENGTH, 4], config.bin_size, dtype=torch.int32)
    decoder_input_ids[0] = CONTINUE_DECODE_ID
    pos = 1
    for i in range(n - int(n * PREFIX_RATIO), n):
        bbox, ids = bboxes[i], words[i][1]
        K = len(ids)
        for offset in range(K):
            decoder_input_ids[pos + offset] = ids[offset]
        pos += K
        decoder_input_ids[pos] = LOCATE_TOKEN_ID
        decoder_input_bboxes[pos] = bbox
        pos += 1
    decoder_input_ids = decoder_input_ids[:pos].cuda().unsqueeze(dim=0)
    decoder_input_bboxes = decoder_input_bboxes[:pos, :].cuda().unsqueeze(dim=0)
    flag = False
    i = pos - 1
    decode_ids = []
    word_flag = True
    pre_i = i
    repeat_cnt = {pre_i: 1}
    past_key_values = None
    # Greedy search without repetition
    while i < MAX_LENGTH:
        if past_key_values is None:
            hidden_states, past_key_values = lm_decoder.forward_(encoder_outputs, decoder_input_ids, decoder_input_bboxes, past_key_values=past_key_values, use_cache=True)
            hidden_states = hidden_states.select(dim=1, index=i)
        else:
            decoder_input_ids_ = decoder_input_ids[:, -1].unsqueeze(dim=1)
            decoder_input_bboxes_ = decoder_input_bboxes[:, -1, :].unsqueeze(dim=1)
            hidden_states, past_key_values = lm_decoder.forward_(encoder_outputs, decoder_input_ids_, decoder_input_bboxes_, past_key_values=past_key_values, use_cache=True)
            hidden_states = hidden_states.select(dim=1, index=0)
        if word_flag:
            index = torch.argmax(lm_decoder.lm_head(hidden_states)[:, 2:-3], dim=1) + 2
        else:
            index = torch.argmax(lm_decoder.lm_head(hidden_states)[:, 2:-1], dim=1) + 2
        index_ = index.item()
        if index_ == LOCATE_TOKEN_ID:
            decode_bbox = bbox_decode(hidden_states, return_list=False)[0, :].unsqueeze(dim=0)
            word = tokenizer.decode(decode_ids).strip()
            ious = IOU(bboxes, decode_bbox.squeeze(dim=0)).tolist()
            decode_flag = all([iou < IOU_THRESHOLD or (iou <= IOU_UPPERBOUND and word not in words[iou_index][0] and words[iou_index][0] not in word) for iou_index, iou in enumerate(ious)])
            if decode_flag:
                decoder_input_bboxes = torch.cat([decoder_input_bboxes, decode_bbox.unsqueeze(dim=1)], dim=1)
                bboxes = torch.cat([bboxes, decode_bbox], dim=0)
                words.append((word, decode_ids))
                # print('\t', bboxes[-1, :].tolist(), '\t', word)
                pre_i = i + 1
                repeat_cnt[pre_i] = 1
                decode_ids = []
                word_flag = True
            else:
                i = pre_i
                repeat_cnt[pre_i] += 1
                decoder_input_ids = decoder_input_ids[:, :i + 1]
                decoder_input_bboxes = decoder_input_bboxes[:, :i + 1, :]
                hidden_states, past_key_values = lm_decoder.forward_(encoder_outputs, decoder_input_ids, decoder_input_bboxes, past_key_values=None, use_cache=True)
                hidden_states = hidden_states.select(dim=1, index=i)
                topk_values, topk_indices = torch.topk(lm_decoder.lm_head(hidden_states)[:, 2:-1], k=repeat_cnt[i], dim=1)
                index = topk_indices[:, -1] + 2
                index_ = index.item()
                if index_ == EOS_TOKEN_ID:
                    flag = True
                    break
                decode_ids = [index_]
                decoder_input_bboxes = torch.cat([decoder_input_bboxes, PAD_BBOXES], dim=1)
                word_flag = False
        elif index_ == EOS_TOKEN_ID:
            flag = True
            break
        else:
            decode_ids.append(index_)
            decoder_input_bboxes = torch.cat([decoder_input_bboxes, PAD_BBOXES], dim=1)
            word_flag = False
        decoder_input_ids = torch.cat([decoder_input_ids, index.unsqueeze(dim=0)], dim=1)
        i += 1
    return flag, bboxes, words


if __name__ == '__main__':
    for image_file in args.images:
        result_file = os.path.join(decode_output_dir, os.path.basename(image_file).replace('.png', '.json').replace('.jpg', '.json').replace('.tif', '.json'))
        print('\nDecoding: ' + image_file)
        with torch.no_grad():
            image = Image.open(image_file).convert('RGB')
            image = vitFeatureExtractor(image)
            image = image.cuda().unsqueeze(dim=0)
            results = greedy_search(image)
            shutil.copy(image_file, os.path.join(decode_output_dir, os.path.basename(image_file)))
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results if results is not None else [], f)
