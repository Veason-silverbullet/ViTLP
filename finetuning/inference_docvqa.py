import sys
sys.path.append('..')
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
LOCATE_TOKEN_ID = 50265
VQA_TOEKN_ID = 50261 # [VQA]
ANS_YES_TOKEN_ID = 50262 # [ANS_YES]
ANS_NO_TOKEN_ID = 50263 # [ANS_NO]
EOS_TOKEN_ID = 2
MAX_LENGTH = 1280
IOU_THRESHOLD = 0.5
IOU_UPPERBOUND = 0.8
parser = ArgumentParser(description='ViTLP OCR')
parser.add_argument('--vqa_finetuned_model', required=True, type=str, help='Fine-tuned ViTLP model')
parser.add_argument('--image_width', default=1920, type=int)
parser.add_argument('--image_height', default=2304, type=int)
parser.add_argument('--image', required=True, type=str, help='Image')
parser.add_argument('--question', required=True, type=str, help='Question')
args = parser.parse_args()
tokenizer = BartTokenizer.from_pretrained('../configs/ViTLP-1920-1600/')
config = ViTLPConfig.from_pretrained(args.vqa_finetuned_model)
config.gradient_checkpointing = False
config.LOCATE_TOKEN_ID = LOCATE_TOKEN_ID
assert config.bin_size == 1001
assert args.image[-4:] in ['.jpg', '.png', '.tif'], 'Image format must be in [\'.jpg\', \'.png\', \'.tif\'].'
ViTLP = ViTLPModel.from_pretrained(args.vqa_finetuned_model, config=config)
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
vitFeatureExtractor = ViTFeatureExtractor(do_resize=True, size=[args.image_width, args.image_height], resample=config.resample, do_normalize=True)


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


def vqa(image, decoder_input_ids):
    decoder_input_len = decoder_input_ids.size(1)
    assert image.size(0) == 1 and decoder_input_ids.size(0) == 1
    decoder_input_bboxes = torch.full([1, decoder_input_len, 4], config.bin_size, dtype=torch.int32, device=decoder_input_ids.device)
    encoder_outputs = ViTLP.encoder(image).last_hidden_state
    i = decoder_input_len
    decode_ids = []
    word_flag = True
    bboxes = None
    words = []
    pre_i = i
    repeat_cnt = {}
    # Greedy search without repetition
    while i < MAX_LENGTH:
        hidden_states = lm_decoder.forward_(encoder_outputs, decoder_input_ids, decoder_input_bboxes, past_key_values=None, use_cache=False) # To simplify coding (and save my limited energy), not use KV-cache here. This may make VQA decoding a little slower. For ViTLP KV-cache implementation, see ../decode.py.
        hidden_states = hidden_states.select(dim=1, index=i - 1)
        if i == decoder_input_len:
            index = torch.argmax(lm_decoder.lm_head(hidden_states)[:, 4:-3], dim=1) + 4
        elif word_flag:
            index = torch.argmax(lm_decoder.lm_head(hidden_states)[:, 2:-3], dim=1) + 2
        else:
            index = torch.argmax(lm_decoder.lm_head(hidden_states)[:, 2:-1], dim=1) + 2
        index_ = index.item()
        if i == decoder_input_len:
            if index_ == ANS_YES_TOKEN_ID:
                return [[None, 'Yes']]
            if index_ == ANS_NO_TOKEN_ID:
                return [[None, 'No']]
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
                hidden_states = lm_decoder.forward_(encoder_outputs, decoder_input_ids, decoder_input_bboxes, past_key_values=None, use_cache=False)
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
    results = []
    bboxes = bboxes.tolist()
    for i, (word, decode_ids) in enumerate(words):
        results.append([bboxes[i], word])
    return results


def reformat(s):
    while '  ' in s:
        s = s.replace('  ', ' ')
    while s.endswith(' ?'):
        s = s[:-2] + '?'
    while s.endswith('??'):
        s = s[:-2] + '?'
    while s.endswith(' .'):
        s = s[:-2] + '.'
    s = s.replace(' ,', ',')
    s = s.strip()
    return s


if __name__ == '__main__':
    with torch.no_grad():
        image = Image.open(args.image).convert('RGB')
        image = vitFeatureExtractor(image)
        image = image.cuda().unsqueeze(dim=0)
        decoder_input_ids = [[VQA_TOEKN_ID] + tokenizer.encode(reformat(args.question), add_special_tokens=False) + [EOS_TOKEN_ID]]
        decoder_input_ids = torch.IntTensor(decoder_input_ids).cuda()
        answer_with_bboxes = vqa(image, decoder_input_ids)
        answer_words = [item[1] for item in answer_with_bboxes]
        answer_bboxes = [item[0] for item in answer_with_bboxes]
        print('Image         :', args.image)
        print('Question      :', args.question)
        print('Answer words  :', answer_words)
        print('Answer bboxes :', answer_bboxes)
