import os
import json
import numpy as np
from transformers import BartTokenizer
BIN_SIZE = 1000
LOCATE_TOKEN_ID = 50265
# We re-use three unused tokens in the tokenizer (i.e., 50261, 50262, and 50263) as the special [VQA], [ANS_YES] and [ANS_NO] tokens.
VQA_TOEKN_ID = 50261 # [VQA]
ANS_YES_TOKEN_ID = 50262 # [ANS_YES]
ANS_NO_TOKEN_ID = 50263 # [ANS_NO]
PAD_TOKEN_ID = 1
EOS_TOKEN_ID = 2
MAX_LENGTH = 80
PAD_TOKEN_TYPE = 0
WORD_TOKEN_TYPE = 1
LOCATE_TOKEN_TYPE = 2
PAD_SPAN_TYPE = 0
QUESTION_SPAN_TYPE = 1
ANSWER_SPAN_TYPE = 2
tokenizer = BartTokenizer.from_pretrained('../configs/ViTLP-1920-1600')
USE_WORD_BBOX = True


def reformat(s):
    while '  ' in s:
        s = s.replace('  ', ' ')
    while s.endswith(' ?'):
        s = s[:-2] + '?'
    while s.endswith('??'):
        s = s[:-2] + '?'
    while s.endswith(' .'):
        s = s[:-2] + '.'
    s = s.strip()
    return s


def process_docvqa_train_data(data_dir):
    with open(os.path.join(data_dir, 'train-metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    N = len(metadata)
    tokens = np.full([N, MAX_LENGTH], PAD_TOKEN_ID, dtype=np.int32)
    bboxes = np.full([N, MAX_LENGTH, 4], BIN_SIZE + 1, dtype=np.int32)
    token_types = np.zeros([N, MAX_LENGTH], dtype=np.int8)
    qa_span_types = np.zeros([N, MAX_LENGTH], dtype=np.int8)
    tokens[:, 0] = VQA_TOEKN_ID
    max_pos = 0
    with open(os.path.join(data_dir, 'train-mapping.txt'), 'w', encoding='utf-8') as f:
        for index, item in enumerate(metadata):
            question = reformat(item['question'])
            question_tokens = tokenizer.encode(question, add_special_tokens=False) + [EOS_TOKEN_ID]
            pos = 1
            for token in question_tokens:
                assert token != LOCATE_TOKEN_ID
                tokens[index][pos] = token
                token_types[index][pos] = WORD_TOKEN_TYPE
                qa_span_types[index][pos] = QUESTION_SPAN_TYPE
                pos += 1
            assert (item['TYPE'] in ['answer_without_bbox', 'yes_no_answer']) == ('bboxes' not in item)
            if item['TYPE'] == 'yes_no_answer':
                answer = item['answer']
                if answer == '[ANS_YES]':
                    tokens[index][pos] = ANS_YES_TOKEN_ID
                    token_types[index][pos] = WORD_TOKEN_TYPE
                    qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                    pos += 1
                else:
                    assert answer == '[ANS_NO]', answer
                    tokens[index][pos] = ANS_NO_TOKEN_ID
                    token_types[index][pos] = WORD_TOKEN_TYPE
                    qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                    pos += 1
            elif item['TYPE'] == 'answer_with_bbox':
                if USE_WORD_BBOX:
                    for bbox, word in item['bboxes']:
                        answer_tokens = tokenizer.encode(' ' + word.strip(), add_special_tokens=False)
                        for token in answer_tokens:
                            assert token != LOCATE_TOKEN_ID
                            tokens[index][pos] = token
                            token_types[index][pos] = WORD_TOKEN_TYPE
                            qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                            pos += 1
                        tokens[index][pos] = LOCATE_TOKEN_ID
                        bboxes[index][pos] = bbox
                        token_types[index][pos] = LOCATE_TOKEN_TYPE
                        qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                        pos += 1
                    tokens[index][pos] = EOS_TOKEN_ID
                    token_types[index][pos] = WORD_TOKEN_TYPE
                    qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                    pos += 1
                else:
                    answer_tokens = tokenizer.encode(' ' + answer.strip(), add_special_tokens=False)
                    for token in answer_tokens:
                        assert token != LOCATE_TOKEN_ID
                        tokens[index][pos] = token
                        token_types[index][pos] = WORD_TOKEN_TYPE
                        qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                        pos += 1
                    tokens[index][pos] = LOCATE_TOKEN_ID
                    assert len(item['bboxes']) > 0
                    #### Here, we derive the region-level answer bbox by merging word-level answer bboxes ####
                    x1, y1, x2, y2 = item['bboxes'][0][0]
                    for i in range(1, len(item['bboxes'])):
                        x1 = min(x1, item['bboxes'][i][0][0])
                        y1 = min(y1, item['bboxes'][i][0][1])
                        x2 = max(x2, item['bboxes'][i][0][2])
                        y2 = max(y2, item['bboxes'][i][0][3])
                    bboxes[index][pos] = [x1, y1, x2, y2]
                    token_types[index][pos] = LOCATE_TOKEN_TYPE
                    qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                    pos += 1
            else: # A workaround to incorporate more training samples that cannot be linked with bounding boxes
                assert item['TYPE'] == 'answer_without_bbox', item['TYPE']
                if USE_WORD_BBOX:
                    for word in item['answer_words']:
                        answer_tokens = tokenizer.encode(' ' + word, add_special_tokens=False)
                        for token in answer_tokens:
                            assert token != LOCATE_TOKEN_ID
                            tokens[index][pos] = token
                            token_types[index][pos] = WORD_TOKEN_TYPE
                            qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                            pos += 1
                        tokens[index][pos] = LOCATE_TOKEN_ID
                        token_types[index][pos] = LOCATE_TOKEN_TYPE
                        qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                        pos += 1
                else:
                    answer_tokens = tokenizer.encode(' '.join(item['answer_words']), add_special_tokens=False)
                    for token in answer_tokens:
                        assert token != LOCATE_TOKEN_ID
                        tokens[index][pos] = token
                        token_types[index][pos] = WORD_TOKEN_TYPE
                        qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                        pos += 1
                    tokens[index][pos] = LOCATE_TOKEN_ID
                    token_types[index][pos] = LOCATE_TOKEN_TYPE
                    qa_span_types[index][pos] = ANSWER_SPAN_TYPE
                    pos += 1
            max_pos = max(max_pos, pos)
            f.write(item['image'] + '\n')
    print('Max length:', max_pos)
    np.save(os.path.join(data_dir, 'tokens-train-%d.npy' % MAX_LENGTH), tokens)
    np.save(os.path.join(data_dir, 'bboxes-train-%d.npy' % MAX_LENGTH), bboxes)
    np.save(os.path.join(data_dir, 'token_types-train-%d.npy' % MAX_LENGTH), token_types)
    np.save(os.path.join(data_dir, 'qa_span_types-train-%d.npy' % MAX_LENGTH), qa_span_types)


if __name__ == '__main__':
    process_docvqa_train_data(data_dir = './DocVQA')
