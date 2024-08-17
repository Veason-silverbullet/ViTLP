import os
import json
import re
import editdistance
from tqdm import tqdm
BIN_SIZE = 1000
OCRS = ['ViTLP-OCR', 'Paddle-OCR', 'MS-OCR', 'ViTLP-OCR-augmented', 'ViTLP-OCR-early_stopping', 'ViTLP-OCR-augmented-early_stopping']


def str_eq(s1_: str, s2_: str) -> bool:
    # return s1 == s2
    fuzz = lambda s: s.replace('–', '-').replace('’', '\'').replace('∗', '*').replace('ä', 'a').replace('ă', 'a').replace('ö', 'o').replace('Ö', 'O').replace('ő', 'o').replace('ò', 'o').replace('ó', 'o').replace('ô', 'o').replace('é', 'e').replace('ú', 'u').replace('ü', 'u').replace('Á', 'A').replace('×', 'x').replace('ț', 't').replace('ș', 's').replace('î', 'i').replace('C', 'c').replace('K', 'k').replace('O', 'o').replace('P', 'p').replace('S', 's').replace('U', 'u').replace('V', 'v').replace('W', 'w').replace('X', 'x').replace('Z', 'z').replace('1', 'l').replace('0', 'o').replace('\u2013', '-').replace('\u2014', '-').replace('\u2018', '\'').replace('\u2019', '\'').replace('\u00b4', '\'').replace('\u0060', '\'').replace('“', '\"').replace('”', '\"').replace('\uff02', '\"').replace('″', '\"').replace('•', '.').replace('●', '.').replace('\u201c', '\"').replace('\u201d', '\"').replace('\u2018', '\'').replace('\u2019', '\'').replace(' ', ' ')
    s1 = fuzz(s1_)
    s2 = fuzz(s2_)
    if s1 == s2:
        return True
    n, m = len(s1), len(s2)
    if n < 10 and m < 10:
        return s1 == s2
    else:
        if all([s1[0] == s1[i] for i in range(1, n)]) and all([s2[0] == s2[j] for j in range(1, m)]) and s1[0] == s2[0]:
            return True
    if n - m >= 2 or m - n >= 2:
        return False
    return editdistance.eval(s1, s2) <= 1


def formalize(s: str) -> str:
    s = s.replace('\t', ' ').replace(' ', ' ').strip()
    while '  ' in s:
        s = s.replace('  ', ' ')
    while s.endswith('??') or s.endswith(' ?'):
        s = s[:-2] + '?'
    s = s.replace('in19', 'in 19').replace('in20', 'in 20').replace(' the the ', ' the ').replace(' of of ', ' of ').replace(' are are ', ' are ')
    for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']:
        for i in range(10):
            s = s.replace(month + ',' + str(i), month + ', ' + str(i))
    if not s.endswith(' the\''):
        s = s.replace(' the\'', ' the \'')
    s = s.replace(' ,', ',')
    pos1 = re.findall('\( ', s)
    pos2 = re.findall(' \)', s)
    if len(pos1) == len(pos2) == 1 and pos1[0] < pos2[0]:
        s = s.replace('( ', '(').replace(' )', ')')
    return s


def link(answer: str, ocr: list, normalized: bool=False):
    answer_words = answer.split(' ')
    n = len(ocr)
    m = len(answer_words)
    link_spans = []
    if n >= m:
        for i in range(n - m):
            for j in range(i, i + m):
                if not normalized:
                    if not str_eq(ocr[j][1], answer_words[j - i]):
                        break
                else:
                    if not str_eq(ocr[j][1].lower(), answer_words[j - i].lower()):
                        break
            else:
                link_spans.append(ocr[i: i + m])
    else:
        return None
    if len(link_spans) >= 1:
        return link_spans
    return None


def link_wo_whitespace(answer: str, ocr: list, normalized: bool=False):
    n = len(ocr)
    N = 0
    text = ''
    for item in ocr:
        word = item[1].strip()
        if not normalized:
            text += word
        else:
            text += word.lower()
        N += len(word)
    for suffix_index, suffix in enumerate(['', ',', '.', ')', '%']):
        answer_ = answer.replace(' ', '') + suffix
        if normalized:
            answer_ = answer_.lower()
        if answer_ in text:
            states = [-1 for _ in range(N + 1)]
            states[0] = 0
            pos = 0
            for i in range(n):
                pos += len(ocr[i][1].strip())
                states[pos] = i + 1
            indices = [i.start() for i in re.finditer(re.escape(answer_), text)]
            assert len(indices) > 0, '%s\n%s\n%s' % (answer_, text, str(answer_ in text))
            link_spans = []
            for index in indices:
                start_index = states[index]
                end_index = states[index + len(answer_)]
                if start_index != -1 and end_index != -1:
                    assert start_index < end_index, '%d vs. %d' % (start_index, end_index)
                    link_spans.append(ocr[start_index: end_index])
                    if suffix_index > 0:
                        link_spans[-1][-1][1] = link_spans[-1][-1][1][:-1]
            if len(link_spans) > 0:
                return link_spans
    return None


if __name__ == '__main__':
    with open('train_v1.0_withQT.json', 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    metadata = []
    for item in tqdm(data):
        questionId = item['questionId']
        question = formalize(item['question'])
        if question == '`what is the auth no. for inavo n rivers?': # patch
            question = 'what is the auth no. for inavo n rivers?'
        image = item['image']
        answers = list(map(formalize, item['answers']))
        assert image.startswith('documents/') and image.endswith('.png'), image
        if any([answer.strip().lower() in ['yes.', 'yes'] for answer in answers]):
            linked_span = [None, '[ANS_YES]']
        elif any([answer.strip().lower() in ['no.', 'no'] for answer in answers]):
            linked_span = [None, '[ANS_NO]']
        else:
            linked_span = None
            cache_spans = []
            for ocr_root in OCRS:
                ocr_file = os.path.join(ocr_root, image[len('documents/'): -len('.png')] + '.json')
                with open(ocr_file, 'r', encoding='utf-8') as f:
                    ocr = list(filter(lambda x: len(x[1].strip()) > 0, json.load(f)))
                for answer in answers:
                    if answer == '11-Dcc-95': # patch
                        answer = '11-Dec-95'
                    linked_span_ = link(answer, ocr)
                    if linked_span_ is not None:
                        if len(linked_span_) == 1:
                            linked_span = [linked_span_[0], answer]
                            break
                        else:
                            cache_spans.append([linked_span_, answer])
                if linked_span is not None:
                    break
            if linked_span is None:
                if len(cache_spans) > 0:
                    cache_spans.sort(key=lambda x: len(x[0]))
                    linked_span = [cache_spans[0][0][0], cache_spans[0][1]]
                else:
                    for ocr_root in OCRS:
                        ocr_file = os.path.join(ocr_root, image[len('documents/'): -len('.png')] + '.json')
                        with open(ocr_file, 'r', encoding='utf-8') as f:
                            ocr = list(filter(lambda x: len(x[1].strip()) > 0, json.load(f)))
                        for answer in answers:
                            if answer == '11-Dcc-95': # patch
                                answer = '11-Dec-95'
                            linked_span_ = link_wo_whitespace(answer, ocr)
                            if linked_span_ is not None:
                                if len(linked_span_) == 1:
                                    linked_span = [linked_span_[0], answer]
                                    break
                                else:
                                    cache_spans.append([linked_span_, answer])
                        if linked_span is not None:
                            break
                    if linked_span is None:
                        if len(cache_spans) > 0:
                            cache_spans.sort(key=lambda x: len(x[0]))
                            linked_span = [cache_spans[0][0][0], cache_spans[0][1]]
                        else:
                            for ocr_root in OCRS:
                                ocr_file = os.path.join(ocr_root, image[len('documents/'): -len('.png')] + '.json')
                                with open(ocr_file, 'r', encoding='utf-8') as f:
                                    ocr = list(filter(lambda x: len(x[1].strip()) > 0, json.load(f)))
                                for answer in answers:
                                    if answer == '11-Dcc-95': # patch
                                        answer = '11-Dec-95'
                                    linked_span_ = link(answer, ocr, normalized=True)
                                    if linked_span_ is not None:
                                        if len(linked_span_) == 1:
                                            linked_span = [linked_span_[0], answer]
                                            break
                                        else:
                                            cache_spans.append([linked_span_, answer])
                                if linked_span is not None:
                                    break
                            if linked_span is None:
                                if len(cache_spans) > 0:
                                    cache_spans.sort(key=lambda x: len(x[0]))
                                    linked_span = [cache_spans[0][0][0], cache_spans[0][1]]
                                else:
                                    for ocr_root in OCRS:
                                        ocr_file = os.path.join(ocr_root, image[len('documents/'): -len('.png')] + '.json')
                                        with open(ocr_file, 'r', encoding='utf-8') as f:
                                            ocr = list(filter(lambda x: len(x[1].strip()) > 0, json.load(f)))
                                        for answer in answers:
                                            if answer == '11-Dcc-95': # patch
                                                answer = '11-Dec-95'
                                            linked_span_ = link_wo_whitespace(answer, ocr, normalized=True)
                                            if linked_span_ is not None:
                                                if len(linked_span_) == 1:
                                                    linked_span = [linked_span_[0], answer]
                                                    break
                                                else:
                                                    cache_spans.append([linked_span_, answer])
                                        if linked_span is not None:
                                            break
                                    if linked_span is None:
                                        if len(cache_spans) > 0:
                                            cache_spans.sort(key=lambda x: len(x[0]))
                                            linked_span = [cache_spans[0][0][0], cache_spans[0][1]]
        if linked_span is not None:
            if linked_span[0] is not None:
                metadata.append({
                    'image': image,
                    'questionId': questionId,
                    'question': question,
                    'answer': linked_span[1],
                    'bboxes': linked_span[0]
                })
            else:
                metadata.append({
                    'image': image,
                    'questionId': questionId,
                    'question': question,
                    'answer': linked_span[1]
                })
    print('%d / %d' % (len(metadata), len(data)))
    with open('train-metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
