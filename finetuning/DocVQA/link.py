import os
import json
import re
import editdistance
from copy import deepcopy
from tqdm import tqdm
if not os.path.exists('ocr-cache.json'):
    OCRS = ['ViTLP-OCR', 'Paddle-OCR', 'MS-OCR', 'ViTLP-OCR-augmented', 'ViTLP-OCR-early_stopping', 'ViTLP-OCR-augmented-early_stopping']
    images = []
    with open('images.txt', 'r', encoding='utf-8') as f:
        for line in f:
            image = line.strip()
            if len(image) > 0:
                assert image.endswith('.png')
                images.append(image)
    for OCR in OCRS:
        if not os.path.exists(OCR + '-split'):
            os.mkdir(OCR + '-split')
        for image in images:
            with open(os.path.join(OCR, image[:-4] + '.json'), 'r', encoding='utf-8') as f:
                data = []
                for item in json.load(f):
                    bbox, word = item
                    word = word.strip()
                    if '//' in word or '/' not in word:
                        data.append(item)
                    else:
                        w1, h1, w2, h2 = bbox
                        w = w2 - w1
                        s = ''
                        sub_words = []
                        for c in word:
                            if c == '/':
                                if s != '':
                                    sub_words.append(s)
                                    s = ''
                                sub_words.append('/')
                            else:
                                s += c
                        if s != '':
                            sub_words.append(s)
                        M = len(sub_words)
                        lens = [0 for _ in range(M + 1)]
                        for i in range(1, M + 1):
                            lens[i] = len(sub_words[i - 1])
                        _len_ = len(word)
                        for i in range(M):
                            data.append([[int(w1 + lens[i] * w / _len_), h1, int(w1 + lens[i + 1] * w / _len_), h2], sub_words[i]])
                with open(os.path.join(OCR + '-split', image[:-4] + '.json'), 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
    OCRS += list(map(lambda x: x + '-split', OCRS))
    ocr_cache = {}
    for ocr_root in OCRS:
        for image in images:
            ocr = []
            with open(os.path.join(ocr_root, image[:-len('.png')] + '.json'), 'r', encoding='utf-8') as f:
                for item in filter(lambda x: len(x[1].strip()) > 0, json.load(f)):
                    bbox, word = item
                    word = word.strip()
                    if ' ' not in word:
                        ocr.append(item)
                    else:
                        w1, h1, w2, h2 = bbox
                        w = w2 - w1
                        s = ''
                        sub_words = []
                        for c in word:
                            if c == ' ':
                                if s != '':
                                    sub_words.append(s)
                                    s = ''
                                sub_words.append(' ')
                            else:
                                s += c
                        if s != '':
                            sub_words.append(s)
                        M = len(sub_words)
                        lens = [0 for _ in range(M + 1)]
                        for i in range(1, M + 1):
                            lens[i] = len(sub_words[i - 1])
                        _len_ = len(word)
                        for i in range(M):
                            if len(sub_words[i].strip()) > 0:
                                ocr.append([[int(w1 + lens[i] * w / _len_), h1, int(w1 + lens[i + 1] * w / _len_), h2], sub_words[i]])
            ocr_cache[ocr_root + image] = ocr
    with open('ocr-cache.json', 'w', encoding='utf-8') as f:
        json.dump({
            'OCRS': OCRS,
            'ocr_cache': ocr_cache
        }, f)
else:
    with open('ocr-cache.json', 'r', encoding='utf-8') as f:
        ocr_cache_data = json.load(f)
    OCRS = ocr_cache_data['OCRS']
    ocr_cache = ocr_cache_data['ocr_cache']


def str_eq(s1_: str, s2_: str) -> bool:
    # return s1 == s2
    fuzz = lambda s: s.replace('–', '-').replace('’', '\'').replace('∗', '*').replace('ä', 'a').replace('ă', 'a').replace('ö', 'o').replace('Ö', 'O').replace('ő', 'o').replace('ò', 'o').replace('ó', 'o').replace('ô', 'o').replace('é', 'e').replace('ú', 'u').replace('ü', 'u').replace('Á', 'A').replace('×', 'x').replace('ț', 't').replace('ș', 's').replace('î', 'i').replace('C', 'c').replace('K', 'k').replace('O', 'o').replace('P', 'p').replace('S', 's').replace('U', 'u').replace('V', 'v').replace('W', 'w').replace('X', 'x').replace('Z', 'z').replace('1', 'l').replace('0', 'o').replace('\u2013', '-').replace('\u2014', '-').replace('\u2018', '\'').replace('\u2019', '\'').replace('\u00b4', '\'').replace('\u0060', '\'').replace('“', '\"').replace('”', '\"').replace('\uff02', '\"').replace('″', '\"').replace('•', '.').replace('●', '.').replace('\u201c', '\"').replace('\u201d', '\"').replace(' ', ' ')
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
    while s.endswith(' .'):
        s = s[:-2] + '.'
    s = s.replace('in19', 'in 19').replace('in20', 'in 20').replace(' the the ', ' the ').replace(' of of ', ' of ').replace(' are are ', ' are ')
    for month in ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Sept', 'Oct', 'Nov', 'Dec']:
        for i in range(32):
            s = s.replace(month + ',' + str(i), month + ', ' + str(i))
    if not s.endswith(' the\''):
        s = s.replace(' the\'', ' the \'')
    s = s.replace(' ,', ',')
    pos1 = re.findall('\( ', s)
    pos2 = re.findall(' \)', s)
    if len(pos1) == len(pos2) == 1 and pos1[0] < pos2[0]:
        s = s.replace('( ', '(').replace(' )', ')')
    return s


def link(answer: str, ocr: list, normalized: bool=False, indexing: bool=False):
    answer_words = answer.split(' ')
    n = len(ocr)
    m = len(answer_words)
    link_spans = []
    if n >= m:
        for i in range(n - m):
            flag = True
            for j in range(i, i + m):
                if j != i + m - 1:
                    if not normalized:
                        if not str_eq(ocr[j][1], answer_words[j - i]):
                            flag = False
                            break
                    else:
                        if not str_eq(ocr[j][1].lower(), answer_words[j - i].lower()):
                            flag = False
                            break
                else:
                    if not normalized:
                        if (not str_eq(ocr[j][1], answer_words[j - i])) and not (answer_words[j - i][-1] != ocr[j][1][-1] and ocr[j][1][-1] in ',.;:\"\'?)]' and str_eq(ocr[j][1][:-1], answer_words[j - i])):
                            flag = False
                            break
                        if (not str_eq(ocr[j][1], answer_words[j - i])) and (answer_words[j - i][-1] != ocr[j][1][-1] and ocr[j][1][-1] in ',.;:\"\'?)]' and str_eq(ocr[j][1][:-1], answer_words[j - i])):
                            ocr[j][1] = ocr[j][1][:-1]
                    else:
                        if (not str_eq(ocr[j][1].lower(), answer_words[j - i].lower())) and not (answer_words[j - i][-1] != ocr[j][1][-1] and ocr[j][1][-1] in ',.;:\"\'?)]' and str_eq(ocr[j][1][:-1].lower(), answer_words[j - i].lower())):
                            flag = False
                            break
                        if (not str_eq(ocr[j][1].lower(), answer_words[j - i].lower())) and (answer_words[j - i][-1] != ocr[j][1][-1] and ocr[j][1][-1] in ',.;:\"\'?)]' and str_eq(ocr[j][1][:-1].lower(), answer_words[j - i].lower())):
                            ocr[j][1] = ocr[j][1][:-1]
            if flag:
                if not indexing:
                    link_spans.append(ocr[i: i + m])
                else:
                    link_spans.append([ocr[i: i + m], [i, i + m - 1]])
    else:
        return None
    if len(link_spans) >= 1:
        return link_spans
    _answer_word_ = deepcopy(answer_words[0])
    for delimiter in '\"\'([':
        if _answer_word_[0] != delimiter:
            answer_words[0] = delimiter + _answer_word_
            for i in range(n - m):
                flag = True
                for j in range(i, i + m):
                    if j != i + m - 1:
                        if not normalized:
                            if not str_eq(ocr[j][1], answer_words[j - i]):
                                flag = False
                                break
                        else:
                            if not str_eq(ocr[j][1].lower(), answer_words[j - i].lower()):
                                flag = False
                                break
                    else:
                        if not normalized:
                            if (not str_eq(ocr[j][1], answer_words[j - i])) and not (answer_words[j - i][-1] != ocr[j][1][-1] and ocr[j][1][-1] in ',.;:\"\'?)]' and str_eq(ocr[j][1][:-1], answer_words[j - i])):
                                flag = False
                                break
                            if (not str_eq(ocr[j][1], answer_words[j - i])) and (answer_words[j - i][-1] != ocr[j][1][-1] and ocr[j][1][-1] in ',.;:\"\'?)]' and str_eq(ocr[j][1][:-1], answer_words[j - i])):
                                ocr[j][1] = ocr[j][1][:-1]
                        else:
                            if (not str_eq(ocr[j][1].lower(), answer_words[j - i].lower())) and not (answer_words[j - i][-1] != ocr[j][1][-1] and ocr[j][1][-1] in ',.;:\"\'?)]' and str_eq(ocr[j][1][:-1].lower(), answer_words[j - i].lower())):
                                flag = False
                                break
                            if (not str_eq(ocr[j][1].lower(), answer_words[j - i].lower())) and (answer_words[j - i][-1] != ocr[j][1][-1] and ocr[j][1][-1] in ',.;:\"\'?)]' and str_eq(ocr[j][1][:-1].lower(), answer_words[j - i].lower())):
                                ocr[j][1] = ocr[j][1][:-1]
                if flag:
                    if not indexing:
                        link_spans.append(ocr[i: i + m])
                        link_spans[-1][0][1] = link_spans[-1][0][1][1:]
                    else:
                        link_spans.append([ocr[i: i + m], [i, i + m - 1]])
                        link_spans[-1][0][0][1] = link_spans[-1][0][0][1][1:]
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
    text = text.replace('’', '\'').replace('\u2018', '\'').replace('\u2019', '\'').replace('\u00b4', '\'').replace('\u0060', '\'').replace('“', '\"').replace('”', '\"').replace('\uff02', '\"').replace('″', '\"').replace('•', '.').replace('●', '.').replace('\u201c', '\"').replace('\u201d', '\"').replace(' ', ' ')
    for suffix_index, suffix in enumerate(['', ',', '.', ')', '%']):
        answer_ = answer.replace(' ', '') + suffix
        if normalized:
            answer_ = answer_.lower()
        answer_ = answer_.replace('’', '\'').replace('\u2018', '\'').replace('\u2019', '\'').replace('\u00b4', '\'').replace('\u0060', '\'').replace('“', '\"').replace('”', '\"').replace('\uff02', '\"').replace('″', '\"').replace('•', '.').replace('●', '.').replace('\u201c', '\"').replace('\u201d', '\"').replace(' ', ' ')
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
    _answer_ = deepcopy(answer)
    for delimiter in [['\"', '\"'], ['\'', '\''], ['(', ')'], ['[', ']']]:
        if _answer_[0] != delimiter[0] and _answer_[-1] != delimiter[1]:
            answer = delimiter[0] + _answer_ + delimiter[1]
            for suffix_index, suffix in enumerate(['', ',', '.', ')', '%']):
                answer_ = answer.replace(' ', '') + suffix
                if normalized:
                    answer_ = answer_.lower()
                answer_ = answer_.replace('’', '\'').replace('\u2018', '\'').replace('\u2019', '\'').replace('\u00b4', '\'').replace('\u0060', '\'').replace('“', '\"').replace('”', '\"').replace('\uff02', '\"').replace('″', '\"').replace('•', '.').replace('●', '.').replace('\u201c', '\"').replace('\u201d', '\"').replace(' ', ' ')
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
                            assert link_spans[-1][0][1][0] == delimiter[0], str(link_spans[-1]) + '\n' + answer
                            link_spans[-1][0][1] = link_spans[-1][0][1][1:]
                            if suffix == '':
                                assert link_spans[-1][-1][1][-1] == delimiter[1], str(delimiter) + '\n' + str(link_spans[-1]) + '\n' + answer_
                            else:
                                assert link_spans[-1][-1][1][-2:] == delimiter[1] + suffix, str(delimiter) + '\n' + str(link_spans[-1]) + '\n' + answer_
                            link_spans[-1][-1][1] = link_spans[-1][-1][1][:-1]
                            if suffix_index > 0:
                                link_spans[-1][-1][1] = link_spans[-1][-1][1][:-1]
                    if len(link_spans) > 0:
                        return link_spans
    return None


patch = {
    "645": {
        "TYPE": "answer_with_bbox",
        "image": "documents/mtyj0226_16.png",
        "questionId": 645,
        "question": "what is the average intake of sodium in US?",
        "answer": "3,000-5,000 milligrams per day",
        "bboxes": [
            [
                [
                    232,
                    504,
                    360,
                    517
                ],
                "3,000-5,000"
            ],
            [
                [
                    58,
                    517,
                    175,
                    530
                ],
                "milligrams"
            ],
            [
                [
                    181,
                    520,
                    219,
                    530
                ],
                "per"
            ],
            [
                [
                    224,
                    517,
                    265,
                    530
                ],
                "day"
            ]
        ]
    }
}


if __name__ == '__main__':
    with open('train_v1.0_withQT.json', 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    metadata = []
    cnt1, cnt2, cnt3 = 0, 0, 0
    for item in tqdm(data):
        questionId = item['questionId']
        question = formalize(item['question'])
        if question == '`what is the auth no. for inavo n rivers?': # patch
            question = 'what is the auth no. for inavo n rivers?'
        image = item['image']
        answers = list(map(formalize, item['answers']))
        for i in range(len(answers)):
            if answers[i] == '11-Dcc-95': # patch
                answers[i] = '11-Dec-95'
        assert image.startswith('documents/') and image.endswith('.png'), image
        image = image[len('documents/'):]

        if any([answer.strip().lower() in ['yes.', 'yes'] for answer in answers]):
            linked_span = [None, '[ANS_YES]']
        elif any([answer.strip().lower() in ['no.', 'no'] for answer in answers]):
            linked_span = [None, '[ANS_NO]']
        else:
            linked_span = None

        # Phase 1: Native link
        if linked_span is None:
            cache_spans = []
            for ocr_root in OCRS:
                ocr = ocr_cache[ocr_root + image]
                for answer in answers:
                    linked_span_ = link(answer, deepcopy(ocr))
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

        # Phase 2: Native link without white space
        if linked_span is None:
            for ocr_root in OCRS:
                ocr = ocr_cache[ocr_root + image]
                for answer in answers:
                    linked_span_ = link_wo_whitespace(answer, deepcopy(ocr))
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

        # Phase 3: Cross link
        if linked_span is None:
            for ocr_root in OCRS:
                ocr = ocr_cache[ocr_root + image]
                for answer in answers:
                    answer_words = answer.split(' ')
                    M = len(answer_words)
                    if M >= 4:
                        possible_link_spans = []
                        for i in range(2, M - 2):
                            answer_span1 = ' '.join(answer_words[:i]).strip()
                            answer_span2 = ' '.join(answer_words[i:]).strip()
                            linked_span1 = link(answer_span1, deepcopy(ocr), normalized=False, indexing=True)
                            linked_span2 = link(answer_span2, deepcopy(ocr), normalized=False, indexing=True)
                            if linked_span1 is not None and linked_span2 is not None:
                                N1, N2 = len(linked_span1), len(linked_span2)
                                if (M > 4 and (N1 == 1 or N2 == 1)) or (M == 4 and N1 == 1 and N2 == 1):
                                    linked_span1.sort(key=lambda x: x[1][0])
                                    linked_span2.sort(key=lambda x: x[1][0])
                                    if N1 == 1:
                                        index1 = 0
                                        start_index1, end_index1 = linked_span1[0][1]
                                        flag = False
                                        for j in range(N2):
                                            start_index2, end_index2 = linked_span2[j][1]
                                            if end_index1 < start_index2:
                                                index2 = j
                                                flag = True
                                                break
                                    else:
                                        index2 = 0
                                        start_index2, end_index2 = linked_span2[0][1]
                                        flag = False
                                        for j in range(N1):
                                            start_index1, end_index1 = linked_span1[N1 - 1 - j][1]
                                            if end_index1 < start_index2:
                                                index1 = N1 - 1 - j
                                                flag = True
                                                break
                                    if flag:
                                        linked_span_ = []
                                        for item in linked_span1[index1][0]:
                                            linked_span_.append(item)
                                        for item in linked_span2[index2][0]:
                                            linked_span_.append(item)
                                        possible_link_spans.append(linked_span_)
                        if len(possible_link_spans) == 1:
                            linked_span = [possible_link_spans[0], answer]
                            break
                if linked_span is not None:
                    break

        # Phase 4: Normalized link
        if linked_span is None:
            for ocr_root in OCRS:
                ocr = ocr_cache[ocr_root + image]
                for answer in answers:
                    linked_span_ = link(answer, deepcopy(ocr), normalized=True)
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

        # Phase 5: Normalized link without white space
        if linked_span is None:
            for ocr_root in OCRS:
                ocr = ocr_cache[ocr_root + image]
                for answer in answers:
                    linked_span_ = link_wo_whitespace(answer, deepcopy(ocr), normalized=True)
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

        # Phase 6: Normalized cross link
        if linked_span is None:
            for ocr_root in OCRS:
                ocr = ocr_cache[ocr_root + image]
                for answer in answers:
                    answer_words = answer.split(' ')
                    M = len(answer_words)
                    if M >= 4:
                        possible_link_spans = []
                        for i in range(2, M - 2):
                            answer_span1 = ' '.join(answer_words[:i]).strip()
                            answer_span2 = ' '.join(answer_words[i:]).strip()
                            linked_span1 = link(answer_span1, deepcopy(ocr), normalized=True, indexing=True)
                            linked_span2 = link(answer_span2, deepcopy(ocr), normalized=True, indexing=True)
                            if linked_span1 is not None and linked_span2 is not None:
                                N1, N2 = len(linked_span1), len(linked_span2)
                                if N1 == 1 and N2 == 1:
                                    linked_span1.sort(key=lambda x: x[1][0])
                                    linked_span2.sort(key=lambda x: x[1][0])
                                    if N1 == 1:
                                        index1 = 0
                                        start_index1, end_index1 = linked_span1[0][1]
                                        flag = False
                                        for j in range(N2):
                                            start_index2, end_index2 = linked_span2[j][1]
                                            if end_index1 < start_index2:
                                                index2 = j
                                                flag = True
                                                break
                                    else:
                                        index2 = 0
                                        start_index2, end_index2 = linked_span2[0][1]
                                        flag = False
                                        for j in range(N1):
                                            start_index1, end_index1 = linked_span1[N1 - 1 - j][1]
                                            if end_index1 < start_index2:
                                                index1 = N1 - 1 - j
                                                flag = True
                                                break
                                    if flag:
                                        linked_span_ = []
                                        for item in linked_span1[index1][0]:
                                            linked_span_.append(item)
                                        for item in linked_span2[index2][0]:
                                            linked_span_.append(item)
                                        possible_link_spans.append(linked_span_)
                        if len(possible_link_spans) == 1:
                            linked_span = [possible_link_spans[0], answer]
                            break
                if linked_span is not None:
                    break

        if str(questionId) in patch:
            metadata.append(patch[str(questionId)])
        elif linked_span is not None:
            if linked_span[0] is not None:
                metadata.append({
                    'TYPE': 'answer_with_bbox',
                    'image': 'documents/' + image,
                    'questionId': questionId,
                    'question': question,
                    'answer': linked_span[1],
                    'bboxes': linked_span[0]
                })
                cnt1 += 1
            else:
                metadata.append({
                    'TYPE': 'yes_no_answer',
                    'image': 'documents/' + image,
                    'questionId': questionId,
                    'question': question,
                    'answer': linked_span[1]
                })
                cnt2 += 1
        else: # A workaround to incorporate more training samples that cannot be linked with bounding boxes
            answers.sort(key=lambda x: len(x.split(' ')))
            answer_words = answers[0].strip().split(' ')
            metadata.append({
                'TYPE': 'answer_without_bbox',
                'image': 'documents/' + image,
                'questionId': questionId,
                'question': question,
                'answer_words': answer_words
            })
            cnt3 += 1
    print('%d / %d = %.2f%%' % (cnt1, len(data), cnt1 / len(data) * 100))
    print('%d / %d = %.2f%%' % (cnt2, len(data), cnt2 / len(data) * 100))
    print('%d / %d = %.2f%%' % (cnt3, len(data), cnt3 / len(data) * 100))
    with open('train-metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
