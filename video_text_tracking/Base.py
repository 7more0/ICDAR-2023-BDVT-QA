import numpy as np
import os, sys
import json
import Levenshtein
from scipy.optimize import linear_sum_assignment


class BBox:
    def __init__(self, box_id, text_id, points, trans, confidence):
        self.id = box_id
        self.text_id = text_id
        self.points = points
        self.trans = trans
        self.confidence = float(confidence)


class Text:
    def __init__(self, text_id, box_id, trans=''):
        self.id = text_id
        self.box_id = [box_id]
        # self.trans = ''
        self.trans = trans


class Frame:
    def __init__(self, f_id, f_data):
        self.id = f_id
        self.bboxes = {}
        for bx in f_data:
            self.bboxes[bx['box_id']] = BBox(bx['box_id'], bx['text_id'], bx['points'], bx['transcription'],
                                             bx['confidence'])


class Video:
    def __init__(self, f, convert=False):

        self.data = json.load(open(f, 'r', encoding='utf-8'))
        if convert:
            self.convertor = OCRConvertor()
            self.data = self.convertor(self.data)

        self.frame_pattern = 'frame_%d'
        self.frames = {}

        self.texts = {}
        self.full_prag = ''

        self._parse_()

    def _parse_(self):
        base_keys = ['name', 'id', 'height', 'width', 'fps']
        for base_k in base_keys:
            setattr(self, base_k, self.data[base_k])

        self.frames = {f_id: Frame(f_id, f_data) for f_id, f_data in self.data['anns'].items()}


class TextSimilarity:
    def __init__(self, method='edit'):
        self.method = method

    def __call__(self, txt1, txt2):
        return Levenshtein.distance(txt1, txt2)


class PosSimilarity:
    def __init__(self, method='euclidean'):
        pass

    def __call__(self, txt1, txt2):
        return np.linalg.norm(np.array(txt1[-1]) - np.array(txt2[-1]))


class OCRConvertor:
    def __init__(self, flag='ocr2ann'):
        self.flag = flag

    def __call__(self, data):
        if self.flag == 'ocr2ann':
            new_data = {}
            base_keys = ['name', 'id', 'height', 'width', 'fps']
            for base_k in base_keys:
                new_data[base_k] = ''
            frame_data = {}
            id_count = 2023
            for f, f_d in data.items():
                frame_data[f] = []
                for box in f_d:
                    frame_data[f].append({'box_id': id_count,
                                          'text_id': box['text_id'],
                                          'points': [box['points'][i: i + 2] for i in range(0, 8, 2)],
                                          'transcription': box['transcription'],
                                          'confidence': box['confidence'],
                                          })
                    id_count += 1
            new_data['anns'] = frame_data
            return new_data
        elif self.flag == 'ann2ocr':
            # data = data.frames
            frame_data = {}
            for f, f_d in data.items():
                frame_data[f] = [{'text_id': box.text_id,
                                  'points': [box.points[i][j] for i in range(len(box.points)) for j in range(2)],
                                  'transcription': box.trans,
                                  'confidence': box.confidence} for b_key, box in f_d.bboxes.items()]

            return frame_data
