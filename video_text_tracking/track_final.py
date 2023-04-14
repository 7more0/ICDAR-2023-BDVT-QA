import numpy as np
import os, sys
import json
from scipy.optimize import linear_sum_assignment
from copy import deepcopy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

import warnings

warnings.filterwarnings("ignore")
from Base import BBox, Frame, Video, Text, TextSimilarity, PosSimilarity, OCRConvertor


class Hungarian:
    def __init__(self, alpha=0.7, sim_threshold=5):
        self.text_sim = TextSimilarity()
        self.pos_sim = PosSimilarity()

        self.alpha = alpha
        self.sim_threshold = sim_threshold

    def __call__(self, frame1, frame2):
        cost, k1, k2 = self._cost_(frame1, frame2)

        assignment = linear_sum_assignment(cost)
        assignment = np.array(assignment).T

        # debug
        # for f1_k, f2_k in assignment:
        #   print('k1:{}    k2:{}\n'.format(frame1.bboxes[k1[f1_k]].trans,frame2.bboxes[k2[f2_k]].trans))

        # cost_check
        marker = np.zeros((len(k1), len(k2)))
        confirm_assignment = []
        for pair in assignment:
            if cost[pair[0], pair[1]] <= self.sim_threshold:
                marker[pair[0], pair[1]] = 1
                confirm_assignment.append([k1[pair[0]], k2[pair[1]]])
        f1_summary = np.sum(marker, axis=1)
        f2_summary = np.sum(marker, axis=0)

        f1_loose = [k1[i] for i in np.where(f1_summary == 0)[0]]
        f2_loose = [k2[i] for i in np.where(f2_summary == 0)[0]]
        # print(f1_summary)
        # print(f1_loose)

        return confirm_assignment, f1_loose, f2_loose

    def _cost_(self, f1, f2):
        f1_keys = list(f1.bboxes.keys())
        f2_keys = list(f2.bboxes.keys())

        cost_matrix = np.zeros((len(f1_keys), len(f2_keys)))
        for f1_idx, f1_k in enumerate(f1_keys):
            for f2_idx, f2_k in enumerate(f2_keys):
                cost = self.alpha * self.text_sim(f1.bboxes[f1_k].trans, f2.bboxes[f2_k].trans)
                +(1 - self.alpha) * self.pos_sim(f1.bboxes[f1_k].points, f2.bboxes[f2_k].points)
                cost_matrix[f1_idx, f2_idx] = cost

        return cost_matrix, f1_keys, f2_keys


class DataWashing:
    def __init__(self):
        self.flag = ''

    def __call__(self, video, **kwargs):
        self.video = video
        frames = video.frames
        # initiate tracker
        # self.texts = []
        # self.box_to_frame = {}
        self.register_box = []

        # sample_keys = self._sampling_(len(list(frames.keys())))
        # sample_keys = sorted(sample_keys)
        sample_keys = list(frames.keys())

        for f_key, f in frames.items():
            bx_keys = list(f.bboxes.keys())
            for bx in bx_keys:
                if bx in self.register_box:
                    frames[f_key].bboxes.pop(bx)
                else:
                    self.register_box.append(bx)

        video.frames = frames
        return video


def KMean3(input):
    if len(input) == 0:
        return ""
    else:

        vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), stop_words=None)
        vectors = vectorizer.fit_transform(input).toarray()

        kmeans = KMeans(n_clusters=3, random_state=0).fit(vectors)  # changed number of clusters to 2

        min_distance = float('inf')
        min_cluster_index = None

        for i in range(kmeans.n_clusters):
            distances = []
            for j, vector in enumerate(vectors):
                if kmeans.labels_[j] == i:
                    distance = np.linalg.norm(vector - kmeans.cluster_centers_[i])
                    distances.append((j, distance))

                if len(distances) > 0:
                    closest_text_index, closest_distance = min(distances, key=lambda x: x[1])

                    if closest_distance < min_distance:
                        min_distance = closest_distance
                        min_cluster_index = i

                        result_text = input[closest_text_index]
        return result_text


import re


class TextTracker:
    def __init__(self, alpha=0.7, s_rate=1, sim_threshold=0.6):
        self.hug = Hungarian(alpha=alpha)
        self.s_interal = int(1 / s_rate)
        self.sim_threshold = sim_threshold
        self.full_parag = []

        self.text_trans = TextTrans()

    def __call__(self, video):
        self.full_parag = []
        self.box_to_frame = {}
        self.tracking(video)
        # print(self.box_to_frame)

        # text completion
        for t_id, txt in enumerate(self.texts):
            self.texts[t_id].trans = self.prompt(txt)
        self.texts, ocr_out = self.text_trans(video, self.texts, self.box_to_frame)

        self.full_parag = [txt.trans for txt in self.texts]
        full_parag = ",".join('%s' % id for id in self.full_parag)
        full_parag = re.sub(r',+', ',', full_parag)
        full_parag = re.sub(r'^,+|,+$|[\x00-\x1f\x7f-\xff]', '', full_parag)
        return full_parag, ocr_out
        # return full_parag, {}

    def tracking(self, video: Video):
        self.video = video
        frames = video.frames
        # initiate tracker
        self.texts = []
        self.box_to_frame = {}

        # sample_keys = self._sampling_(len(list(frames.keys())))
        # sample_keys = sorted(sample_keys)
        sample_keys = list(frames.keys())
        # print(sample_keys)
        self.tracking_txts = {}
        self.text_id = 0
        self.track_board = {}
        # initiate texts for frame_0
        for box in frames[sample_keys[0]].bboxes.keys():
            self.box_to_frame[box] = sample_keys[0]
            self.tracking_txts[self.text_id] = Text(self.text_id, box)
            self.track_board[box] = self.text_id
            self.text_id += 1

        for s_f_idx in range(1, len(sample_keys), 1):
            f1, f2 = frames[sample_keys[s_f_idx - 1]], frames[sample_keys[s_f_idx]]

            confirm_assignment, f1_loose, f2_loose = self.hug(f1, f2)

            for box_f1, box_f2 in confirm_assignment:
                self.box_to_frame[box_f1] = f1.id
                self.box_to_frame[box_f2] = f2.id
                self._update_board(box_f1, box_next=box_f2, tp='next')
            for box_f1 in f1_loose:
                self.box_to_frame[box_f1] = f1.id
                self._update_board(box_f1, tp='end')
            for box_f2 in f2_loose:
                self.box_to_frame[box_f2] = f2.id
                self._update_board(box_f2, tp='start')
            # print(self.track_board)

        for box_f2 in list(self.track_board.keys()):
            self.box_to_frame[box_f2] = f2.id
            self._update_board(box_f2, 'end')

        assert len(self.track_board) == 0

    def _update_board(self, box, tp='next', box_next=''):
        if tp == 'end':
            self.texts.append(self.tracking_txts[self.track_board[box]])
            self.tracking_txts.pop(self.track_board[box])
            self.track_board.pop(box)

        elif tp == 'start':
            self.tracking_txts[self.text_id] = Text(self.text_id, box)
            self.track_board[box] = self.text_id
            self.text_id += 1

        elif tp == 'next':
            self.tracking_txts[self.track_board[box]].box_id.append(box_next)
            self.track_board[box_next] = self.track_board[box]
            self.track_board.pop(box)

    def _sampling_(self, frames_len):
        self.frame_pattern = 'frame_{:d}'
        return [self.frame_pattern.format(i) for i in range(frames_len) if i % self.s_interal == 0]

    def prompt(self, txt: Text):
        box_texts = []
        box_texts1 = []

        for bid in txt.box_id:
            bbox = self.video.frames[self.box_to_frame[bid]].bboxes[bid]
            text = bbox.trans
            if text is not None:
                box_texts.append(str(text))
            else:
                box_texts.append('')

        box_texts1 = box_texts
        box_texts1 = [elem for elem in box_texts1 if elem != '#####']
        # print(box_texts1)

        if len(box_texts1) == 0:
            return ''
        else:
            if len(set(box_texts1)) == 1:
                return box_texts1[0]
            else:
                if len(box_texts1) <= 3:
                    counter = Counter(box_texts1)
                    most_common_str = counter.most_common(1)[0][0]
                    return most_common_str

                else:
                    return KMean3(box_texts1)


class Processor:
    def __init__(self):
        self.tracker = TextTracker()
        self.data_washer = DataWashing()

    def __call__(self, *args, **kwargs):
        vid_list = os.listdir(kwargs['root'])
        collector = {}
        failed = []
        for vid_idx, vid in enumerate(vid_list):
            try:
                ann_data = Video(os.path.join(kwargs['root'], vid), convert=True)
                ann_data = self.data_washer(ann_data)
                res, ocr_out = self.tracker(ann_data)
                # print(res)
                collector[vid] = res
                json.dump(ocr_out, open(os.path.join(kwargs['write_ocr'], vid),
                                        'w', encoding='utf-8'), indent=4)
            except Exception as e:
                print('{}:{}'.format(vid, e))
                failed.append(vid)

        json.dump(collector, open(kwargs['write'], 'w+', encoding='utf-8'), indent=4)

        return True


class TextTrans:
    def __init__(self):
        self.text_transform = lambda x: x
        self.format_transform = OCRConvertor(flag='ann2ocr')

    def __call__(self, video, vid_track, box2frame):
        replaced_video = deepcopy(video.frames)

        # bbox text replace
        # assign_text_id = 2023
        for t_k, txt in enumerate(vid_track):
            new_text = self.text_transform(txt.trans)
            vid_track[t_k].trans = new_text
            # vid_track[t_k].text_id = assign_text_id
            # assign_text_id+=1
            for box in txt.box_id:
                replaced_video[box2frame[box]].bboxes[box].trans = new_text

        # video.frames = replaced_video
        ocr_out = self.format_transform(replaced_video)

        return vid_track, ocr_out


if __name__ == '__main__':
    # root = './data/train_annotations'
    root = './data/ocr_v3'
    # ann_loader = Video('./data/train_annotations/vid_6371842732441606.json')
    # # debug
    # t_tracker = TextTracker()
    # res = t_tracker(ann_loader)

    pro = Processor()
    pro(root=root,
        write='./output/tracking_out_ocr_v3.json',
        write_ocr='./output/recon_ocr')

    pass
