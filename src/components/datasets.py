from torch.utils.data import Dataset
import json
import torch
import os
import numpy as np
import cv2

class VQAv2Dataset(Dataset):
    def __init__(self, questions_path, annotations_path, word2idx, answer2idx, feat_dir):

        ## Loading questions and answers
        with open(questions_path) as f:
            self.questions = json.load(f)['questions']
        with open(annotations_path) as f:
            self.annotations = {ann['question_id']: ann for ann in json.load(f)['annotations']}

        self.word2idx = word2idx
        self.answer2idx = answer2idx
        self.feat_dir = feat_dir
        self.split_name = os.path.basename(feat_dir)

        ## Getting question features (truncate at 14 words, or pad to 14 words)
        self.q_indices = []
        for q in self.questions:
            tokens = q['question'].lower().split()[:14] ## truncating
            indices = [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]
            indices += [self.word2idx['<pad>']] * (14 - len(indices))   ## padding
            self.q_indices.append(indices)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        entry = self.questions[idx]
        ann = self.annotations.get(entry['question_id'], {'answers': []})   ## to store answers
        question_id = str(entry['question_id'])

        ## Loading precomputed image features from .npy files. Shape: [64, 512]
        image_id = entry['image_id']
        # Construct filename using COCO naming convention
        filename = f"COCO_{self.split_name}_{image_id:012d}.npy"
        feat_path = os.path.join(self.feat_dir, filename)
        img_feat = np.load(feat_path)
        img_feat = torch.from_numpy(img_feat).float()

        ## Question features [14]
        q_indices = torch.tensor(self.q_indices[idx], dtype=torch.long)

        ## Answer vectors [3129]
        ans_vector = torch.zeros(len(self.answer2idx), dtype=torch.float)
        if ann["answers"]:
            counts = {}
            for ans_obj in ann['answers']:
                ans = ans_obj['answer']
                if ans in self.answer2idx:
                    counts[ans] = counts.get(ans, 0) + 1
            for ans, cnt in counts.items():
                ans_vector[self.answer2idx[ans]] = min(cnt/3, 1.0)  ## soft scores with min(count/3, 1) normalization
        return img_feat, q_indices, ans_vector, question_id

class VQAv2VizDataset(VQAv2Dataset):
    def __init__(self, questions_path, annotations_path, word2idx, answer2idx, feat_dir, image_dir):

        self.image_dir = image_dir

        ## Loading questions and answers
        with open(questions_path) as f:
            self.questions = json.load(f)['questions']
        with open(annotations_path) as f:
            self.annotations = {ann['question_id']: ann for ann in json.load(f)['annotations']}

        self.word2idx = word2idx
        self.answer2idx = answer2idx
        self.feat_dir = feat_dir
        self.split_name = os.path.basename(feat_dir)

        ## Getting question features (truncate at 14 words, or pad to 14 words)
        self.q_indices = []
        for q in self.questions:
            tokens = q['question'].lower().split()[:14] ## truncating
            indices = [self.word2idx.get(t, self.word2idx['<unk>']) for t in tokens]
            indices += [self.word2idx['<pad>']] * (14 - len(indices))   ## padding
            self.q_indices.append(indices)

    def __getitem__(self, idx):
        entry = self.questions[idx]
        ann = self.annotations.get(entry['question_id'], {'answers': []})   ## to store answers
        question_id = str(entry['question_id'])

        ## Loading precomputed image features from .npy files. Shape: [64, 512]
        image_id = entry['image_id']
        # Construct filename using COCO naming convention
        filename_feat = f"COCO_{self.split_name}_{image_id:012d}.npy"
        feat_path = os.path.join(self.feat_dir, filename_feat)
        img_feat = np.load(feat_path)
        img_feat = torch.from_numpy(img_feat).float()

        filename_raw = f"COCO_{self.split_name}_{image_id:012d}.jpg"
        raw_path = os.path.join(self.image_dir, filename_raw)
        raw = cv2.imread(raw_path)
        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


        ## Question features [14]
        q_indices = torch.tensor(self.q_indices[idx], dtype=torch.long)

        ## Answer vectors [3129]
        ans_vector = torch.zeros(len(self.answer2idx), dtype=torch.float)
        if ann["answers"]:
            counts = {}
            for ans_obj in ann['answers']:
                ans = ans_obj['answer']
                if ans in self.answer2idx:
                    counts[ans] = counts.get(ans, 0) + 1
            for ans, cnt in counts.items():
                ans_vector[self.answer2idx[ans]] = min(cnt/3, 1.0)  ## soft scores with min(count/3, 1) normalization
        return img_feat, q_indices, ans_vector, question_id, raw_rgb