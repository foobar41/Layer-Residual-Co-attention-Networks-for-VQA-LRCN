import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

def load_glove_embeddings(glove_path):
    glove_dict = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            glove_dict[word] = vector
    return glove_dict

def build_vocab_and_answers(train_questions, train_annotations, glove_dict, top_answers=3129):
    word_counts = {}
    for q in train_questions:
        words = q['question'].lower().split()
        for w in words:
            if w in glove_dict:
                word_counts[w] = word_counts.get(w,0) + 1

    vocab = ['<pad>', '<unk>'] + [w for w in word_counts]
    word2idx = {w:i for i,w in enumerate(vocab)}

    ## building ans vocab
    ans_counts = {}
    for ann in train_annotations:
        for ans in ann['answers']:
            a = ans['answer']
            ans_counts[a] = ans_counts.get(a, 0) + 1

    top_ans = sorted(ans_counts.items(), key=lambda x: x[1], reverse=True)[:top_answers]
    answer2idx = {ans:i for i, (ans,_) in enumerate(top_ans)}

    return word2idx, answer2idx

def build_embedding_matrix(word2idx, glove_dict, embedding_dim=300):
    """
    Build an embedding matrix for all words in word2idx using glove_dict.
    """
    matrix = np.zeros((len(word2idx), embedding_dim), dtype=np.float32)
    for word, idx in word2idx.items():
        if word in glove_dict:
            matrix[idx] = glove_dict[word]
        else:
            # For <pad>, <unk>, or OOV words, use zeros or random
            matrix[idx] = np.zeros(embedding_dim, dtype=np.float32)
    return matrix


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
        return img_feat, q_indices, ans_vector

## IMPORTANT STUFF TO INCLUDE IN MAIN MODEL FOR FEATURE CREATION
# class LRCN(torch.nn.Module):
#     def __init__(self, word2idx, glove_weights):
#         super().__init__()

#         ## Question encoder
#         self.embed = torch.nn.Embedding.from_pretrained(
#             torch.stack([torch.from_numpy(glove_weights[w])
#             if w in glove_weights else torch.randn(300)
#             for w in word2idx]), freeze=False
#         )

#         self.lstm = torch.nn.LSTM(300, 512, batch_first=True)

#     def forward(self, img_feats, q_indices):

#         ## Question features [B, 14, 300] -> [B, 14, 512]
#         q_emb = self.embed(q_indices)
#         q_feats, _ = self.lstm(q_emb)

if __name__=="__main__":
    ## FILE PATHS
    WORD2IDX_PATH = "word2idx.npy"
    ANSWER2IDX_PATH = "answer2idx.npy"
    GLOVE_PATH = "glove.42B.300d.txt"
    EMBEDDING_MATRIX_PATH = "glove_embedding_matrix.npy"

    TRAINING_ANNOTATIONS = "./data/VQAv2/annotations/v2_mscoco_train2014_annotations.json"
    TRAINING_QUESTIONS = "./data/VQAv2/questions/v2_OpenEnded_mscoco_train2014_questions.json"
    TRAINING_IMAGE_FEATURES_PATH = "./data/VQAv2/images/img_features/train2014"

    VAL_ANNOTATIONS = "./data/VQAv2/annotations/v2_mscoco_val2014_annotations.json"
    VAL_QUESTIONS = "./data/VQAv2/questions/v2_OpenEnded_mscoco_val2014_questions.json"
    VAL_IMAGE_FEATURES_PATH = "./data/VQAv2/images/img_features/val2014"

    # 3. Build vocabularies (chcks if word2idx and answer2idx numpy files already exist)
    if os.path.exists(WORD2IDX_PATH) and os.path.exists(ANSWER2IDX_PATH):
        print("Loading existing vocab files...")
        word2idx = np.load(WORD2IDX_PATH, allow_pickle=True).item()
        answer2idx = np.load(ANSWER2IDX_PATH, allow_pickle=True).item()
    else:
        print("Building new vocab files...")
        # 1. Load GloVe embeddings
        glove_dict = load_glove_embeddings(GLOVE_PATH)

        # 2. Load training data
        with open(TRAINING_ANNOTATIONS) as f:
            train_anns = json.load(f)["annotations"]
        with open(TRAINING_QUESTIONS) as f:
            train_questions = json.load(f)["questions"]
        word2idx, answer2idx = build_vocab_and_answers(
            train_questions, train_anns, glove_dict
        )
         # 4. Save vocabularies (for later use)
        np.save(WORD2IDX_PATH, word2idx)
        np.save(ANSWER2IDX_PATH, answer2idx)

    print('W2I: ', len(word2idx))   ## 10723
    print('A2I: ', len(answer2idx)) ## 3129

    # 5. Initialize dataset
    train_dataset = VQAv2Dataset(
        questions_path=TRAINING_QUESTIONS,
        annotations_path=TRAINING_ANNOTATIONS,
        word2idx=word2idx,
        answer2idx=answer2idx,
        feat_dir=TRAINING_IMAGE_FEATURES_PATH
    )

    val_dataset = VQAv2Dataset(
        questions_path=VAL_QUESTIONS,
        annotations_path=VAL_ANNOTATIONS,
        word2idx=word2idx,
        answer2idx=answer2idx,
        feat_dir=VAL_IMAGE_FEATURES_PATH
    )

    # Only build embedding matrix if it doesn't exist
    # Embedding matrix will be loaded in the model to assign embeddings to the textual features.
    # Added loading this matrix to model and not preprocessing in case we want to unfreeze (train) the embeddings, paper is not clear on whether they are trained or not
    if not os.path.exists(EMBEDDING_MATRIX_PATH):
        print("Building GloVe embedding matrix...")
        # Load GloVe if not already loaded
        if 'glove_dict' not in locals():
            glove_dict = load_glove_embeddings(GLOVE_PATH)
        embedding_matrix = build_embedding_matrix(word2idx, glove_dict, embedding_dim=300)
        np.save(EMBEDDING_MATRIX_PATH, embedding_matrix)
    # else:
    #     print("Loading existing GloVe embedding matrix...")
    #     embedding_matrix = np.load(EMBEDDING_MATRIX_PATH)

    # # 6. Verify sample
    # img_feat, q_indices, ans_vec = train_dataset[0]
    # print(f"Image features: {img_feat.shape}")  # Should be torch.Size([64, 512])
    # print(f"Question indices: {q_indices.shape}")  # torch.Size([14])
    # print(f"Answer vector: {ans_vec.shape}")  # torch.Size([3129])
