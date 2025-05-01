import numpy as np

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