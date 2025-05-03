import numpy as np
import os
from preprocessing.text_feature_extraction import load_glove_embeddings, build_vocab_and_answers, build_embedding_matrix
import json
import argparse
from components.constants import WORD2IDX, ANSWER2IDX, EMBEDDING_MATRIX


def preprocess_text_build_vocab(glove_path, text_feature_output_dir, questions_path, annontations_path):

    word2idx_path = os.path.join(text_feature_output_dir, WORD2IDX)
    answer2idx_path = os.path.join(text_feature_output_dir, ANSWER2IDX)
    if os.path.exists(word2idx_path) and os.path.exists(answer2idx_path):
        print("GloVe embeddings and vocabularies already exist. Skipping...")
        return
    
    print("Building GloVe vocab files...")
    # Loading the original GloVe Embeddings
    glove_dict = load_glove_embeddings(glove_path)

    # Loading the input datasets. Assuming the dataset follows the VQAv2 json structure
    question_dataset = json.load(open(questions_path, 'r'))['questions']
    annotation_dataset = json.load(open(annontations_path, 'r'))['annotations']

    # Building the vocabularies
    word2idx, answer2idx = build_vocab_and_answers(question_dataset, annotation_dataset, glove_dict)

    # Saving the vocabularies
    np.save(word2idx_path, word2idx)
    np.save(answer2idx_path, answer2idx)

    print(f"Word2idx saved at: {word2idx_path}")
    print(f"Answer2idx saved at: {answer2idx_path}")
    return


def preprocess_text_build_embedding_matrix(glove_path, text_feature_output_dir):

    embedding_matrix_path = os.path.join(text_feature_output_dir, EMBEDDING_MATRIX)
    if os.path.exists(embedding_matrix_path):
        print("Embedding matrix already exists. Skipping...")
        return

    print("Building GloVe embedding matrix...")
    # Loading the original GloVe Embeddings
    glove_dict = load_glove_embeddings(glove_path)

    # Loading the vocabularies
    word2idx_path = os.path.join(text_feature_output_dir, WORD2IDX)
    if not os.path.exists(word2idx_path):
        raise FileNotFoundError(f"word2idx.npy not found in {text_feature_output_dir}. Please build the vocabularies first.")
    word2idx = np.load(word2idx, allow_pickle=True).item()

    # Building the embedding matrix
    embedding_matrix = build_embedding_matrix(word2idx, glove_dict, embedding_dim=300)

    # Saving the embedding matrix
    np.save(embedding_matrix_path, embedding_matrix)

    print(f"Embedding matrix saved at: {embedding_matrix_path}")
    return


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Preprocess to build vocabularies and embeddings using GloVe visual question answering dataset.")

    parser.add_argument(
        "--glove_path",
        type=str,
        required=False,
        help="Path to the original GloVe embeddings file.",
        default="./data/glove.42B.300d.txt"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Directory to save the processed files. This includes the vocabularies and the embedding matrix.",
        default="./data/VQAv2/text_features/"
    )

    parser.add_argument(
        "--questions_file_path",
        type=str,
        required=False,
        help="Path to questions data file. Should be a json file following the VQAv2 dataset format.",
        default="./data/VQAv2/questions/v2_OpenEnded_mscoco_train2014_questions.json"
    )

    parser.add_argument(
        "--annotations_file_path",
        type=str,
        required=False,
        help="Path to annotations data file. Should be a json file following the VQAv2 dataset format.",
        default="./data/VQAv2/annotations/v2_mscoco_train2014_annotations.json"
    )


    args = parser.parse_args()


    preprocess_text_build_vocab(
        args.glove_path,
        args.output_dir,
        args.questions_file_path,
        args.annotations_file_path
    )

    preprocess_text_build_embedding_matrix(
        args.glove_path,
        args.output_dir
    )

    print("Preprocessing completed!")

## FILE PATHS

# TRAINING_ANNOTATIONS = "./data/VQAv2/annotations/v2_mscoco_train2014_annotations.json"
# TRAINING_QUESTIONS = "./data/VQAv2/questions/v2_OpenEnded_mscoco_train2014_questions.json"
# TRAINING_IMAGE_FEATURES_PATH = "./data/VQAv2/images/img_features/train2014"

# VAL_ANNOTATIONS = "./data/VQAv2/annotations/v2_mscoco_val2014_annotations.json"
# VAL_QUESTIONS = "./data/VQAv2/questions/v2_OpenEnded_mscoco_val2014_questions.json"
# VAL_IMAGE_FEATURES_PATH = "./data/VQAv2/images/img_features/val2014"