import argparse
import os
import logging
from datetime import datetime
import torch
import numpy as np
from components.constants import ANSWER2IDX, WORD2IDX, EMBEDDING_MATRIX
from components.datasets import VQADataset

# Macros
LOG_FILE = datetime.now().strftime("./logs/%Y-%m-%d_%H-%M-%S.log")

# Setting logging configuration
def setup_logging() -> logging.Logger:
    # Check if the logs directory exists, if not create it
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    
    # Create a logger for this folder
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Setup console and file handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(LOG_FILE)

    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def train(args, logger):
    # Setup which devices to use

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")

    # Load the dataset

    # To load the dataset, we first need to load the vocabularies and the embedding matrix for text
    word2idx_path = os.path.join(args.text_features_dir, WORD2IDX)
    answer2idx_path = os.path.join(args.text_features_dir, ANSWER2IDX)
    embedding_matrix_path = os.path.join(args.text_features_dir, EMBEDDING_MATRIX)
    if not os.path.exists(word2idx_path):
        logger.error(f"Word2idx file not found at {word2idx_path}")
        return
    if not os.path.exists(answer2idx_path):
        logger.error(f"Answer2idx file not found at {answer2idx_path}")
        return
    if not os.path.exists(embedding_matrix_path):
        logger.error(f"Embedding matrix file not found at {embedding_matrix_path}")
        return
    
    word2idx = np.load(word2idx_path, allow_pickle=True).item()
    answer2idx = np.load(answer2idx_path, allow_pickle=True).item()
    embedding_matrix = np.load(embedding_matrix_path, allow_pickle=True)

    train_dataset = VQADataset(
        questions_path=args.questions_path,
        annotations_path=args.annotations_path,
        word2idx=word2idx,
        answer2idx=answer2idx,
        feat_dir=args.image_features_dir
    )

    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train the Layer residual co-attention based netowrks for Visual question answering according the paper")

    parser.add_argument(
        "--use_cuda",
        type=bool,
        action="store_true",
        help="Use CUDA for training if available",
        default=False
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training"
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=13,
        help="Number of epochs to train"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--text_features_dir",
        type=str,
        required=True,
        help="Path to the text features directory. This directory should contain the vocabularies and the embedding matrix.",
        default="./data/VQAv2/text_features/"
    )

    parser.add_argument(
        "--image_features_dir",
        type=str,
        required=True,
        help="Path to the trainig image features directory. This directory should contain the preprocessed ResNest152 features of the dataset.",
        default="./data/VQAv2/image_features/train2014"
    )

    parser.add_argument(
        "--questions_path",
        type=str,
        required=True,
        help="Path to the questions file.",
        default="./data/VQAv2/questions/train_questions.json"
    )

    parser.add_argument(
        "--annotations_path",
        type=str,
        required=True,
        help="Path to the annotations file.",
        default="./data/VQAv2/annotations/train_annotations.json"
    )


    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Logging started. All logs for this run will be saved at {LOG_FILE}")

