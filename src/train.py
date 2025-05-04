import argparse
import os
import logging
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import traceback
from tqdm import tqdm

# Custom imports
from components.constants import ANSWER2IDX, WORD2IDX, EMBEDDING_MATRIX
from components.datasets import VQAv2Dataset
from components.lrcn_scheduler import LRCNCustomScheduler
from model import LRCN

# Macros
LOG_FILE = datetime.now().strftime("./logs/%Y-%m-%d_%H-%M-%S.log")

# Setting logging configuration
def setup_logging() -> logging.Logger:
    # Check if the logs directory exists, if not create it
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    
    # Create a logger for this folder
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

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
        logger.error(f"Word2idx file not found at {word2idx_path}. Please perform the necessary preprocessing. Exiting...")
        return 1
    if not os.path.exists(answer2idx_path):
        logger.error(f"Answer2idx file not found at {answer2idx_path}. Please perform the necessary preprocessing. Exiting...")
        return 1
    if not os.path.exists(embedding_matrix_path):
        logger.error(f"Embedding matrix file not found at {embedding_matrix_path}. Please perform the necessary preprocessing. Exiting...")
        return 1
    
    logger.debug(f"Loading word2idx from {word2idx_path}")
    word2idx = np.load(word2idx_path, allow_pickle=True).item()

    logger.debug(f"Loading answer2idx from {answer2idx_path}")
    answer2idx = np.load(answer2idx_path, allow_pickle=True).item()

    # Initialize the dataset
    logger.debug(f"Loading training dataset from {args.train_questions_path}, {args.train_annotations_path} and {args.train_image_features_dir}")
    train_dataset = VQAv2Dataset(
        questions_path=args.train_questions_path,
        annotations_path=args.train_annotations_path,
        word2idx=word2idx,
        answer2idx=answer2idx,
        feat_dir=args.train_image_features_dir
    )

    logger.debug(f"Loading validation dataset from {args.val_questions_path}, {args.val_annotations_path} and {args.val_image_features_dir}")
    val_dataset = VQAv2Dataset(
        questions_path=args.val_questions_path,
        annotations_path=args.val_annotations_path,
        word2idx=word2idx,
        answer2idx=answer2idx,
        feat_dir=args.val_image_features_dir
    )

    logger.debug(f"Initializing the model")
    model = LRCN(
        architecture_type=args.variant,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_answers=len(answer2idx),
        embedding_matrix_path=embedding_matrix_path,
    ).to(device)
    criterion=nn.BCELoss()

    # Setting up the optimizer: Adam
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2.5e-5)

    # Setting up the learning rate scheduler
    scheduler = LRCNCustomScheduler(
        optimizer=optimizer,
        base_lr=args.learning_rate,
        compare_lr=2.5e-5,
        decay_factor=0.2,
        milestones=[10, 12]
    )

    # Setting up the checkpoint directory
    if not os.path.exists(args.ckpt_dir):
        logger.info(f"Checkpoint directory doesn't exist! Creating checkpoint directory at {args.ckpt_dir}")
        os.makedirs(args.ckpt_dir)

    # Log all the necessary hyperparameters and model details
    logger.info(f"[Training Info] Architecture Type: {args.variant}")
    logger.info(f"[Training Info] Batch Size: {args.batch_size}")
    logger.info(f"[Training Info] Number of Epochs: {args.num_epochs}")
    logger.info(f"[Training Info] Learning Rate: {args.learning_rate}")
    logger.info(f"[Training Info] Number of Heads: {args.num_heads}")
    logger.info(f"[Training Info] Hidden Dimension: {args.hidden_dim}")
    logger.info(f"[Training Info] Number of Layers: {args.num_layers}")
    logger.info(f"[Training Info] Checkpoint Directory: {args.ckpt_dir}")
    logger.info(f"[Training Info] Training Dataset Size: {len(train_dataset)}")
    logger.info(f"[Training Info] Validation Dataset Size: {len(val_dataset)}")

    logger.debug(f"Creating DataLoader for training and validation datasets")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    logger.info(f"Training started...")
    for epoch in range(args.num_epochs):
        model.train()
        for batch in tqdm(train_loader, total=len(train_loader)):
            # Get the data
            img_features, question_vector, answer_vector = batch

            # Move the data to the device
            img_features = img_features.to(device)
            question_vector = question_vector.to(device)
            answer_vector = answer_vector.to(device)

            # Forward pass
            outputs = model(img_features, question_vector)

            # Extract scores from output
            scores = outputs['scores']
            # Compute the loss
            loss = criterion(scores, answer_vector)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Log the training loss 
        if (epoch + 1) % args.log_every == 0:
            logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {loss.item():.4f}")

        # Save the model checkpoint
        if (epoch + 1) % args.ckpt_every == 0:
            checkpoint_path = os.path.join(args.ckpt_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model checkpoint saved at {checkpoint_path}")
        
        
        # Validation step
        if (epoch + 1) % args.validate_every == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in val_loader:
                    img_features, question_vector, answer_vector = val_batch
                    img_features = img_features.to(device)
                    question_vector = question_vector.to(device)
                    answer_vector = answer_vector.to(device)

                    outputs = model(img_features, question_vector)
                    loss = F.binary_cross_entropy(outputs, answer_vector)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            logger.info(f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {val_loss:.4f}")
    return 0

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train the Layer residual co-attention based netowrks for Visual question answering according the paper")

    parser.add_argument(
        "--use_cuda",
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
        help="Path to the text features directory. This directory should contain the vocabularies and the embedding matrix.",
        default="./data/VQAv2/text_features/"
    )

    parser.add_argument(
        "--train_image_features_dir",
        type=str,
        help="Path to the trainig image features directory. This directory should contain the preprocessed ResNest152 features of the dataset.",
        default="./data/VQAv2/image_features/train2014"
    )

    parser.add_argument(
        "--train_questions_path",
        type=str,
        help="Path to the questions file.",
        default="./data/VQAv2/questions/train_questions.json"
    )

    parser.add_argument(
        "--train_annotations_path",
        type=str,
        help="Path to the annotations file.",
        default="./data/VQAv2/annotations/train_annotations.json"
    )

    parser.add_argument(
        "--val_image_features_dir",
        type=str,
        help="Path to the trainig image features directory. This directory should contain the preprocessed ResNest152 features of the dataset.",
        default="./data/VQAv2/image_features/train2014"
    )

    parser.add_argument(
        "--val_questions_path",
        type=str,
        help="Path to the questions file.",
        default="./data/VQAv2/questions/train_questions.json"
    )

    parser.add_argument(
        "--val_annotations_path",
        type=str,
        help="Path to the annotations file.",
        default="./data/VQAv2/annotations/train_annotations.json"
    )

    parser.add_argument(
        "--variant",
        type=str,
        choices=["pure_stacking", "co_stacking", "encoder_decoder"],
        help="Variant of the feature encoding model to use. Options are: pure_stacking, co_stacking, encoder_decoder",
        required=True
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension for the model"
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="Number of layers in the model"
    )

    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./checkpoints/",
        help="Directory to save the checkpoints"
    )

    parser.add_argument(
        "--ckpt_every",
        type=int,
        default=1,
        help="Save checkpoint every n epochs"
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="Log every n epochs"
    )

    parser.add_argument(
        "--validate_every",
        type=int,
        default=1,
        help="Validate every n epochs"
    )

    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Logging started. All logs for this run will be saved at {LOG_FILE}")

    try: 
        exit_code = train(args, logger)
    except Exception as e:
        logger.error(f"Unexpected exception occured: {e}")
        error_msg = traceback.format_exc()
        logger.error(error_msg)
        exit_code = 1
    if exit_code != 0:
        logger.error(f"Training failed. Please check the logs for more details.")
        exit(exit_code)
    else:
        logger.info("Training completed successfully!")
