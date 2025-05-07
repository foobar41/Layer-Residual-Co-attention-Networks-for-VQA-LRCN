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
import matplotlib.pyplot as plt
import cv2

import torchvision.transforms as transforms

# Custom imports
from components.constants import ANSWER2IDX, WORD2IDX, EMBEDDING_MATRIX
from components.datasets import VQAv2Dataset, VQAv2VizDataset
from components.lrcn_scheduler import LRCNCustomScheduler
from model import LRCN

# Macros
LOG_FILE = datetime.now().strftime("./logs/viz_%Y-%m-%d_%H-%M-%S.log")

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

def viz(args, logger):
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
    logger.debug(f"Loading validation dataset from {args.val_questions_path}, {args.val_annotations_path} and {args.val_image_features_dir}")
    val_dataset = VQAv2VizDataset(
        questions_path=args.val_questions_path,
        annotations_path=args.val_annotations_path,
        word2idx=word2idx,
        answer2idx=answer2idx,
        feat_dir=args.val_image_features_dir,
        image_dir=args.val_image_dir
    )

    logger.debug(f"Initializing and loading the model")
    model = LRCN(
        architecture_type=args.variant,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_answers=len(answer2idx),
        embedding_matrix_path=embedding_matrix_path,
    ).to(device)

    model_path = args.model_path
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=os.cpu_count()//2
    )
   
    output_hidden_states = args.output_hidden_states
    output_attn_weights = args.output_attn_weights
        
    model.eval()
    all_outputs = []
    cnt = 0
    with torch.no_grad():
        for val_batch in tqdm(val_loader, total=len(val_loader)):
            cnt += 1
            img_features, question_vector, answer_vector, q_id, raw = val_batch


            # print("debug-shapes:", img_features.shape, question_vector.shape, answer_vector.shape, q_id, raw.shape)
            img_features = img_features.to(device)
            question_vector = question_vector.to(device)
            answer_vector = answer_vector.to(device)

            outputs = model(img_features=img_features, 
                            text_features=question_vector, 
                            output_hidden_states=output_hidden_states, 
                            output_attn_weights=output_attn_weights)

            q_id = int(q_id[0])
            curr_path = f"/projectnb/cs585bp/projects/Catastrophe/Vision-Language-For-The-Blind/src/plots/{q_id}"
            os.makedirs(curr_path, exist_ok=True)

            overlay(args, outputs, val_batch, curr_path, logger)

            if cnt==100: break

    return outputs

def overlay(args, outputs, batch, path, logger):
    img_features, question_vector, answer_vector, q_id, raw = batch
    q_id = q_id[0]
    question_vector = question_vector.numpy()[0]
    answer_vector = answer_vector.numpy()[0]
    print('debug-overlay (question-vector):', question_vector)

    word2idx_path = os.path.join(args.text_features_dir, WORD2IDX)
    word2idx = np.load(word2idx_path, allow_pickle=True).item()
    id2word = {str(v): k for k, v in word2idx.items()}

    q = ''
    for _id in question_vector:
        q = q + ' ' + id2word[str(_id)]
    print('debug-overlay (question):', q)

    answer2idx_path = os.path.join(args.text_features_dir, ANSWER2IDX)
    ans2idx = np.load(answer2idx_path, allow_pickle=True).item()
    id2ans = {str(v): k for k, v in ans2idx.items()}

    a = id2ans[str(np.argmax(answer_vector))]
    print('debug-overlay (answer):', a)

    raw = raw[0].float() / 255.0
    raw = raw.permute(2, 0, 1)
    transform = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
    raw = transform(raw)
    raw = raw.permute(1, 2, 0).numpy()

    # Auto‑detect left/right bars by looking for nearly‑black columns:
    gray = cv2.cvtColor((raw*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # a column is “content” if any pixel > black_threshold
    black_thresh = 10
    content_cols = np.any(gray > black_thresh, axis=0)
    left, right = np.where(content_cols)[0][[0, -1]]

    # Crop to [ :, left:right+1 ]
    raw_crop = raw[:, left:right+1, :]
    Hc, Wc = raw_crop.shape[:2]

    attn_map = outputs['attn_weights']['image_guided'].cpu().numpy()[0]
    heat_all = attn_map # image region importances
    overlays = []
    for i, (heat, _id, question) in enumerate(zip(heat_all.T, question_vector, q.split())):
        if _id==0: break
        heat_grid = heat.reshape(8, 8)

        # h_img, w_img = 512, 640
        h_img, w_img = Hc, Wc
        heat_full = cv2.resize(
            heat_grid, 
            (w_img, h_img),            # note: cv2.resize takes (width, height)
            interpolation=cv2.INTER_CUBIC
        )

        heat_full = cv2.GaussianBlur(heat_full, ksize=(11, 11), sigmaX=15, sigmaY=15)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(raw_crop)
        # ax1.set_title("Cropped Content")
        ax1.axis("off")

        ax2.imshow(raw_crop)
        ax2.imshow(heat_full, cmap="jet", alpha=0.5, interpolation="bilinear")
        ax2.axis("off")

        # 1) place your title on the figure, not on an axes
        fig.suptitle(f"q_id: {q_id}, word: {question}, ans: {a}", fontsize=16)

        # 2) leave ~5% of the top margin for the suptitle
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # 3) save with bbox_inches='tight' so nothing gets cut off
        if question == '<unk>': question = 'eof'
        fig.savefig(os.path.join(path, f"{i}-{question}.jpg"), dpi=1200, bbox_inches="tight")
        plt.close(fig)


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
        "--text_features_dir",
        type=str,
        help="Path to the text features directory. This directory should contain the vocabularies and the embedding matrix.",
        default="./data/VQAv2/text_features/"
    )

    parser.add_argument(
        "--val_image_dir",
        type=str,
        help="Path to the raw validation image directory. This directory should contain the preprocessed ResNest152 features of the dataset.",
        default="./data/VQAv2/images/val/val2014"
    )

    parser.add_argument(
        "--val_image_features_dir",
        type=str,
        help="Path to the validation image features directory. This directory should contain the preprocessed ResNest152 features of the dataset.",
        default="./data/VQAv2/image_features/val2014"
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
        "--model_path",
        type=str,
        default="/projectnb/cs585bp/projects/Catastrophe/Vision-Language-For-The-Blind/src/checkpoints/model_epoch_13.pth",
        help="Path to load the pretrained model"
    )

    parser.add_argument(
        "--output_hidden_states",
        action="store_true",
        help="Return hidden states of all layers if True",
        default=False
    )

    parser.add_argument(
        "--output_attn_weights",
        action="store_true",
        help="Return last layer's attention weights of input image and text if True",
        default=False
    )

    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Logging started. All logs for this run will be saved at {LOG_FILE}")

    try:
        outputs = viz(args, logger)
    except Exception as e:
        logger.error(f"Unexpected exception occured: {e}")
        error_msg = traceback.format_exc()
        logger.error(error_msg)
    
    # print(f"{outputs.keys()}, image_self: {outputs['attn_weights']['image_self'].shape}, image_guided: {outputs['attn_weights']['image_guided'].shape}, text_self: {outputs['attn_weights']['text_self'].shape}")