from preprocessing.image_feature_extraction import ImageFeatureExtractor
import os
import argparse


def preprocess_image(feature_output_dir, image_dir, split):

    extractor = ImageFeatureExtractor(feature_output_dir)
    split_dir = os.path.join(image_dir, split)
    if os.path.exists(split_dir):
        print(f'Extracting features from {split_dir}.')
        extractor.process_directory(split_dir, split)
    else:
        print(f'{split_dir} path DOES NOT exist')
    
    print(f"Image features saved at: {feature_output_dir}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images to extract ResNext152 features.")

    parser.add_argument(
        "--feature_output_dir",
        type=str,
        required=False,
        help="Directory to save the extracted features.",
        default="./data/VQAv2/images/image_features"
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        required=False,
        help="Directory containing the images.",
        default="./data/VQAv2/images/train"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        help="Split name of the dataset. ",
        default="train2014")
    
    parser.add_argument(
        "--overwrite",
        type=bool,
        required=False,
        help="Overwrite existing features.",
        default=False
    )
    
    args = parser.parse_args()
    if args.overwrite:
        preprocess_image(args.feature_output_dir, args.image_dir, args.split)
    else:
        output_dir = os.path.join(args.feature_output_dir, args.split)
        if os.path.exists(output_dir):
            print(f"Features already exist at {output_dir}. Use --overwrite to overwrite.")
        else:
            preprocess_image(args.feature_output_dir, args.image_dir, args.split)

# config = {
#     'image_dir': ['./data/VQAv2/images/train','./data/VQAv2/images', './data/VQAv2/images'],
#     'feat_dir': './data/VQAv2/images/img_features',
#     'splits': ['train2014', 'val2014', 'test2015']
# }