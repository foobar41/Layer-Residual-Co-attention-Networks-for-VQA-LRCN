## Need to run this file only once
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

class ImageFeatureExtractor:
    def __init__(self, feat_dir):
        self.feat_dir = feat_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## Create resnet152 backbone without classification layers
        resnet = models.resnet152(pretrained=True)
        self.feature_extractor = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ).to(self.device).eval()

        # self.downsample = nn.Conv2d(2048, 2048, kernel_size=2, stride=2).to(self.device)
        # # self.projection = nn.Conv2d(2048, 512, kernel_size=1).to(self.device)
        # self.projection = nn.Linear(2048, 512).to(self.device)

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(448),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def process_image(self, img_path):
        img = Image.open(img_path)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        ## Extract features
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)

        ## Pad to 16x16
        if features.size(2) != 16 or features.size(3) != 16:  # For NCHW format
            features = nn.functional.pad(features, (1,1,1,1))

        ## Downsample and project
        # features = self.downsample(features)

        # # Permute to [1, 8, 8, 2048] for linear layer
        # features = features.permute(0, 2, 3, 1).contiguous()
        # # Apply linear projection [1, 8, 8, 512]
        # features = self.projection(features)
        # features = features.view(1, -1, 512)

        ## Features returned are (64, 512)
        return features.squeeze(0).cpu().detach().numpy()

    def process_directory(self, img_dir, split_name):
        output_dir = os.path.join(self.feat_dir, split_name)
        os.makedirs(output_dir, exist_ok=True)
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

        for filename in tqdm(img_files, desc=f'Preprocessing {split_name}'):
            img_path = os.path.join(img_dir, filename)
            try:
                features = self.process_image(img_path)
                base_name = os.path.splitext(filename)[0]
                np.save(os.path.join(output_dir, f'{base_name}.npy'), features)
            except Exception as e:
                print(f'Error processing {filename}: {str(e)}')