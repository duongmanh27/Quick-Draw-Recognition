import argparse
import os
import cv2 as cv
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from Model import ModelQD
from config import *


def get_args() :
    parser = argparse.ArgumentParser("Quick Draw classifier")
    parser.add_argument("--image-size", '-i', type=int, default=28, help="Common size of all images")
    parser.add_argument("--image-path", "-d", type=str, default="images/eye_1.jpg",
        help="Path to data")
    parser.add_argument("--checkpoint_dir", type=str, default="models", help="Where to store the trained model")
    args, knows = parser.parse_known_args()
    return args


def app(args) :
    transform = ToTensor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModelQD(num_classes=10)

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best.pt"), weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    image = cv.imread(args.image_path, cv.IMREAD_GRAYSCALE)
    image = transform(image).unsqueeze(0).to(device)
    softmax = nn.Softmax()
    with torch.no_grad() :
        result = model(image)
        prob = softmax(result)[0]
        predict_class = CLASSES[torch.argmax(prob)]
    print(predict_class)


if __name__ == '__main__' :
    args = get_args()
    app(args)
