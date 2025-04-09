import argparse
import os
import cv2 as cv
import numpy as np
from config import *
import torch
from Model import ModelQD
from torchvision.transforms import ToTensor



def get_args() :
    parser = argparse.ArgumentParser("Quick Draw classifier")
    parser.add_argument("--image-size", '-i', type= int, default=28, help="Common size of all images")
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
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv.namedWindow("App Painting")
    global ix, iy, is_drawing
    is_drawing = False
    def draw_app(event, x, y, flags, param) :
        global ix, iy, is_drawing
        if event == cv.EVENT_LBUTTONDOWN :
            is_drawing = True
            ix , iy = x, y
        elif event == cv.EVENT_MOUSEMOVE :
            if is_drawing == True :
                cv.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
                ix = x
                iy = y
        elif event == cv.EVENT_LBUTTONUP :
            is_drawing = False
            cv.line(image, (ix, iy), (x, y), WHITE_RGB, 5)
            ix = x
            iy = y
        return x, y
    cv.setMouseCallback("App Painting", draw_app)
    while(True) :
        key = cv.waitKey(10)
        if key == 27 or cv.getWindowProperty("App Painting", cv.WND_PROP_VISIBLE) < 1 :
            break
        cv.imshow("App Painting", 255- image)
        if key == ord(" ") :
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            ys, xs = np.nonzero(image)
            min_x = np.min(xs)
            max_x = np.max(xs)
            min_y = np.min(ys)
            max_y = np.max(ys)
            image_gray = image[min_y : max_y , min_x :max_x]

            image = cv.resize(image_gray, (28, 28))
            image = transform(image).unsqueeze(0).to(device)
            result = model(image)
            print(CLASSES[torch.argmax(result[0])])
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            ix = -1
            iy = -1



if __name__ == '__main__':
    args = get_args()
    app(args)

