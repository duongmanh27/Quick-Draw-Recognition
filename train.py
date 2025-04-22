import torch.optim as optim
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os
from Model import ModelQD
from dataset import DataQuickDraw
import matplotlib.pyplot as plt
import numpy as np
from tqdm.autonotebook import tqdm
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from config import CLASSES

def get_args() :
    parser = argparse.ArgumentParser("Quick Draw classifier")
    parser.add_argument("--batch_size", "-b", type=int, default=2, help="Batch size of train and val process")
    parser.add_argument("--image-size", '-i', type= int, default=28, help="Common size of all images")
    parser.add_argument("--data-path","-d" ,type=str, default="E:\\Data\\quick_draw", help="Path to data")
    parser.add_argument("--lr", '-l', type=int, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", '-e', type=int, default=50, help="Learning rate")
    parser.add_argument("--tensorboard-dir", '-t', type=str, default="quick_draw_board", help="Where to store the tensorboard logging")
    parser.add_argument("--checkpoint_dir", type=str, default="trained_model", help="Where to store the trained model")
    parser.add_argument("--resume", "-r", type=bool, default=False, help="Continue training from last session")
    args, knows = parser.parse_known_args()
    return args
def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('Confusion_matrix', figure, epoch)
    writer.flush()
def train(args) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_train = DataQuickDraw(args.data_path, is_train=True)
    data_val = DataQuickDraw(args.data_path, is_train=False)
    train_loader = DataLoader(data_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_loader = DataLoader(data_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    model = ModelQD(num_classes=data_train.num_classes)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.resume :
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "last.pt"))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['start_epoch']
        best_acc = checkpoint['best_acc']
    else:
        start_epoch = 0
        best_acc = -1

    if not os.path.isdir(args.checkpoint_dir) :
        os.mkdir(args.checkpoint_dir)
    if not os.path.isdir(args.tensorboard_dir) :
        os.mkdir(args.tensorboard_dir)
    writer = SummaryWriter(args.tensorboard_dir)
    num_iters_per_epoch = len(train_loader)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, args.epochs) :
        model.train()
        losses = []
        progress_bar_train = tqdm(train_loader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar_train) :
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            losses.append(loss.item())
            avg_loss = np.mean(losses)
            progress_bar_train.set_description(
                "Epoch : {}/{}. Loss : {:.4f}".format(epoch +1,args.epochs, avg_loss)
            )
            writer.add_scalar(tag= "Train/Loss", scalar_value=avg_loss, global_step=epoch * num_iters_per_epoch + iter)
            loss.backward()
            optimizer.step()
        #val
        model.eval()
        losses = []
        all_labels = []
        all_predicts = []
        progress_bar_val = tqdm(val_loader, colour="yellow")
        for iter, (images, labels) in enumerate(progress_bar_val) :
            all_labels.extend(labels)
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            predictions = torch.argmax(output, dim= 1)
            all_predicts.extend(predictions.tolist())
            loss = criterion(output, labels)
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        avg_acc = accuracy_score(all_labels, all_predicts)
        print("Epoch : {}/{}. Loss : {:.4f}. Accuracy : {:.4f}".format(epoch + 1, args.epochs, avg_loss, avg_acc))
        writer.add_scalar(tag="Val/Loss", scalar_value=avg_loss, global_step=epoch)
        writer.add_scalar(tag="Val/Accuracy", scalar_value=avg_acc, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predicts), CLASSES, epoch)
        checkpoint = {
            'epoch' : epoch + 1,
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict() ,
            "best_acc" : best_acc
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))
        if not np.isclose(avg_acc, best_acc, atol=1e-6) and avg_acc > best_acc:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pt"))
            best_acc = avg_acc
if __name__ == '__main__':
    args = get_args()
    train(args)


