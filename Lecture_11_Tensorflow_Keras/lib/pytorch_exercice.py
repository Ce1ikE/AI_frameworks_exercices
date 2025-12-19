import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import datetime as dt
import json

from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 
from pathlib import Path

from .pytorch_models import ConvNet

FONTDICT = {    
    'fontsize': 14,
    'fontweight': 'bold',
    'color': 'darkblue',
    "fontfamily": "monospace"
}

def pytorch_example(results_path: Path = Path("./results")):
    results_path = results_path / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    batch_size = 32

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root='./datasets', 
        train=True,
        download=True, 
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )
    testset = torchvision.datasets.CIFAR10(
        root='./datasets', 
        train=False,
        download=True, 
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )

    # Plotting some images from the dataset
    classes = ('airplane', 'automobile', 'bird', 'cat', 
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(imgs):
        plt.figure(figsize=(8,8))
        plt.title("Sample Images from CIFAR-10 Dataset", fontdict=FONTDICT)
        imgs = imgs/2+0.5 #unnormalize
        np_imgs = imgs.numpy()
        plt.imshow(np.transpose(np_imgs, (1,2,0)))
        plt.savefig(results_path / "sample_images_pytorch.png")
        plt.close()

    #one batch of random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    im_grid = torchvision.utils.make_grid(images[0:25], nrow=5)
    imshow(im_grid)
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(25)))

    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Training loop
    steps_per_epoch = len(trainloader)

    for epoch in tqdm(range(10), desc="Training epochs", unit="epoch"):
        running_loss = 0.0
        for (inputs, labels) in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            #forward + loss
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #backward + optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'[{epoch +1}] loss: {running_loss/steps_per_epoch:.3f}')
    print('Train is finished')    

    n_correct = 0
    n_total = 0

    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for(images, labels) in tqdm(testloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            #take class with highest value as prediction
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            n_total += labels.size(0)
            n_correct += (predicted == labels).sum().item()
    cm = confusion_matrix(all_labels, all_predictions)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(10,8))
    sns.heatmap(
        cm_df, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=True,
        linewidths=0.5,
        linecolor='black',
        annot_kws={
            "fontsize":12,
            "fontweight":"bold",
            "fontfamily":"monospace"
        },
        xticklabels=classes,
        yticklabels=classes
    )
    plt.ylabel('Actual', fontdict=FONTDICT)
    plt.xlabel('Predicted', fontdict=FONTDICT)
    plt.title('Confusion Matrix', fontdict=FONTDICT)
    plt.savefig(results_path / "confusion_matrix_pytorch.svg", format='svg')
    plt.close()

    with open(results_path / "pytorch_results.json", "w") as f:
        json.dump({
            "accuracy": n_correct/n_total,
            "n_correct": n_correct,
            "n_total": n_total,
        }, f, indent=4)

    # save the model
    torch.save(model.state_dict(), results_path / "convnet_cifar10_pytorch.pth")

    # Collect all predictions and labels for confusion matrix
    print(f"Accuracy on test set: {n_correct/n_total:.3f}")

if __name__ == "__main__":
    pytorch_example(results_path=Path("./results"))