import torch
import os
import pandas as pd 
import numpy as np
from torchvision.io import decode_image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Lambda
from torchvision.transforms import v2 as v2
from torchvision.utils import draw_bounding_boxes
from torchvision import tv_tensors
from torchvision.transforms.functional import get_image_size
from sklearn import preprocessing
import time
import math

# device = torch.device('xpu:0' if torch.xpu.is_available() else 'cpu')

target_size = (512, 512)


transform_pipeline = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float),
    v2.Resize(target_size)
])

def readBoxes(loc: str) -> tuple:
    rt = []
    dta = pd.read_csv(loc)
    for i in range(dta['filename'].size):
        # rt = rt + ((dta["xmin"][i], dta["ymin"][i], dta["xmax"][i], dta["ymax"][i]), )
        newArr = [dta["xmin"][i], dta["ymin"][i], dta["xmax"][i], dta["ymax"][i]]
        rt.append(newArr)
    
    return rt
    


device = "xpu" if torch.xpu.is_available() else "cpu"

def convertBoxes(oldX: int, oldY: int, newSize: tuple, bb: tuple) -> tuple:
    newY = newSize[1] / oldY
    newX = newSize[0] / oldX
    return (bb[0] * newX, bb[1] * newY, bb[2] * newX, bb[3] * newY)

boxes = readBoxes('./dogDataset/train.csv')
# print(boxes[0])

# print(readBoxes('./dogDataset/train.csv'))

# Build our class for our dataset
class CustomImageDataset(Dataset):
    def __init__ (self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file) 
        self.img_dir = img_dir
        # self.bboxs = bboxs
        self.transform = transform
        self.target_transform = target_transform
        print('created Dataset')

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        # print(image.size())
        height, width = get_image_size(image)
        # print(self.img_labels["xmin"][idx])
        labels = self.img_labels["label"][idx]
        bbox = np.array([self.img_labels["xmin"][idx], self.img_labels["ymin"][idx], self.img_labels["xmax"][idx], self.img_labels["ymax"][idx]], dtype=np.float32)
        # bbox = torch.from_numpy(bbox)
        bbox = tv_tensors.BoundingBoxes(bbox, format="XYXY", dtype=torch.float, canvas_size=(width, height))
        
        if self.transform:
            image, bbox = self.transform(image, bbox)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # print(labels)
        # print(image.size())
        return image, bbox, labels


dogImagesDir = './dogDataset/dogImages'


training_data = CustomImageDataset('./dogDataset/train.csv', dogImagesDir, transform_pipeline, None)
test_data = CustomImageDataset('./dogDataset/validate.csv', dogImagesDir, transform_pipeline, None)



# use dataloader pin_memory=True, pin_device=device
# REMASK THESE SOME ARE BAD!!!
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# train_dataloader = train_dataloader.to(device)
# test_dataloader = test_dataloader.to(device)

# print('Made dataloaders')
# #displat image and label
# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0][0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")
# # print(train_features.size())

# print(train_boxes)


def show_boxes(dataloader):
    fig = plt.figure(figsize=(16, 10))
    for image, bbox, labels in dataloader:
        rg = image.size(dim=0)
        # print(image[2][0][0])

        for i in range(rg):
            ax = fig.add_subplot(math.ceil(rg/7), 7, i+1)
            bb = bbox[i]
            print(bb)
            # bb = tv_tensors.BoundingBoxes(bb, format="XYXY", dtype=torch.uint8, canvas_size=target_size)
            img = image[i].to(torch.uint8).squeeze()
            # print(bb.size())
            img_bb = draw_bounding_boxes(image=img, boxes=bb, labels=["Athena"], colors="red", width=3)
            finalImg = v2.ToPILImage()(img_bb)
            ax.imshow(finalImg)
    
    mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()    
    plt.show()


show_boxes(train_dataloader)

# exit()

# plt.show()
print("Using", device)

# Building the network
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Conv2d(1,3, kernel_size=3, stride=2),
            torch.nn.MaxPool2d(1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(3, 6, kernel_size=3, stride=2),
            torch.nn.MaxPool2d(1, stride=1),
            torch.nn.ReLU(),

            # torch.nn.Linear(512*1536, 512),
            torch.nn.Linear(196607, 512),
            torch.nn.ReLU(),
            # torch.nn.Flatten(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
            # torch.nn.Softmax(dim=0)
        )

    def forward(self, x):
        # print(x)
        x = self.flatten(x)
        x = x.unsqueeze(dim=0)
        logits = self.linear_relu_stack(x)
        return logits



model = NeuralNetwork().to(device)
# print(model)


# X = torch.rand(1,28,28, device=device)
# logits = model(X)
# pred_probab = torch.nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicated class: {y_pred}")


# x = torch.ones(5)
# y = torch.zeros(3)
# w = torch.randn(5,3, requires_grad=True) 
# b = torch.randn(3, requires_grad=True)
# z = torch.matmul(x, w)+b

# loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

# print(f"Gradient for loos function: {loss.grad_fn}")

# loss.backward()

# print(device)

epochs = 40 # Number of times to iterate over the dataset
learning_rate = 1e-10 # How much to update model parameters at each bath/epoch (Smaller = Slow learning, Large Values = Unpredicatble behavior while training)
batch_size = 64 # the #of data samples propagated through the network before the parameters are updated



#epoch training/testing loops

loss_fn = torch.nn.CrossEntropyLoss() # Loss function -> combines Mean Square Error and Negative Log Likelihood functions
label_encoder = preprocessing.LabelEncoder()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #Adjusts model parameters to reduce model error in each step

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set to training mode
    model.train()
    # print(next(iter(dataloader)))
    for image, bbox, labels in dataloader:
        image = image.to(device, non_blocking=True)
        numLabel = label_encoder.fit_transform(labels)
        tLabel = torch.tensor(numLabel)
        tLabel = tLabel.to(device, non_blocking=True)
        pred = model(image).to(device)
        print(pred)
        print(pred.shape)
        # print(tLabel)
        loss = loss_fn(pred,tLabel)
        optimizer.zero_grad()
        print("-")
        loss.backward()
        print("--")
        optimizer.step()
        print("---")
        print(f"Epoch loss: {loss.item()}")
        #Some back propagation
        


def test_loop(dataloader, model, loss_fn):
    # fpr test set model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Using torch.no_grad() means no gradients are being computed during test mode
    with torch.no_grad():
        for image, bbox, labels in dataloader:
            image = image.to(device, non_blocking=True)
            pred = model(image).to(device)
            tLabel = torch.tensor(label_encoder.fit_transform(labels)).to(device, non_blocking=True)
            # print(tLabel)
            test_loss += loss_fn(pred, tLabel).item()
            correct += (pred.argmax(1) == tLabel).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)



print(device)
# save model
# torch.save(model.state_dict(), "Athena_Network.pth")


def isAthena(imageDir, model, trained=False, model_name=None):
    model.eval() #switch to eval mode
    img = decode_image(imageDir)
    img = transform_pipeline(img)
    img = img.unsqueeze(0).flatten(start_dim=1)
    img = img.to(device)
    # img = DataLoader(img, batch_size=1, shuffle=False)
    # img = img.unsqueeze(0)
    # print(img.size())
    output = model(img).to(device)
    print(round(output[0, 0].item(), 5))
    rounded = round(output[0, 0].item())
    print(rounded)
    fileName = imageDir.split("/")[-1]
    print(f"{fileName}: True") if rounded < 0.5 else print(f"{fileName}: False")


# model = torch.load('Athena_Network.pth', weights_only=False)
isAthena('./testImages/isDog.jpg', model)
isAthena('./testImages/notDog.jpg', model)
isAthena('./testImages/campfire.jpg', model)
isAthena('./testImages/darkSky.jpg', model)
isAthena('./testImages/topDog.jpg', model)
isAthena('./testImages/closeUpAthena.jpg', model)


# time.sleep(7)
print("Done")
