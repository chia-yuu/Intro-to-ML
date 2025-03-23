# resnet 18, python 3.11.7
import os
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
from PIL import Image

# setting gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(device): print("using gpu!")
else: print("no gpu:(")

categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
train_dir = 'Images/train'
val_dir = 'Images/val'

# train & val
# mkdir
for c in categories:
    val_cat_dir = os.path.join(val_dir, c)
    os.makedirs(val_cat_dir)

# separate data
for c in categories:
    cat_train_dir = os.path.join(train_dir, c)
    cat_val_dir = os.path.join(val_dir, c)
    img = os.listdir(cat_train_dir)
    val_img = random.sample(img, int(0.2 * len(img)))
    for m in val_img:
        shutil.move(os.path.join(cat_train_dir, m), os.path.join(cat_val_dir, m))

# generate img
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 7)
model.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_acc = 0.0
best_model_wts = None

# train
for epoch in range(25):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    scheduler.step()

    # eval
    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Epoch {epoch+1} Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = model.state_dict()

# store the best model
torch.save(best_model_wts, 'model.pth')
model.load_state_dict(best_model_wts)

model.eval()
imdex_map = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Neutral": 4, "Sad": 5, "Surprise": 6}

# pred
prediction_data = []
for filename in os.listdir('Images/test'):
    image_path = os.path.join('Images/test', filename)
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        label = preds.item()

    file_name = os.path.splitext(filename)[0]
    prediction_data.append({
        'filename': file_name,
        'label': label
    })

df = pd.DataFrame(prediction_data)
df.to_csv('pred.csv', index=False)

# model01: lr = 1e-3, epoch = 20, ac = 0.43473
# model02: lr = 1e-4, epoch = 20, ac = 0.49886 ** good
# model03: lr = 1e-4, epoch = 30, ac = 0.49545 ** good
# model04: lr = 1e-5, epoch = 20, ac = 0.43586
# model05: lr = 1e-5, epoch = 30, ac = 0.46424
# model06: lr = 1e-5, epoch = 50, ac = 0.47843
# model07: lr = 1e-4, epoch = 50, ac = 0.51475 ** best
# model08: lr = 1e-4, epcho = 100, ac = 0.50227 (overfit)
# model09: lr = reduce, epoch = 40, val_accuracy: 0.3588
# model10: lr = 1e-4, epoch = 100, layer.trainable=100, val_accuracy: 0.4557
# model11: lr = 1e-4, epoch = 60, ---
# model12: lr = 1e-4, epcho = 40, ac = 0.50454

'''
lr = 1e-4
model02: lr = 1e-4, epoch = 20, ac = 0.49886 ** good
model03: lr = 1e-4, epoch = 30, ac = 0.49545 ** good
model07: lr = 1e-4, epoch = 50, ac = 0.51475 ** best
model08: lr = 1e-4, epcho = 100, ac = 0.50227 (overfit)
model10: lr = 1e-4, epoch = 100, layer.trainable=100, val_accuracy: 0.4557
model11: lr = 1e-4, epoch = 60    terminal 1
model12: lr = 1e-4, epcho = 40, ac = 0.50454
'''

# res18_2: SDG lr=0.001 momentum=0.9, StepLR step_size=7 gamma=0.1, epoch 25  ** best
# res18_3: SDG lr=0.001 momentum=0.9, ReduceLROnPlateau factor=0.2 patience=3, epoch 50
# -- : SDG lr=0.001 momentum=0.9, StepLR step_size=4 gamma=0.1, epoch 30
# res18_4: SDG lr=0.001 momentum=0.9, StepLR step_size=10 gamma=0.2, epoch 30
# res18_5: SDG lr=0.001 momentum=0.9, StepLR step_size=7 gamma=0.1, epoch 50
# res18_6: SDG lr=0.001 momentum=0.8, StepLR step_size=7 gamma=0.1, epoch 30, val ac = 0.58
# res18_7: SDG lr=0.0001 momentum=0.9, StepLR step_size=7 gamma=0.1, epoch 30