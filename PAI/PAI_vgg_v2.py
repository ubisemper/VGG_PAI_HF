import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch
from torch import nn, optim
from torchvision.models import vgg16
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import numpy as np
from tqdm import tqdm
import argparse

def normalize_image(image):
    image = np.array(image)
    min_val = np.min(image)
    max_val = np.max(image)
    image = (image - min_val) / (max_val - min_val) * 255 
    return Image.fromarray(image.astype(np.uint8))

class NPZDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.files = [os.path.join(root_dir, class_dir, file)
                      for class_dir in self.classes
                      for file in os.listdir(os.path.join(root_dir, class_dir))]
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path)['data']
        data = np.nan_to_num(data)
        data = Image.fromarray(data)
        label = self.classes.index(os.path.basename(os.path.dirname(file_path)))

        if self.transform:
            data = self.transform(data)

        return data, label

def main(args):
    data_path = args.data_path
    project_name = args.project_name
    run_name = args.run_name
    lr = args.lr
    momentum = args.momentum
    batch_size = args.batch_size
    epochs = args.epochs
    freeze_proportion = args.freeze_proportion
    model_path = args.model_path
    
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    val_path = os.path.join(data_path, 'val')
    
    wandb.init(project=project_name, name=run_name)
    wandb.config.update({"lr": lr, "momentum": momentum, "batch_size": batch_size, "epochs": epochs, "freeze_proportion": freeze_proportion})
    
    transform = transforms.Compose([transforms.Lambda(normalize_image),
                                transforms.Resize((224, 224)),
                                transforms.Grayscale(num_output_channels=3),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(20),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    train_data = NPZDataset(train_path, transform=transform)
    test_data = NPZDataset(test_path, transform=transform)
    val_data = NPZDataset(val_path, transform=transform)
    
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size)
    valloader = DataLoader(val_data, batch_size=batch_size)
    
    if args.model_path is None:
        model = vgg16(pretrained=True)
        print(f'## Using standerd imagenet as pretrained model')
    else:
        model = vgg16(pretrained=False)
        model.load_state_dict(torch.load(model_path))
    
    print(f'## Using {model_path} as pretrained model')
    
    model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=8)
    
    total_layers = len(list(model.features.parameters()))
    freeze_layers = int(total_layers * freeze_proportion)
    print(f'## Total layers: {total_layers}, layers frozen: {freeze_layers}')

    for i, param in enumerate(model.features.parameters()):
        if i < freeze_layers:
            param.requires_grad = False
        
    for name, param in model.named_parameters():
        print(f'## Layer: {name}, Frozen: {not param.requires_grad}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    device = torch.device("cuda:0")
    model = model.to(device)

    pbar = tqdm(total=epochs, desc="Training Progress")
    
    for epoch in range(epochs):  
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss.backward()
            optimizer.step()

        train_accuracy = correct / total

        model.eval()
        with torch.no_grad():
            all_labels = []
            all_predictions = []
            for inputs, labels in valloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

            val_accuracy = accuracy_score(all_labels, all_predictions)

        pbar.update(1)
        pbar.set_postfix({"Epoch": epoch, "Loss": loss.item() / total, "Accuracy": correct / total})
        wandb.log({"epoch": epoch, "loss": loss.item(), "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
    

    model.eval()
    with torch.no_grad():
        all_labels = []
        all_predictions = []
        test_loader = testloader
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        wandb.log({"test_confusion_matrix": wandb.plot.confusion_matrix(probs=None,
                                y_true=all_labels,
                                preds=all_predictions,
                                class_names=['0', '1', '2', '3', '4', '5', '6', '7'])})
        wandb.log({"test_accuracy": accuracy, "test_f1": f1})

    print(f'Accuracy on test data: {accuracy}, F1 Score on test data: {f1}')
    
    torch.save(model.state_dict(), f"{run_name}.pth")
    wandb.save(f"{run_name}.pth")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument("--project_name", type=str, required=True, help="Project name wandb")
    parser.add_argument("--run_name", type=str, required=True, help="Run name wandb")
    parser.add_argument("--lr", type=float, required=False, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, required=False, default=0.9, help="Momentum")
    parser.add_argument("--batch_size", type=int, required=False, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, required=False, default=10, help="Number of epochs")
    parser.add_argument("--freeze_proportion", type=float, required=True, help="Proportion of layers to freeze.")
    parser.add_argument("--model_path", type=str, required=False, default=None, help="Path to model")
    args = parser.parse_args()
    
    main(args)