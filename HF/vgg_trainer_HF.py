import argparse
from tqdm import tqdm
from dataloader import CustomImageDataset, get_image_list
from torchvision import transforms
from torch.utils.data import random_split
import torch
from torch import nn, optim
from torchvision.models import vgg16
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
import wandb


def main(args):
    data_path = args.data_path
    project_name = args.project_name
    run_name = args.run_name
    lr = args.lr
    momentum = args.momentum
    batch_size = args.batch_size
    epochs = args.epochs
    freeze_proportion = args.freeze_proportion
    
    dataset_path = data_path
    
    wandb.init(project=project_name, name=run_name)
    wandb.config.update({"lr": lr, "momentum": momentum, "batch_size": batch_size, "epochs": epochs, "freeze_proportion": freeze_proportion})
    
    folders = ['07062021', '08032021', '12042021', '15022021']
    folders = [f"{dataset_path}/{folder}" for folder in folders]

    index_files = [f"{dataset_path}/07062021.csv", f"{dataset_path}/08032021.csv", f"{dataset_path}/12042021.csv", f"{dataset_path}/15022021.csv"]

    image_list = get_image_list(folders=folders, index_files=index_files, isMerged=True)

    transform = transforms.Compose([
                transforms.Resize((224, 224)),  
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor(), 
                ])
                                   
    dataset = CustomImageDataset(image_list, transform=transform, isColor=True)
    
    train_proportion = 0.7
    val_proportion = 0.2
    
    total_size = len(dataset)
    train_size = int(train_proportion * total_size)
    val_size = int(val_proportion * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = vgg16(pretrained=True)
    
    model.classifier[6] = nn.Linear(in_features=model.classifier[6].in_features, out_features=3)

    total_layers = len(list(model.features.parameters()))
    freeze_layers = int(total_layers * freeze_proportion)
    
    print(f'Total layers: {total_layers}, layers frozen: {freeze_layers}')

    for i, param in enumerate(model.features.parameters()):
        if i < freeze_layers:
            param.requires_grad = False
        
    for name, param in model.named_parameters():
        print(f'Layer: {name}, Frozen: {not param.requires_grad}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    device = torch.device("cuda:0")
    model = model.to(device)

    pbar = tqdm(total=epochs, desc="Training Progress")
    
    best_val_accuracy = 0.0
    path_to_save_model = f'{run_name}.pth'
    
    for epoch in range(epochs): 
        correct = 0
        total = 0
        for inputs, labels in train_loader:
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
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

            val_accuracy = accuracy_score(all_labels, all_predictions)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), path_to_save_model)

        pbar.update(1)
        pbar.set_postfix({"Epoch": epoch, "Loss": loss.item() / total, "Accuracy": correct / total})
        wandb.log({"epoch": epoch, "loss": loss.item(), "train_accuracy": train_accuracy, "val_accuracy": val_accuracy})
    
    model.eval()
    with torch.no_grad():
        all_labels = []
        all_predictions = []
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
                                class_names=['0', '1', '2'])})
        wandb.log({"test_accuracy": accuracy, "test_f1": f1})
    
    print(f'Accuracy on test data: {accuracy}, F1 Score on test data: {f1}')
    torch.save(model.state_dict(), f"{run_name}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument("--project_name", type=str, required=True, help="Project name wandb")
    parser.add_argument("--run_name", type=str, required=True, help="Run name wandb")
    parser.add_argument("--lr", type=float, required=False, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, required=False, default=0.9, help="Momentum")
    parser.add_argument("--batch_size", type=int, required=False, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, required=False, default=10, help="Number of epochs")
    parser.add_argument("--freeze_proportion", type=float, required=True, help="Proportion of layers to freeze, '1', '15', '30', '50' ")
    args = parser.parse_args()
    
    main(args)
    
    