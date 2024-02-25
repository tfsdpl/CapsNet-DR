import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from capsnet import CapsNet, CapsuleLoss
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pre_processing
from datasets import DRDataset
writer = SummaryWriter('runs/experimento_5')



# Check cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train_model(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)

    model = CapsNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CapsuleLoss()

    # model = CapsNet().to(device)
    # criterion = CapsuleLoss()
    #0.88
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-8)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.05)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.01)



    BATCH_SIZE = 6

    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor(),
    #     transforms.Grayscale(num_output_channels=3),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = DRDataset(csv_file='datasets/APTOS/train.csv', root_dir='pre_processing/datasets/APTOS/train', transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Train
    EPOCHES = 50
    model.train()
    for ep in range(EPOCHES):
        batch_id = 1
        correct, total, total_loss = 0, 0, 0.
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
            logits, reconstruction = model(images)

            # Compute loss & accuracy
            loss = criterion(images, labels, logits, reconstruction)
            correct += torch.sum(  torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('Loss/train', loss.item(), ep * len(train_loader) + batch_id)
            writer.add_scalar('Accuracy/train', accuracy, ep * len(train_loader) + batch_id)

            print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(ep + 1,
                                                                          batch_id,
                                                                          total_loss / batch_id,
                                                                          accuracy))
            batch_id += 1
        scheduler.step()
        writer.add_scalar('Loss/epoch', total_loss / len(train_loader), ep)
        writer.add_scalar('Accuracy/epoch', correct / total, ep)
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))

    # Eval
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
        logits, reconstructions = model(images)
        pred_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(pred_labels == torch.argmax(labels, dim=1)).item()
        total += len(labels)
    print('Accuracy: {}'.format(correct / total))

    # Save model
    torch.save(model.state_dict(), './model/capsnet_ep{}_acc{}.pt'.format(EPOCHES, correct / total))

def objective(trial):
    validation_accuracy = train_model(trial)
    return validation_accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    print("Melhores parâmetros: ", study.best_trial.params)

# if __name__ == '__main__':
#     main()
#     #pre_processing.run()
