from torch import optim
from torchvision.transforms import transforms
from capsnet import CapsNet, CapsuleLoss
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from datasets import DRDataset
import torch
import numpy as np
import random
import os
import random


seed = 42
seed_np = np.random.randint(seed)
random.seed(seed_np)
np.random.seed(seed_np)

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Se você estiver usando CUDA com PyTorch
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Para ambientes que utilizam multi-threading
os.environ['PYTHONHASHSEED'] = str(seed)

#tensorboard --logdir=runs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main():
    valor = 0.000001
    for i in range(50):
        LR = 0.000005
        WD =  0.000005
        writer = SummaryWriter(f'runs/experimento_{i}')
        model = CapsNet().to(device)
        criterion = CapsuleLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        BATCH_SIZE = 6
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
            if ep == 20:
                LR =  LR+valor

                optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)


            writer.add_scalar('Loss/epoch', total_loss / len(train_loader), ep)
            writer.add_scalar('Accuracy/epoch', correct / total, ep)
            print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))
            if correct / total >= 0.5:
                break

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
        torch.save(model.state_dict(), './model/capsnet_ep{}_acc{}_wd{}_lr{}.pt'.format(EPOCHES, correct / total, WD, LR))
        valor += 0.000001


if __name__ == '__main__':
    main()
    #pre_processing.run()
