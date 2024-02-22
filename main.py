import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from capsnet import CapsNet, CapsuleLoss
from torch.utils.data import DataLoader, random_split

import pre_processing
from datasets import DRDataset



# Check cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    model = CapsNet().to(device)
    criterion = CapsuleLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)


    BATCH_SIZE = 128

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = DRDataset(csv_file='datasets/idrid/test.csv', root_dir='pre_processing/datasets/idrid/test', transform=transform)
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
            correct += torch.sum(
                torch.argmax(logits, dim=1) == torch.argmax(labels, dim=1)).item()
            total += len(labels)
            accuracy = correct / total
            total_loss += loss
            loss.backward()
            optimizer.step()
            print('Epoch {}, batch {}, loss: {}, accuracy: {}'.format(ep + 1,
                                                                      batch_id,
                                                                      total_loss / batch_id,
                                                                      accuracy))
            batch_id += 1
        scheduler.step(ep)
        print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))

    # Eval
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        # Add channels = 1
        images = images.to(device)
        # Categogrical encoding
        labels = torch.eye(10).index_select(dim=0, index=labels).to(device)
        logits, reconstructions = model(images)
        pred_labels = torch.argmax(logits, dim=1)
        correct += torch.sum(pred_labels == torch.argmax(labels, dim=1)).item()
        total += len(labels)
    print('Accuracy: {}'.format(correct / total))

    # Save model
    torch.save(model.state_dict(), './model/capsnet_ep{}_acc{}.pt'.format(EPOCHES, correct / total))


if __name__ == '__main__':
    main()
    #pre_processing.run()
