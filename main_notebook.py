import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms, models
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np




    return (
        DataLoader,
        accuracy_score,
        confusion_matrix,
        datasets,
        f1_score,
        models,
        nn,
        optim,
        plt,
        torch,
        transforms,
    )


@app.cell
def _():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("sayedgamal99/smoke-fire-detection-yolo")

    print("Path to dataset files:", path)
    return


@app.cell
def _(torch):
    data_dir = "dataset"
    batch_size = 32
    num_epochs = 5
    lr = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return batch_size, data_dir, device, lr, num_epochs


@app.cell
def _(DataLoader, batch_size, data_dir, datasets, transforms):
    # ===== ТРАНСФОРМАЦИИ =====
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(f"{data_dir}/train", transform=transform_train)
    val_dataset = datasets.ImageFolder(f"{data_dir}/val", transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


@app.cell
def _(device, lr, models, nn, optim):
    # ===== ПРЕДОБУЧЕННАЯ МОДЕЛЬ =====
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model
    return criterion, model, optimizer


@app.cell
def _(
    criterion,
    device,
    model,
    num_epochs,
    optimizer,
    plt,
    torch,
    train_loader,
    val_loader,
):
    # for epoch in range(num_epochs):
    #     model.train()
    #     running_loss = 0

    #     for images, labels in train_loader:
    #         images, labels = images.to(device), labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # Списки для хранения метрик
    train_losses_epoch = []  # средняя потеря за эпоху
    val_accuracies = []      # точность на валидации после эпохи

    # Цикл обучения (уже есть)
    all_loss = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            all_loss.append(loss.item())
    
        avg_loss = running_loss / len(train_loader)
        train_losses_epoch.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Валидация после эпохи
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        val_accuracies.append(val_acc)
        print(f"Validation Accuracy: {val_acc:.4f}")

    # После окончания обучения строим графики
    plt.figure(figsize=(15, 5))

    # График потерь по итерациям
    plt.subplot(1, 3, 1)
    plt.plot(all_loss, linewidth=0.5)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss per iteration')
    plt.grid(True)

    # График средней потери по эпохам
    plt.subplot(1, 3, 2)
    plt.plot(range(1, num_epochs+1), train_losses_epoch, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average training loss per epoch')
    plt.grid(True)

    # График точности на валидации
    plt.subplot(1, 3, 3)
    plt.plot(range(1, num_epochs+1), val_accuracies, marker='s', color='green', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_plots.png', dpi=150)
    plt.show()
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), "fire_model.pth")
    print("Модель сохранена!")
    return


@app.cell
def _(
    accuracy_score,
    confusion_matrix,
    device,
    f1_score,
    model,
    torch,
    val_loader,
):
    def _():
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        f1_sc = f1_score(all_labels, all_preds)
        print(f"acc: {acc}")
        print(f"f1_score: {f1_sc}")
        print("Confusion Matrix:")
        return print(cm)


    _()
    return


if __name__ == "__main__":
    app.run()
