import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import random
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, random_split
    from torchvision.datasets import OxfordIIITPet
    return (
        DataLoader,
        Dataset,
        Image,
        OxfordIIITPet,
        Path,
        TF,
        mo,
        nn,
        np,
        optim,
        plt,
        random,
        random_split,
        torch,
    )


@app.cell
def _(nn, torch):
    class UNet(nn.Module):
        def __init__(self, in_channels=3, num_classes=3):
            super().__init__()

            self.enc1 = self.conv_block(in_channels, 64)
            self.enc2 = self.conv_block(64, 128)
            self.enc3 = self.conv_block(128, 256)
            self.enc4 = self.conv_block(256, 512)

            self.bottleneck = self.conv_block(512, 1024)

            self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = self.conv_block(1024, 512)

            self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = self.conv_block(512, 256)

            self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = self.conv_block(256, 128)

            self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = self.conv_block(128, 64)

            self.out = nn.Conv2d(64, num_classes, 1)

            self.pool = nn.MaxPool2d(2)

        def conv_block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            enc1 = self.enc1(x)
            enc2 = self.enc2(self.pool(enc1))
            enc3 = self.enc3(self.pool(enc2))
            enc4 = self.enc4(self.pool(enc3))

            bottleneck = self.bottleneck(self.pool(enc4))

            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat([dec4, enc4], dim=1)
            dec4 = self.dec4(dec4)

            dec3 = self.upconv3(dec4)
            dec3 = torch.cat([dec3, enc3], dim=1)
            dec3 = self.dec3(dec3)

            dec2 = self.upconv2(dec3)
            dec2 = torch.cat([dec2, enc2], dim=1)
            dec2 = self.dec2(dec2)

            dec1 = self.upconv1(dec2)
            dec1 = torch.cat([dec1, enc1], dim=1)
            dec1 = self.dec1(dec1)

            return self.out(dec1)
    return (UNet,)


@app.cell
def _(Dataset, Image, OxfordIIITPet, TF, np, random, torch):
    class AugmentedPetDataset(Dataset):
        def __init__(self, root, split="trainval", img_size=256, augment=True):
            self.dataset = OxfordIIITPet(
                root=root, split=split, target_types="segmentation", download=True
            )
            self.img_size = img_size
            self.augment = augment and split == "trainval"

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, mask = self.dataset[idx]

            img = TF.resize(img, [self.img_size, self.img_size])
            mask = TF.resize(
                mask, [self.img_size, self.img_size], interpolation=Image.NEAREST
            )

            if self.augment:
                if random.random() > 0.5:
                    img = TF.hflip(img)
                    mask = TF.hflip(mask)

                if random.random() > 0.5:
                    img = TF.vflip(img)
                    mask = TF.vflip(mask)

                if random.random() > 0.5:
                    angle = random.uniform(-30, 30)
                    img = TF.rotate(img, angle)
                    mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

                if random.random() > 0.5:
                    img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))

                if random.random() > 0.5:
                    img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))

            img = TF.to_tensor(img)
            img = TF.normalize(
                img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
            mask = mask - 1
            mask = torch.clamp(mask, 0, 2)

            return img, mask
    return (AugmentedPetDataset,)


@app.function
def calculate_iou(pred, target, num_classes=3):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append((intersection / union).item())

    return ious


@app.cell
def _(np):
    def train_epoch(model, loader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        all_ious = []

        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            batch_ious = calculate_iou(preds, masks)
            all_ious.append(batch_ious)

        avg_loss = total_loss / len(loader)
        mean_iou = np.nanmean(all_ious)

        return avg_loss, mean_iou
    return (train_epoch,)


@app.cell
def _(np, torch):
    def validate(model, loader, criterion, device):
        model.eval()
        total_loss = 0
        all_ious = []

        with torch.no_grad():
            for imgs, masks in loader:
                imgs, masks = imgs.to(device), masks.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, masks)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                batch_ious = calculate_iou(preds, masks)
                all_ious.append(batch_ious)

        avg_loss = total_loss / len(loader)
        mean_iou = np.nanmean(all_ious)

        return avg_loss, mean_iou
    return (validate,)


@app.cell
def _(mo, nn, optim, torch, train_epoch, validate):
    def train_model(model, train_loader, val_loader, config, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, verbose=False
        )

        best_iou = 0
        history = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}

        for epoch in range(config["epochs"]):
            train_loss, train_iou = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_iou = validate(model, val_loader, criterion, device)

            history["train_loss"].append(train_loss)
            history["train_iou"].append(train_iou)
            history["val_loss"].append(val_loss)
            history["val_iou"].append(val_iou)

            scheduler.step(val_iou)

            mo.output.append(
                mo.md(
                    f"**Epoch {epoch + 1}/{config['epochs']}:** Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}"
                )
            )

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), "best_model.pth")

        return history, best_iou
    return (train_model,)


@app.cell
def _(mo):
    mo.md("""
    # Semantic Segmentation on Oxford-IIIT Pet Dataset

    This notebook implements semantic segmentation using U-Net architecture on the Oxford-IIIT Pet dataset.

    ## Configuration
    """)
    return


@app.cell
def _(mo, torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "./data"

    mo.md(f"**Device:** {device}")
    return data_root, device


@app.cell
def _(mo):
    mo.md("""
    ## Load Datasets
    """)
    return


@app.cell
def _(AugmentedPetDataset, Path, data_root, mo, random_split):
    Path(data_root).mkdir(exist_ok=True)

    train_dataset = AugmentedPetDataset(
        root=data_root, split="trainval", img_size=256, augment=True
    )
    test_dataset = AugmentedPetDataset(
        root=data_root, split="test", img_size=256, augment=False
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    mo.md(f"""
    **Dataset sizes:**
    - Train: {len(train_subset)}
    - Validation: {len(val_subset)}
    - Test: {len(test_dataset)}
    """)
    return test_dataset, train_subset, val_subset


@app.cell
def _(mo):
    mo.md("""
    ## Hyperparameter Configuration
    """)
    return


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(
        start=1e-5, stop=1e-2, step=1e-5, value=5e-4, label="Learning Rate"
    )
    batch_size_select = mo.ui.dropdown([4, 8, 16, 32], value=16, label="Batch Size")
    weight_decay_slider = mo.ui.slider(
        start=1e-6, stop=1e-3, step=1e-6, value=1e-4, label="Weight Decay"
    )
    epochs_slider = mo.ui.slider(start=5, stop=100, step=5, value=30, label="Epochs")

    mo.hstack(
        [
            mo.vstack([lr_slider, weight_decay_slider]),
            mo.vstack([batch_size_select, epochs_slider]),
        ]
    )
    return batch_size_select, epochs_slider, lr_slider, weight_decay_slider


@app.cell
def _(mo):
    run_training = mo.ui.run_button(label="Start Training")
    run_training
    return (run_training,)


@app.cell
def _(
    DataLoader,
    UNet,
    batch_size_select,
    device,
    epochs_slider,
    lr_slider,
    mo,
    run_training,
    train_model,
    train_subset,
    val_subset,
    weight_decay_slider,
):
    mo.stop(not run_training.value, mo.md("Click **Start Training** to begin"))

    config = {
        "lr": lr_slider.value,
        "batch_size": batch_size_select.value,
        "weight_decay": weight_decay_slider.value,
        "epochs": epochs_slider.value,
    }

    train_loader = DataLoader(
        train_subset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_subset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )

    model = UNet(in_channels=3, num_classes=3).to(device)
    history, best_val_iou = train_model(model, train_loader, val_loader, config, device)

    mo.md(f"### Training Complete! Best Validation IoU: **{best_val_iou:.4f}**")
    return config, history, model


@app.cell
def _(history, mo, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, history["train_iou"], label="Train IoU", marker="o")
    axes[1].plot(epochs_range, history["val_iou"], label="Val IoU", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("Training and Validation IoU")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Evaluate on Test Set
    """)
    return


@app.cell
def _(DataLoader, config, test_dataset):
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=4
    )
    return (test_loader,)


@app.cell
def _(device, mo, model, np, test_loader, torch):
    def evaluate_model(model, test_loader, device):
        model.eval()
        all_ious = []
        class_ious = [[] for _ in range(3)]

        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                outputs = model(imgs)
                preds = outputs.argmax(dim=1)

                batch_ious = calculate_iou(preds, masks)
                all_ious.append(batch_ious)

                for cls in range(3):
                    if not np.isnan(batch_ious[cls]):
                        class_ious[cls].append(batch_ious[cls])

        mean_iou = np.nanmean(all_ious)

        class_names = ["Background", "Pet", "Border"]
        class_iou_values = {}
        for cls, name in enumerate(class_names):
            if class_ious[cls]:
                cls_iou = np.mean(class_ious[cls])
                class_iou_values[name] = cls_iou

        return mean_iou, class_iou_values

    model.load_state_dict(torch.load("best_model.pth"))
    test_iou, class_iou_values = evaluate_model(model, test_loader, device)

    mo.md(f"""
    ### Test Set Results

    **Mean IoU: {test_iou:.4f}**

    Per-class IoU:
    - Background: {class_iou_values.get("Background", 0):.4f}
    - Pet: {class_iou_values.get("Pet", 0):.4f}
    - Border: {class_iou_values.get("Border", 0):.4f}
    """)
    return (class_iou_values,)


@app.cell
def _(class_iou_values, mo, plt):
    fig_test, ax_test = plt.subplots(figsize=(8, 5))

    classes = list(class_iou_values.keys())
    class_scores = list(class_iou_values.values())
    bars = ax_test.bar(classes, class_scores, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax_test.set_ylabel("IoU")
    ax_test.set_title("Per-Class IoU on Test Set")
    ax_test.set_ylim([0, 1])
    ax_test.grid(True, axis="y")

    for bar, score in zip(bars, class_scores):
        height = bar.get_height()
        ax_test.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{score:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    mo.mpl.interactive(fig_test)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Sample Predictions
    """)
    return


@app.cell
def _(device, mo, model, np, plt, test_dataset, torch):
    def show_predictions(model, dataset, device, num_samples=3):
        model.eval()
        fig_pred, axes_pred = plt.subplots(
            num_samples, 3, figsize=(12, 4 * num_samples)
        )

        indices = np.random.choice(len(dataset), num_samples, replace=False)

        with torch.no_grad():
            for i, idx in enumerate(indices):
                img, mask = dataset[idx]

                img_input = img.unsqueeze(0).to(device)
                output = model(img_input)
                pred = output.argmax(dim=1).squeeze().cpu().numpy()

                img_display = img.permute(1, 2, 0).cpu().numpy()
                img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array(
                    [0.485, 0.456, 0.406]
                )
                img_display = np.clip(img_display, 0, 1)

                axes_pred[i, 0].imshow(img_display)
                axes_pred[i, 0].set_title("Input Image")
                axes_pred[i, 0].axis("off")

                axes_pred[i, 1].imshow(mask.cpu().numpy(), cmap="tab10", vmin=0, vmax=2)
                axes_pred[i, 1].set_title("Ground Truth")
                axes_pred[i, 1].axis("off")

                axes_pred[i, 2].imshow(pred, cmap="tab10", vmin=0, vmax=2)
                axes_pred[i, 2].set_title("Prediction")
                axes_pred[i, 2].axis("off")

        plt.tight_layout()
        return fig_pred

    predictions_fig = show_predictions(model, test_dataset, device, num_samples=3)
    mo.mpl.interactive(predictions_fig)
    return


if __name__ == "__main__":
    app.run()
