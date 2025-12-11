import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from logly import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import OxfordIIITPet

matplotlib.use("Agg")
matplotlib.use("Agg")


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
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask = mask - 1
        mask = torch.clamp(mask, 0, 2)

        return img, mask


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

        logger.info(f"Epoch {epoch + 1}/{config['epochs']}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "best_model.pth")
            logger.success(f"  Saved best model with IoU: {best_iou:.4f}")

    return history, best_iou


def hyperparameter_search(train_dataset, val_dataset, device):
    search_space = {
        "lr": [1e-4, 5e-4, 1e-3],
        "batch_size": [8, 16],
        "weight_decay": [1e-5, 1e-4],
    }

    results = []

    for lr in search_space["lr"]:
        for batch_size in search_space["batch_size"]:
            for weight_decay in search_space["weight_decay"]:
                logger.info(f"\n{'=' * 60}")
                logger.info(
                    f"Testing: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}"
                )
                logger.info(f"{'=' * 60}\n")

                config = {
                    "lr": lr,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "epochs": 10,
                }

                train_loader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
                )

                model = UNet(in_channels=3, num_classes=3).to(device)
                history, best_iou = train_model(
                    model, train_loader, val_loader, config, device
                )

                results.append(
                    {"config": config, "best_val_iou": best_iou, "history": history}
                )

    best_result = max(results, key=lambda x: x["best_val_iou"])

    logger.info(f"\n{'=' * 60}")
    logger.success("Best configuration:")
    logger.info(json.dumps(best_result["config"], indent=2))
    logger.success(f"Best validation IoU: {best_result['best_val_iou']:.4f}")
    logger.info(f"{'=' * 60}\n")

    return best_result, results


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

    logger.info("\nTest Set Evaluation:")
    logger.success(f"Mean IoU: {mean_iou:.4f}")
    logger.info("\nPer-class IoU:")
    class_names = ["Background", "Pet", "Border"]
    class_iou_values = {}
    for cls, name in enumerate(class_names):
        if class_ious[cls]:
            cls_iou = np.mean(class_ious[cls])
            logger.info(f"  {name}: {cls_iou:.4f}")
            class_iou_values[name] = cls_iou

    return mean_iou, class_iou_values


def create_diagrams(history, hp_results, class_iou_values, best_config):
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history["train_iou"], label="Train IoU", marker="o")
    axes[0, 1].plot(epochs, history["val_iou"], label="Val IoU", marker="s")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("IoU")
    axes[0, 1].set_title("Training and Validation IoU")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    hp_configs = [
        f"lr={r['config']['lr']}\nbs={r['config']['batch_size']}\nwd={r['config']['weight_decay']}"
        for r in hp_results
    ]
    hp_ious = [r["best_val_iou"] for r in hp_results]
    axes[1, 0].barh(range(len(hp_ious)), hp_ious)
    axes[1, 0].set_yticks(range(len(hp_ious)))
    axes[1, 0].set_yticklabels(hp_configs, fontsize=8)
    axes[1, 0].set_xlabel("Validation IoU")
    axes[1, 0].set_title("Hyperparameter Search Results")
    axes[1, 0].grid(True, axis="x")

    classes = list(class_iou_values.keys())
    class_scores = list(class_iou_values.values())
    bars = axes[1, 1].bar(
        classes, class_scores, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )
    axes[1, 1].set_ylabel("IoU")
    axes[1, 1].set_title("Per-Class IoU on Test Set")
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, axis="y")
    for bar, score in zip(bars, class_scores):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{score:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "training_results.png", dpi=300, bbox_inches="tight")
    logger.success(
        f"Saved training results diagram to {output_dir / 'training_results.png'}"
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    config_text = "Best Hyperparameters:\n\n"
    for key, value in best_config.items():
        config_text += f"{key}: {value}\n"

    ax.text(
        0.5,
        0.5,
        config_text,
        ha="center",
        va="center",
        fontsize=14,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.axis("off")
    plt.savefig(output_dir / "best_config.png", dpi=300, bbox_inches="tight")
    logger.success(f"Saved best config diagram to {output_dir / 'best_config.png'}")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_root = "./data"
    Path(data_root).mkdir(exist_ok=True)

    logger.info("Loading datasets...")
    train_dataset = AugmentedPetDataset(
        root=data_root, split="trainval", img_size=256, augment=True
    )
    test_dataset = AugmentedPetDataset(
        root=data_root, split="test", img_size=256, augment=False
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    logger.info(
        f"Train size: {len(train_subset)}, Val size: {len(val_subset)}, Test size: {len(test_dataset)}"
    )

    logger.info("\nStarting hyperparameter search...")
    best_result, hp_results = hyperparameter_search(train_subset, val_subset, device)

    logger.info("\nTraining final model with best hyperparameters...")
    best_config = best_result["config"]
    best_config["epochs"] = 50

    train_loader = DataLoader(
        train_subset, batch_size=best_config["batch_size"], shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_subset, batch_size=best_config["batch_size"], shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=best_config["batch_size"], shuffle=False, num_workers=4
    )

    final_model = UNet(in_channels=3, num_classes=3).to(device)
    history, best_val_iou = train_model(
        final_model, train_loader, val_loader, best_config, device
    )

    logger.info("\nLoading best model for final evaluation...")
    final_model.load_state_dict(torch.load("best_model.pth"))
    test_iou, class_iou_values = evaluate_model(final_model, test_loader, device)

    results = {
        "best_config": best_config,
        "best_val_iou": float(best_val_iou),
        "test_iou": float(test_iou),
        "class_iou": class_iou_values,
        "timestamp": datetime.now().isoformat(),
    }

    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nFinal Results:")
    logger.success(f"Best validation IoU: {best_val_iou:.4f}")
    logger.success(f"Test IoU: {test_iou:.4f}")
    logger.info("Results saved to training_results.json")

    logger.info("\nGenerating result diagrams...")
    create_diagrams(history, hp_results, class_iou_values, best_config)


if __name__ == "__main__":
    main()
