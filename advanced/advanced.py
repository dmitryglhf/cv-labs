import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    from pathlib import Path

    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    import torch
    import torch.nn as nn
    from logly import logger
    from monai.data import DataLoader, Dataset
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        RandAdjustContrastd,
        RandFlipd,
        RandGaussianNoised,
        RandRotated,
        RandShiftIntensityd,
        Resized,
        ScaleIntensityRanged,
        Spacingd,
        ToTensord,
    )
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    return (
        Compose,
        CropForegroundd,
        DataLoader,
        Dataset,
        DiceLoss,
        DiceMetric,
        EnsureChannelFirstd,
        LoadImaged,
        Orientationd,
        Path,
        RandAdjustContrastd,
        RandFlipd,
        RandGaussianNoised,
        RandRotated,
        RandShiftIntensityd,
        Resized,
        ScaleIntensityRanged,
        Spacingd,
        ToTensord,
        logger,
        nib,
        nn,
        np,
        plt,
        re,
        torch,
        tqdm,
        train_test_split,
    )


@app.cell
def _(logger, torch):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return (device,)


@app.cell
def _(Path, logger, re):
    # Data loading utilities
    DATA_DIR = Path("./data/imagecas")

    def tryint(s):
        try:
            return int(s)
        except (ValueError, TypeError):
            return s

    def alphanum_key(s):
        return [tryint(c) for c in re.split("([0-9]+)", s)]

    def sort_nicely(l):
        l.sort(key=alphanum_key)

    # Load dataset
    pattern = re.compile(r"^\d+\.(label|img)\.")
    all_files = list(DATA_DIR.rglob("*.nii.gz"))
    filtered_files = [f.name for f in all_files if pattern.match(f.name)]
    sort_nicely(filtered_files)

    images = [
        str(DATA_DIR / file) for file in filtered_files if file.endswith("img.nii.gz")
    ]
    segs = [
        str(DATA_DIR / file) for file in filtered_files if file.endswith("label.nii.gz")
    ]

    dataset = [{"image": img, "label": lbl} for img, lbl in zip(images, segs)]
    logger.info(f"Total samples: {len(dataset)}")
    return (dataset,)


@app.cell
def _(dataset, nib, plt):
    # Visualize sample data
    if len(dataset) > 0:
        sample = dataset[0]
        img = nib.load(sample["image"]).get_fdata()
        seg = nib.load(sample["label"]).get_fdata()

        slice_z = img.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img[:, :, slice_z], cmap="gray")
        axes[0].set_title("CT Image")
        axes[0].axis("off")

        axes[1].imshow(seg[:, :, slice_z], cmap="jet")
        axes[1].set_title("Segmentation Mask")
        axes[1].axis("off")

        axes[2].imshow(img[:, :, slice_z], cmap="gray")
        axes[2].imshow(seg[:, :, slice_z], cmap="jet", alpha=0.3)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.gca()
    return


@app.cell
def _(dataset, logger, train_test_split):
    # Train/Val/Test split
    train_val, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_val, test_size=0.25, random_state=42)

    logger.info(
        f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
    )
    return test_data, train_data, val_data


@app.cell
def _(
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandRotated,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    np,
):
    # Define transforms
    base_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    augmentation_transforms = Compose(
        [
            RandRotated(
                keys=["image", "label"],
                range_x=np.pi / 6,
                range_y=np.pi / 6,
                range_z=np.pi / 6,
                prob=0.3,
                mode=("bilinear", "nearest"),
                align_corners=True,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.3),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.3),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.3),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
            RandGaussianNoised(keys=["image"], std=0.05, prob=0.3),
        ]
    )

    final_transforms = Compose(
        [
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            Resized(
                keys=["image", "label"],
                spatial_size=(128, 128, 64),
                mode=("trilinear", "nearest"),
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    train_transforms = Compose(
        [base_transforms, augmentation_transforms, final_transforms]
    )
    val_test_transforms = Compose([base_transforms, final_transforms])
    return train_transforms, val_test_transforms


@app.cell
def _(
    DataLoader,
    Dataset,
    test_data,
    torch,
    train_data,
    train_transforms,
    val_data,
    val_test_transforms,
):
    # Create datasets and dataloaders
    train_dataset = Dataset(data=train_data, transform=train_transforms)
    val_dataset = Dataset(data=val_data, transform=val_test_transforms)
    test_dataset = Dataset(data=test_data, transform=val_test_transforms)

    batch_size = 1
    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, train_loader, val_loader


@app.cell
def _(plt, train_loader):
    # Visualize augmented sample
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        image = batch["image"][0][0]
        label = batch["label"][0][0]

        slice_idx = image.shape[0] // 2
        image_slice = image[slice_idx]
        label_slice = label[slice_idx]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(image_slice.cpu().numpy(), cmap="gray")
        plt.title("Augmented Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(label_slice.cpu().numpy(), cmap="gray")
        plt.title("Augmented Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(image_slice.cpu().numpy(), cmap="gray")
        plt.imshow(label_slice.cpu().numpy(), cmap="Reds", alpha=0.5)
        plt.title("Overlay")
        plt.axis("off")

        plt.tight_layout()
        plt.gca()
    return


@app.cell
def _(nn, torch):
    # UNet3D Model
    class UNet3D(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, features=(16, 32, 64, 128)):
            super().__init__()
            self.downs = nn.ModuleList()
            self.ups = nn.ModuleList()
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

            for feature in features:
                self.downs.append(self._block(in_channels, feature))
                in_channels = feature

            self.bottleneck = self._block(features[-1], features[-1] * 2)

            for feature in reversed(features):
                self.ups.append(
                    nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
                )
                self.ups.append(self._block(feature * 2, feature))

            self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

        def forward(self, x):
            skip_connections = []

            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

            x = self.bottleneck(x)
            skip_connections = skip_connections[::-1]

            for i in range(0, len(self.ups), 2):
                x = self.ups[i](x)
                skip_connection = skip_connections[i // 2]
                if x.shape != skip_connection.shape:
                    x = self._resize(x, skip_connection.shape)
                x = torch.cat((skip_connection, x), dim=1)
                x = self.ups[i + 1](x)

            return torch.sigmoid(self.final_conv(x))

        def _block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        def _resize(self, x, target_shape):
            diff = [target_shape[i] - x.shape[i] for i in range(2, 5)]
            pad = [d // 2 for d in diff]
            return nn.functional.pad(
                x,
                (
                    pad[2],
                    diff[2] - pad[2],
                    pad[1],
                    diff[1] - pad[1],
                    pad[0],
                    diff[0] - pad[0],
                ),
            )
    return (UNet3D,)


@app.cell
def _(DiceLoss, DiceMetric, UNet3D, logger, plt, torch, tqdm):
    # UNet3D Artery Segmentation Wrapper
    class UNet3DArterySegmentation:
        def __init__(
            self,
            learning_rate=1e-4,
            batch_size=2,
            num_epochs=50,
            features=(16, 32, 64, 128),
            device=None,
        ):
            self.lr = learning_rate
            self.batch_size = batch_size
            self.num_epochs = num_epochs
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            self.model = UNet3D(in_channels=1, out_channels=1, features=features).to(
                self.device
            )
            self.loss_fn = DiceLoss(sigmoid=True)
            self.metric = DiceMetric(include_background=False, reduction="mean")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            self.train_losses = []
            self.val_losses = []

        def fit(self, train_loader, val_loader):
            for epoch in range(self.num_epochs):
                self.model.train()
                train_loss = 0.0

                for batch in tqdm(
                    train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs} - Training"
                ):
                    images = batch["image"].to(self.device)
                    masks = batch["label"].to(self.device)

                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)
                self.train_losses.append(avg_train_loss)

                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(
                        val_loader,
                        desc=f"Epoch {epoch + 1}/{self.num_epochs} - Validation",
                    ):
                        images = batch["image"].to(self.device)
                        masks = batch["label"].to(self.device)

                        outputs = self.model(images)
                        loss = self.loss_fn(outputs, masks)
                        val_loss += loss.item()

                    avg_val_loss = val_loss / len(val_loader)
                    self.val_losses.append(avg_val_loss)

                logger.info(
                    f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
                )

            self._plot_losses()

        def predict(self, test_loader):
            self.model.eval()
            dice_scores = []

            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Testing"):
                    images = batch["image"].to(self.device)
                    masks = batch["label"].to(self.device)

                    outputs = self.model(images)
                    outputs_bin = (outputs > 0.5).float()

                    self.metric.reset()
                    self.metric(outputs_bin, masks)
                    dice = self.metric.aggregate().item()
                    dice_scores.append(dice)

            avg_dice = sum(dice_scores) / len(dice_scores)
            logger.info(f"Average Dice score on test set: {avg_dice:.4f}")
            return avg_dice

        def _plot_losses(self):
            plt.figure(figsize=(8, 5))
            plt.plot(self.train_losses, label="Train Loss")
            plt.plot(self.val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Dice Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.show()

        def save_model(self, path="unet3d_artery_segmentation.pth"):
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")

        def load_model(self, path="unet3d_artery_segmentation.pth"):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"Model loaded from {path}")
    return (UNet3DArterySegmentation,)


@app.cell
def _(UNet3DArterySegmentation, device):
    # Initialize model
    model = UNet3DArterySegmentation(
        learning_rate=1e-4,
        batch_size=2,
        num_epochs=10,
        features=(16, 32, 64, 128),
        device=device,
    )
    return (model,)


@app.cell
def _(plt, torch):
    # Visualization function
    def visualize_predictions(model, test_loader, num_samples=3, slice_idx="middle"):
        model.model.eval()
        samples_shown = 0

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(model.device)
                labels = batch["label"].to(model.device)
                preds = model.model(images)
                preds_bin = (preds > 0.5).float()

                for i in range(images.shape[0]):
                    if samples_shown >= num_samples:
                        return

                    image = images[i, 0].cpu().numpy()
                    label = labels[i, 0].cpu().numpy()
                    pred = preds_bin[i, 0].cpu().numpy()

                    if slice_idx == "middle":
                        slice_id = image.shape[0] // 2
                    else:
                        slice_id = slice_idx

                    img_slice = image[slice_id]
                    label_slice = label[slice_id]
                    pred_slice = pred[slice_id]

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(img_slice, cmap="gray")
                    axs[0].set_title("CT Slice")
                    axs[0].axis("off")

                    axs[1].imshow(label_slice, cmap="gray")
                    axs[1].set_title("Ground Truth")
                    axs[1].axis("off")

                    axs[2].imshow(pred_slice, cmap="gray")
                    axs[2].set_title("Prediction")
                    axs[2].axis("off")

                    plt.tight_layout()
                    plt.show()

                    samples_shown += 1
    return (visualize_predictions,)


@app.cell
def _(model, train_loader, val_loader):
    model.fit(train_loader, val_loader)
    return


@app.cell
def _(logger, model, test_loader):
    avg_dice = model.predict(test_loader)
    logger.info(f"Avg dice: {avg_dice}")
    return


@app.cell
def _(model, num_samples_slider, test_loader, visualize_predictions):
    visualize_predictions(model, test_loader, num_samples=num_samples_slider.value)
    return


@app.cell
def _(model, model_path_input):
    model.save_model(model_path_input.value)
    return


if __name__ == "__main__":
    app.run()
