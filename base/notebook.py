import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from logly import logger

    logger.info("Starting semantic segmentation pipeline")
    return (logger,)


@app.cell
def _(logger):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision
    import torchvision.transforms as T
    from torchvision.models.segmentation import deeplabv3_resnet50
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import numpy as np
    from pathlib import Path
    import cv2
    from sklearn.model_selection import train_test_split
    import optuna
    from tqdm import tqdm
    import json

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return (
        A,
        DataLoader,
        Dataset,
        Path,
        ToTensorV2,
        cv2,
        deeplabv3_resnet50,
        device,
        json,
        nn,
        np,
        optim,
        optuna,
        torch,
        tqdm,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Task Formulation
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Semantic Segmentation on Pascal VOC 2012**

    Dataset: Pascal VOC 2012 with 20 object classes + background

    Metric: Mean Intersection over Union (mIoU)

    Justification: mIoU measures overlap between predicted and ground truth masks across all classes, making it standard for semantic segmentation evaluation
    """)
    return


@app.cell
def _(A, Dataset, Path, ToTensorV2, cv2):
    class VOCSegmentationDataset(Dataset):
        def __init__(self, root_dir, split="train", transform=None):
            self.root_dir = Path(root_dir)
            self.split = split
            self.transform = transform

            self.images_dir = self.root_dir / "JPEGImages"
            self.masks_dir = self.root_dir / "SegmentationClass"

            split_file = self.root_dir / "ImageSets" / "Segmentation" / f"{split}.txt"
            with open(split_file) as f:
                self.image_ids = [line.strip() for line in f]

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            img_id = self.image_ids[idx]

            img_path = self.images_dir / f"{img_id}.jpg"
            mask_path = self.masks_dir / f"{img_id}.png"

            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            mask = mask.long()
            mask[mask == 255] = 0

            return image, mask


    def get_train_transforms(image_size):
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


    def get_val_transforms(image_size):
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return VOCSegmentationDataset, get_train_transforms, get_val_transforms


@app.cell
def _(mo):
    mo.md("""
    ## Model Architecture
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **DeepLabV3 with ResNet-50 backbone**

    Architecture: DeepLabV3 with Atrous Spatial Pyramid Pooling (ASPP)

    Pretraining: ImageNet-1K pretrained ResNet-50 backbone

    Justification: DeepLabV3 achieves strong performance on semantic segmentation tasks through multi-scale feature extraction with atrous convolutions, balancing accuracy and computational efficiency
    """)
    return


@app.cell
def _(deeplabv3_resnet50, device, logger, nn):
    class SegmentationModel(nn.Module):
        def __init__(self, num_classes=21, pretrained=True):
            super().__init__()
            self.model = deeplabv3_resnet50(
                weights="DEFAULT" if pretrained else None,
                num_classes=num_classes
            )

        def forward(self, x):
            return self.model(x)["out"]


    def create_model(num_classes=21, pretrained=True):
        model = SegmentationModel(num_classes=num_classes, pretrained=pretrained)
        logger.info(f"Created model with {num_classes} classes, pretrained={pretrained}")
        return model.to(device)
    return (create_model,)


@app.cell
def _(mo):
    mo.md("""
    ## Training Configuration
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Fixed hyperparameters:**
    - num_epochs: 50 (sufficient convergence for VOC dataset size)
    - num_classes: 21 (20 objects + background)
    - image_size: 512 (balance between detail and computational cost)
    - pretrained: True (transfer learning accelerates convergence)

    **Tunable hyperparameters:**
    - learning_rate: [1e-5, 1e-3]
    - batch_size: [4, 8, 16]
    - optimizer: [Adam, SGD]
    - weight_decay: [1e-5, 1e-3]

    **Tuning method:** Optuna with TPE sampler (10 trials)
    """)
    return


@app.cell
def _(device, logger, np, torch):
    def calculate_miou(pred, target, num_classes=21):
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        ious = []
        for cls in range(num_classes):
            pred_mask = pred == cls
            target_mask = target == cls

            intersection = np.logical_and(pred_mask, target_mask).sum()
            union = np.logical_or(pred_mask, target_mask).sum()

            if union == 0:
                continue

            iou = intersection / union
            ious.append(iou)

        return np.mean(ious) if ious else 0.0


    def train_epoch(model, dataloader, criterion, optimizer):
        model.train()
        total_loss = 0.0
        total_miou = 0.0

        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred = outputs.argmax(dim=1)
            miou = calculate_miou(pred, masks)
            total_miou += miou

        avg_loss = total_loss / len(dataloader)
        avg_miou = total_miou / len(dataloader)

        return avg_loss, avg_miou


    def validate(model, dataloader, criterion):
        model.eval()
        total_loss = 0.0
        total_miou = 0.0

        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                total_loss += loss.item()

                pred = outputs.argmax(dim=1)
                miou = calculate_miou(pred, masks)
                total_miou += miou

        avg_loss = total_loss / len(dataloader)
        avg_miou = total_miou / len(dataloader)

        logger.info(f"Validation - Loss: {avg_loss:.4f}, mIoU: {avg_miou:.4f}")

        return avg_loss, avg_miou
    return calculate_miou, train_epoch, validate


@app.cell
def _(
    DataLoader,
    VOCSegmentationDataset,
    create_model,
    get_train_transforms,
    get_val_transforms,
    logger,
    nn,
    optim,
    torch,
    tqdm,
    train_epoch,
    validate,
):
    def train_model(
        data_dir,
        learning_rate,
        batch_size,
        optimizer_name,
        weight_decay,
        num_epochs=50,
        image_size=512
    ):
        train_dataset = VOCSegmentationDataset(
            data_dir,
            split="train",
            transform=get_train_transforms(image_size)
        )
        val_dataset = VOCSegmentationDataset(
            data_dir,
            split="val",
            transform=get_val_transforms(image_size)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        model = create_model(num_classes=21, pretrained=True)
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        if optimizer_name == "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_miou = 0.0
        best_model_state = None

        logger.info(f"Starting training - LR: {learning_rate}, BS: {batch_size}, Optimizer: {optimizer_name}")

        for epoch in tqdm(range(num_epochs), desc="Training"):
            train_loss, train_miou = train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_miou = validate(model, val_loader, criterion)

            scheduler.step()

            if val_miou > best_miou:
                best_miou = val_miou
                best_model_state = model.state_dict().copy()
                logger.info(f"Epoch {epoch+1}: New best mIoU: {best_miou:.4f}")

        if best_model_state:
            model.load_state_dict(best_model_state)
            torch.save(best_model_state, "best_model.pth")

        return best_miou, model
    return (train_model,)


@app.cell
def _(mo):
    mo.md("""
    ## Hyperparameter Tuning
    """)
    return


@app.cell
def _(logger, optuna, train_model):
    def objective(trial, data_dir):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        logger.info(f"Trial {trial.number}: LR={learning_rate}, BS={batch_size}, OPT={optimizer_name}")

        best_miou, _ = train_model(
            data_dir=data_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            num_epochs=10
        )

        return best_miou


    def tune_hyperparameters(data_dir, n_trials=10):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, data_dir),
            n_trials=n_trials,
            show_progress_bar=True
        )

        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best mIoU: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study.best_params
    return (tune_hyperparameters,)


@app.cell
def _(mo):
    mo.md("""
    ## Model Evaluation
    """)
    return


@app.cell
def _(
    DataLoader,
    VOCSegmentationDataset,
    calculate_miou,
    device,
    get_val_transforms,
    logger,
    torch,
):
    def evaluate_model(model, data_dir, image_size=512, batch_size=8):
        test_dataset = VOCSegmentationDataset(
            data_dir,
            split="val",
            transform=get_val_transforms(image_size)
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        model.eval()
        total_miou = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                pred = outputs.argmax(dim=1)

                miou = calculate_miou(pred, masks)
                total_miou += miou
                num_batches += 1

        final_miou = total_miou / num_batches
        logger.info(f"Test set mIoU: {final_miou:.4f}")

        return final_miou
    return (evaluate_model,)


@app.cell
def _(mo):
    mo.md("""
    ## Execution Pipeline
    """)
    return


@app.cell
def _(evaluate_model, json, logger, train_model, tune_hyperparameters):
    def run_pipeline(data_dir, tune=True, n_trials=10):
        logger.info("="*50)
        logger.info("Starting Semantic Segmentation Pipeline")
        logger.info("="*50)

        if tune:
            logger.info("Phase 1: Hyperparameter Tuning")
            best_params = tune_hyperparameters(data_dir, n_trials=n_trials)

            with open("best_params.json", "w") as f:
                json.dump(best_params, f, indent=2)
        else:
            with open("best_params.json", "r") as f:
                best_params = json.load(f)

        logger.info("Phase 2: Final Training")
        final_miou, final_model = train_model(
            data_dir=data_dir,
            learning_rate=best_params["learning_rate"],
            batch_size=best_params["batch_size"],
            optimizer_name=best_params["optimizer"],
            weight_decay=best_params["weight_decay"],
            num_epochs=50
        )

        logger.info("Phase 3: Model Evaluation")
        test_miou = evaluate_model(final_model, data_dir)

        logger.info("="*50)
        logger.info(f"Pipeline Complete - Final Test mIoU: {test_miou:.4f}")
        logger.info("="*50)

        return test_miou, final_model
    return (run_pipeline,)


@app.cell
def _(mo):
    data_dir_input = mo.ui.text(
        label="VOC Dataset Path",
        value="/path/to/VOCdevkit/VOC2012"
    )
    tune_checkbox = mo.ui.checkbox(label="Run hyperparameter tuning", value=False)
    n_trials_slider = mo.ui.slider(
        label="Number of tuning trials",
        start=5,
        stop=20,
        value=10,
        show_value=True
    )

    mo.vstack([data_dir_input, tune_checkbox, n_trials_slider])
    return data_dir_input, n_trials_slider, tune_checkbox


@app.cell
def _(data_dir_input, mo, n_trials_slider, run_pipeline, tune_checkbox):
    run_button = mo.ui.run_button(label="Start Training")

    if run_button.value:
        test_miou_result, trained_model = run_pipeline(
            data_dir=data_dir_input.value,
            tune=tune_checkbox.value,
            n_trials=n_trials_slider.value
        )

    run_button
    return


if __name__ == "__main__":
    app.run()
