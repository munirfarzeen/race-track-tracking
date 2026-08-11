from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


class FolderMaskProvider:
    def __init__(self, mask_dir):
        self.mask_dir = Path(mask_dir)

    def mask_for(self, image, frame_name):
        mask_path = self.mask_dir / f"{Path(frame_name).stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Missing mask, copying frame unchanged: {mask_path}")
        return mask


class ModelMaskProvider:
    def __init__(
        self,
        checkpoint_path,
        device="cuda",
        threshold=0.45,
        morph_size=13,
        max_components=2,
        min_component_area=900,
        min_component_height_ratio=0.25,
        min_component_aspect=1.6,
        max_component_width_ratio=0.18,
        no_geometry_filter=False,
    ):
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.model, self.image_size = load_model(checkpoint_path, self.device)
        self.threshold = threshold
        self.morph_size = morph_size
        self.max_components = max_components
        self.min_component_area = min_component_area
        self.min_component_height_ratio = min_component_height_ratio
        self.min_component_aspect = min_component_aspect
        self.max_component_width_ratio = max_component_width_ratio
        self.no_geometry_filter = no_geometry_filter

    def mask_for(self, image, frame_name):
        del frame_name
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        tensor = (
            torch.from_numpy(resized.astype(np.float32) / 255.0)
            .permute(2, 0, 1)[None]
            .to(self.device)
        )

        with torch.no_grad():
            prob = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()

        mask = (prob >= self.threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = filter_prediction_geometry(mask, self)

        if self.morph_size > 0 and np.any(mask):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.morph_size, self.morph_size),
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = filter_prediction_geometry(mask, self)

        return mask


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SmallUNet(base_channels=checkpoint.get("base_channels", 32)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("image_size", 512)


def filter_prediction_geometry(mask, config):
    if config.no_geometry_filter:
        return mask

    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    candidates = []

    min_height = max(1, int(height * config.min_component_height_ratio))
    max_width = max(1, int(width * config.max_component_width_ratio))

    for label_id in range(1, num_labels):
        _, _, w, h, area = stats[label_id]
        aspect = h / max(1, w)

        if area < config.min_component_area:
            continue
        if h < min_height:
            continue
        if w > max_width:
            continue
        if aspect < config.min_component_aspect:
            continue

        vertical_score = float(area) * min(aspect, 5.0)
        candidates.append((vertical_score, label_id))

    if not candidates:
        return np.zeros_like(mask)

    keep_labels = {
        label_id
        for _, label_id in sorted(candidates, reverse=True)[: config.max_components]
    }
    return np.where(np.isin(labels, list(keep_labels)), 255, 0).astype(np.uint8)

