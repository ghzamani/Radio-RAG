from torch.utils.data import Dataset


class RadiologyDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]

        # Keep the image in PIL format
        image = example["image"].convert("RGB")  # Do NOT manually convert to tensor!

        findings = example["findings"] if example["findings"] else ""
        impression = example["impression"] if example["impression"] else ""
        return image, findings, impression


class RadiologyCollator:
    def __init__(self, processor):
        self.image_processor = processor

    def __call__(self, batch):
        images, findings, impressions = zip(*batch)
        pixel_values = self.image_processor(images, return_tensors="pt").pixel_values
        return pixel_values, findings, impressions