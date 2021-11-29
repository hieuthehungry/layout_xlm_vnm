import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class GTTTDataset(Dataset):
    """GTTT dataset."""
    def __init__(self, annotations, image_dir, label_list, processor=None):
        """
        Args:
        annotations (List[List]): List of lists containing the word-level annotations (words, labels, boxes).
        image_dir (string): Directory with all the document images.
        processor (LayoutLMv2Processor): Processor to prepare the text + image.
        """
        self.words, self.labels, self.boxes, self.file_paths = annotations
        self.image_dir = image_dir
        self.image_file_names = [os.path.basename(f) for f in self.file_paths]
        self.processor = processor
        self.label_list = label_list
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}
    def __len__(self):
        return len(self.image_file_names)
    
    def __getitem__(self, idx):
        # first, take an image
        item = self.image_file_names[idx]
        image = Image.open(os.path.join(self.image_dir, item)).convert("RGB")
        # get word-level annotations
        words = self.words[idx]
        boxes = self.boxes[idx]
        word_labels = self.labels[idx]
        assert len(words) == len(boxes) == len(word_labels)

        word_labels = [self.label2id[label] for label in word_labels]
        # use processor to prepare everything
        encoded_inputs = self.processor(image, words, boxes=boxes, word_labels=word_labels,
        padding="max_length", truncation=True,
        return_tensors="pt", max_length = 512,
        return_token_type_ids = True)

        # remove batch dimension
        for k,v in encoded_inputs.items():
            encoded_inputs[k] = v.squeeze()
        assert encoded_inputs.input_ids.shape == torch.Size([512])
        assert encoded_inputs.attention_mask.shape == torch.Size([512])
        assert encoded_inputs.token_type_ids.shape == torch.Size([512])
        assert encoded_inputs.bbox.shape == torch.Size([512, 4])
        assert encoded_inputs.image.shape == torch.Size([3, 224, 224])
        assert encoded_inputs.labels.shape == torch.Size([512])

        return encoded_inputs