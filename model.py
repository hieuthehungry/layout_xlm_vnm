from transformers.models.layoutxlm.processing_layoutxlm import LayoutXLMProcessor
from transformers import LayoutLMv2Processor
from torch.utils.data import DataLoader, Dataset
from transformers import LayoutLMv2ForTokenClassification, AdamW
import torch
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score)
from PIL import Image

class Trainer:
    def __init__(self, model: LayoutLMv2ForTokenClassification, labels, device= None):
        self.model = model
        self.labels = labels
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for idx, label in enumerate(self.labels)}

    def train(self, train_dataloader: DataLoader, epochs: int = 4):
        global_step = 0
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.model.train() 
        for epoch in range(epochs):  
            print("Epoch:", epoch)
            for batch in tqdm(train_dataloader):
                    # get the inputs;
                    input_ids = batch['input_ids'].to(self.device)
                    bbox = batch['bbox'].to(self.device)
                    image = batch['image'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    token_type_ids = batch['token_type_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # print(bbox)
                    # forward + backward + optimize
                    outputs = self.model(input_ids=input_ids,
                                    bbox=bbox,
                                    image=image,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    labels=labels) 
                    loss = outputs.loss
                    # print loss every 100 steps
                    if global_step % 100 == 0:
                        print(f"Loss after {global_step} steps: {loss.item()}")

                    loss.backward()
                    optimizer.step()
                    global_step += 1

        print("Training finished!!!")

    def eval(self, val_dataloader: DataLoader):
        self.model.eval()
        # forward pass

        preds_val = None
        out_label_ids = None

        # put model in evaluation mode
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(self.device)
                bbox = batch['bbox'].to(self.device)
                image = batch['image'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # forward pass
                outputs = self.model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                                token_type_ids=token_type_ids, labels=labels)
                
                if preds_val is None:
                    preds_val = outputs.logits.detach().cpu().numpy()
                    out_label_ids = batch["labels"].detach().cpu().numpy()
                else:
                    preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(
                        out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0
                    )

        val_result, class_report = self.results_test(preds_val, out_label_ids, self.labels)
        print("Overall results:", val_result)   
        print(class_report)
    
    def results_test(self, preds, out_label_ids, labels):
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != -100:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
        return results, classification_report(out_label_list, preds_list)



    def predict(self, filepath: str, words, boxes, processor = None):
        image = Image.open(filepath).convert("RGB")

        encoding = processor(image, words, boxes=boxes, 
                                        padding="max_length", truncation=True, 
                                        return_tensors="pt", max_length = 512,
                                        return_token_type_ids = True)

        outputs = self.model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'],
                token_type_ids=encoding['token_type_ids'], bbox=encoding['bbox'],
                image=encoding['image'])


        prediction_indices = outputs.logits.argmax(-1).squeeze().tolist()
        print(prediction_indices)

        prediction_indices = outputs.logits.argmax(-1).squeeze().tolist()
        predictions = [self.id2label[label] for gt, label in zip(encoding['labels'].squeeze().tolist(), prediction_indices) if gt != -100]
        return predictions


    def save(self, path = "Checkpoints/"):
        self.model.save_pretrained(path)
    def load(self, path):
        self.model.from_pretrained(path)
