from model import Trainer
from dataset import GTTTDataset
import pandas as pd
from transformers import LayoutLMv2ForTokenClassification, AdamW
from torch.utils.data import DataLoader
from transformers.models.layoutxlm.processing_layoutxlm import LayoutXLMProcessor

train = pd.read_pickle('../misc/train_norm.pkl')
val = pd.read_pickle('../misc/val_norm.pkl')
test = pd.read_pickle('../misc/test_norm.pkl')
all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
processor.feature_extractor.apply_ocr = False




train_dataset = GTTTDataset(annotations=train,
image_dir='/home/coreai/data/db_layout/IDCardChip/21_11_2021/cc_chip_front/RAW_cc_chip_front_20211121',
processor=processor, label_list = all_labels)
val_dataset = GTTTDataset(annotations=val,
image_dir='/home/coreai/data/db_layout/IDCardChip/21_11_2021/cc_chip_front/RAW_cc_chip_front_20211121',
processor=processor, label_list = all_labels)
test_dataset = GTTTDataset(annotations=test,
image_dir='/home/coreai/data/db_layout/IDCardChip/21_11_2021/cc_chip_front/RAW_cc_chip_front_20211121',
processor=processor, label_list = all_labels)

model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutxlm-base', num_labels=len(all_labels))
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2)



trainer = Trainer(model, labels = all_labels, device = "cpu")

print("Begin training")
trainer.train(train_dataloader, epochs = 1)
trainer.save()
trainer.eval(test_dataloader)
