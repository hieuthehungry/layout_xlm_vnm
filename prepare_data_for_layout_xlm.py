import json
from pprint import pprint
from glob import glob
from PIL import Image
import os

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def extract_text_and_bbox(component, key):
    poly = component["chunk"][0]["coords"]
    x1, y1 = poly[0]
    x3, y3 = poly[2]

    bbox = [x1,y1,x3, y3]
    text = component["value"]
    return text, bbox, key


def get_components_from_one_file(data, width, height):

    texts = []
    bboxes = []
    labels = []


    for key in data.keys():
        if data[key]["value_type"] != "TABLE":
            text, bbox, label = extract_text_and_bbox(data[key], key)
            texts.append(text)
            bboxes.append(normalize_box(bbox, width, height))
            labels.append(label)

        else:
            for row in data[key]["value"]:
                for small_key in row["value"].keys():
                    text, bbox, label = extract_text_and_bbox(row["value"][small_key], small_key)
                    texts.append(text)
                    bboxes.append(normalize_box(bbox, width, height))
                    labels.append(label)

    return texts, bboxes, labels
        # bbox = data[key]["chunk"][0]["coords"]
        # x1, y1 = bbox[0]
        # x3, y3 = bbox[2]

        # pprint([x1,y1,x3, y3])
        # pprint(data[key]["normalized"])


if __name__ == '__main__':
    import pickle

    # image = Image.open('dataset/image/0.png')
    # print(image._size)
    image_dir = "dataset/image"

    all_texts = []
    all_bboxes = []
    all_labels = []
    for file in glob("dataset/json/*.json"):
        with open(file) as f:
            data = json.load(f)
        image_file_name = os.path.join(image_dir, os.path.basename(file).replace(".json", ".png"))
        image = Image.open(image_file_name)
        width, height = image._size
        texts, bboxes, labels = get_components_from_one_file(data, width, height)
        all_texts.append(texts)
        all_bboxes.append(bboxes)
        all_labels.append(labels)


    unique_labels = []
    # print(all_texts)
    for all_label in all_labels:
        for lab in all_label:
            unique_labels.append(lab)
    # pprint(set(unique_labels))
    
    pprint(all_bboxes)
    with open('train_norm.pkl', 'wb') as t:
        pickle.dump([all_texts, all_labels, all_bboxes], t)


    labels = ['account_amount',
                'account_name',
                'account_number',
                'accounting_currency',
                'accounting_type',
                'address',
                'amount',
                'amount_text',
                'book_number',
                'currency',
                'customer_code',
                'customer_id',
                'id_date',
                'id_place',
                'interest_rate',
                'issue_date',
                'product_name',
                'product_term',
                'receiver_name',
                'ref',
                'user_inputter']