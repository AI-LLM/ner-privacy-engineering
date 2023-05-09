import os

import pandas as pd
import torch 
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification, AutoConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD

pretrained_model_name = "bert-base-cased"

def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df):

        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels


class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

        self.bert = BertForTokenClassification.from_pretrained(pretrained_model_name, num_labels=len(unique_labels))

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output


def train_loop(model, df_train, df_val, device):

    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][val_label[i] != -100]
              label_clean = val_label[i][val_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_val += acc
              total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')

def evaluate(model, df_test, device):

    test_dataset = DataSequence(df_test)

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)

    total_acc_test = 0.0

    for test_data, test_label in test_dataloader:

            test_label = test_label.to(device)
            mask = test_data['attention_mask'].squeeze(1).to(device)

            input_id = test_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, test_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][test_label[i] != -100]
              label_clean = test_label[i][test_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')

def align_word_ids(texts):
  
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def evaluate_one_text(model, sentence, device):

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    print(sentence)
    print(prediction_label)

if __name__ ==  '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    df_train = pd.read_csv('conll_train_aug_mr_v3_clean.csv')
    df_train.head()
    df_val = pd.read_csv('conll_dev_v3_clean.csv')
    df_test = pd.read_csv('conll_test_v3_clean.csv')

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

    label_all_tokens = False

    labels = [i.split() for i in df_train['labels'].values.tolist()]
    unique_labels = set()

    for lb in labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
    ids_to_labels = {v: k for v, k in enumerate(unique_labels)}


    config = AutoConfig.from_pretrained(pretrained_model_name)
    config.label2id = labels_to_ids
    config.id2label = ids_to_labels
    config._num_labels = len(ids_to_labels)
    with open('config.json', 'w') as fp:
        fp.write(config.to_json_string())
        fp.close()

    LEARNING_RATE = 5e-3
    EPOCHS = 5
    BATCH_SIZE = 2

    model = BertModel()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if not use_cuda and torch.has_mps:
        device = torch.device('mps')
        model.to(device)
    print("Using Device: {}".format(device))

    """Using Device: mps (MacBook Air M1 2020 16G, MacOS 13.3.1)
100%|█| 486/486 [05:45<00:00,  1.41it/s]
Epochs: 1 | Loss:  0.043 | Accuracy:  0.726 | Val_Loss:  0.036 | Accuracy:  0.739
100%|█| 486/486 [05:39<00:00,  1.43it/s]
Epochs: 2 | Loss:  0.036 | Accuracy:  0.746 | Val_Loss:  0.034 | Accuracy:  0.739
100%|█| 486/486 [05:41<00:00,  1.42it/s]
Epochs: 3 | Loss:  0.034 | Accuracy:  0.746 | Val_Loss:  0.032 | Accuracy:  0.739
100%|█| 486/486 [05:51<00:00,  1.38it/s]
Epochs: 4 | Loss:  0.032 | Accuracy:  0.749 | Val_Loss:  0.031 | Accuracy:  0.747
100%|█| 486/486 [06:01<00:00,  1.35it/s]
Epochs: 5 | Loss:  0.031 | Accuracy:  0.761 | Val_Loss:  0.030 | Accuracy:  0.759
Using Device: cpu
100%|█| 486/486 [14:19<00:00,  1.77s/it]
Epochs: 1 | Loss:  0.622 | Accuracy:  0.805 | Val_Loss:  0.495 | Accuracy:  0.853
100%|█| 486/486 [14:10<00:00,  1.75s/it]
Epochs: 2 | Loss:  0.436 | Accuracy:  0.864 | Val_Loss:  0.441 | Accuracy:  0.869
100%|█| 486/486 [14:22<00:00,  1.78s/it]
Epochs: 3 | Loss:  0.368 | Accuracy:  0.886 | Val_Loss:  0.426 | Accuracy:  0.870
100%|█| 486/486 [14:21<00:00,  1.77s/it]
Epochs: 4 | Loss:  0.319 | Accuracy:  0.900 | Val_Loss:  0.403 | Accuracy:  0.877
100%|█| 486/486 [14:27<00:00,  1.78s/it]
Epochs: 5 | Loss:  0.283 | Accuracy:  0.914 | Val_Loss:  0.392 | Accuracy:  0.883
Test Accuracy:  0.859
Using Device: mps
100%|█| 486/486 [05:34<00:00,  1.45it/s]
Epochs: 1 | Loss:  0.043 | Accuracy:  0.728 | Val_Loss:  0.035 | Accuracy:  0.739
100%|█| 486/486 [05:32<00:00,  1.46it/s]
Epochs: 2 | Loss:  0.036 | Accuracy:  0.746 | Val_Loss:  0.034 | Accuracy:  0.739
100%|█| 486/486 [05:23<00:00,  1.50it/s]
Epochs: 3 | Loss:  0.034 | Accuracy:  0.746 | Val_Loss:  0.032 | Accuracy:  0.739
100%|█| 486/486 [05:23<00:00,  1.50it/s]
Epochs: 4 | Loss:  0.032 | Accuracy:  0.748 | Val_Loss:  0.031 | Accuracy:  0.744
100%|█| 486/486 [05:33<00:00,  1.46it/s]
Epochs: 5 | Loss:  0.031 | Accuracy:  0.757 | Val_Loss:  0.029 | Accuracy:  0.758
Test Accuracy:  0.733
    """

    if use_cuda:
        model = model.cuda()

    train_loop(model, df_train, df_val, device)

    torch.save(model, "bert_ner_aug_mr.pt")

    evaluate(model, df_test, device)

    evaluate_one_text(model,"As a surveyor, I want to be able to log into a system and enter information about the geospatial coordinates, building type, and characteristics for each survey I perform in a day, so that all of my survey data is stored in one central location and is easily accessible for analysis and reporting. Acceptance Criteria: The surveyor is able to log into the system using their unique credentials. The surveyor is able to enter the geospatial coordinates (latitude and longitude) of the building being surveyed.", device)
