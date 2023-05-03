import torch 
from transformers import BertTokenizerFast

class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output

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


def evaluate_one_text(model, sentence):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_offsets_mapping=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    print(sentence)
    print(prediction_label)

    #https://huggingface.co/learn/nlp-course/chapter6/3
    """
    probabilities = torch.nn.functional.softmax(logits[0], dim=-1)[0].tolist()
    predictions = logits[0].argmax(dim=-1)[0].tolist()
    print(predictions)
    print(text["offset_mapping"].unsqueeze(0))#TODO: include in results
    results = []
    tokens = text.tokens()
    for idx, pred in enumerate(predictions):
        label = ids_to_labels[pred]
        if label != "O":#TODO: filter CLS,PAD
            results.append(
                {"entity": label, "score": probabilities[idx][pred], "word": tokens[idx]}
            )
    print(results)
    """

if __name__ ==  '__main__':

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    ids_to_labels = torch.load("ids_to_labels.pt")
    print(ids_to_labels)

    model = torch.load("bert_ner_ft.pt",map_location=torch.device('cpu'))
    model.eval()

    evaluate_one_text(model,"As a surveyor, I want to be able to log into a system and enter information about the geospatial coordinates, building type, and characteristics for each survey I perform in a day, so that all of my survey data is stored in one central location and is easily accessible for analysis and reporting. Acceptance Criteria: The surveyor is able to log into the system using their unique credentials. The surveyor is able to enter the geospatial coordinates (latitude and longitude) of the building being surveyed.")
