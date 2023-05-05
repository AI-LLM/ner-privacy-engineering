import torch 
from transformers import BertTokenizerFast, AutoConfig
import os.path

pretrained_model_name = "bert-base-cased"

class BertModel(torch.nn.Module):

    def __init__(self):

        super(BertModel, self).__init__()

    def forward(self, input_id, mask, label):

        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)

        return output


def export(model, sentence):
    #https://huggingface.co/blog/convert-transformers-to-onnx
    text = tokenizer(sentence, padding='max_length', max_length = 512, truncation=True, return_tensors="pt")

    inputs_onnx = dict(text)
    symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
    torch.onnx.export(
           model,
           tuple(inputs_onnx.values()),
           f="bert_ner_ft.onnx",
           input_names=['input_ids', 'attention_mask', 'token_type_ids'],
           output_names=['logits'],
           dynamic_axes={'input_ids': symbolic_names,  # variable lenght axes
                        'attention_mask': symbolic_names,
                        'token_type_ids': symbolic_names},
           do_constant_folding=True,
           opset_version=13)

if __name__ ==  '__main__':

    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)

    ids_to_labels = torch.load("ids_to_labels.pt")

    if not os.path.isfile('config.json'):
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.label2id = {v: k for k, v in ids_to_labels.items()}
        config.id2label = ids_to_labels
        config._num_labels = len(ids_to_labels)
        with open('config.json', 'w') as fp:
            fp.write(config.to_json_string())
            fp.close()

    model = torch.load("bert_ner_ft.pt",map_location=torch.device('cpu'))
    model.eval()

    export(model,"dummy input")
