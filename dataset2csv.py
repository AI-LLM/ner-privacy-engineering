import csv

def read_conll2003_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = []
        sentence = []
        labels = []

        for line in lines:
            if line.strip() == '':
                if sentence:
                    data.append((' '.join(sentence), ' '.join(labels)))
                    sentence = []
                    labels = []
            else:
                tokens = line.strip().split()
                word, label = tokens[0], tokens[-1]
                sentence.append(word)
                labels.append(label)
                
        if sentence:
            data.append((' '.join(sentence), ' '.join(labels)))

        return data

def write_to_csv(file_path, data):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'labels'])
        for row in data:
            writer.writerow(row)

if __name__ == '__main__':
    #input_file = 'dataset/conll_train_aug_mr_v3_clean.txt'
    #output_file = 'conll_train_aug_mr_v3_clean.csv'
    #input_file = 'dataset/conll_test_v3_clean.txt'
    #output_file = 'conll_test_v3_clean.csv'
    input_file = 'dataset/conll_dev_v3_clean.txt'
    output_file = 'conll_dev_v3_clean.csv'

    conll_data = read_conll2003_file(input_file)
    write_to_csv(output_file, conll_data)
