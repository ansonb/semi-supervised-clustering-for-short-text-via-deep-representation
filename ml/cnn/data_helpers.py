import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# def load_data_and_labels(positive_data_file, negative_data_file):
def load_data_and_labels(fileList):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    x_text = []
    y = []

    noOfLabels = len(fileList)
    curLabel = 0
    for file in fileList:
        body = list(open(file, "r").readlines())
        for line in body:
            mail = line.strip()
            mail = re.sub(r'\w*@\w*.\w{2,3}', '', mail) #remove emails
            mail = re.sub(r'[\d]+', '', mail) #remove numbers
            mail = re.sub(r'\w*[\d@$\t-]\w*', '', mail) #remove words containing numbers and special characters
            mail = re.sub(r'[-_|]', ' ', mail) #remove any dashes
            mail = re.sub(r'[.]', '. ', mail) #append . with space
            #remove any extra spaces
            mail = mail.split()
            mail = " ".join(mail)

            x_text.append(mail)

            arr = []
            for _ in range(noOfLabels):
                arr.append(0)
            arr[curLabel] = 1

            if len(y) == 0:
                y = [arr]
            else:
                y = np.concatenate([y,[arr]],0)  

        curLabel += 1
                      

    print(x_text)
    print(y)

    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True, ref = None):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    if ref is not None:
        ref['num_batches_per_epoch'] = num_batches_per_epoch
        print(str(num_batches_per_epoch))
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
