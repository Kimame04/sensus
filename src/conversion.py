from os.path import exists

import gdown as gdown
import torch
import torchtext.vocab as torch_vocab
import numpy as np

from conversion_config import Config
from model import CtrlGenModel

from nltk import pos_tag, word_tokenize

def makeVocab(filename):
    word_to_id = {}
    id_to_word = {}
    idx = 0
    label_list = []
    for line in open(filename):
        fields = line.split()
        if len(fields) > 1:
            label = ' '.join(fields[:])
        else:
            label = fields[0]
        if label not in label_list:
            word_to_id[label] = idx
            id_to_word[idx] = label
            idx += 1
            label_list.append(label)
    return word_to_id, id_to_word


def make_pretrain_embeddings(glove, id_to_word, emb_dim):
    weights_matrix = []
    for i in range(len(id_to_word)):
        try:
            weights_matrix.append(glove.vectors[glove.stoi[id_to_word[i]]])
        except KeyError:
            weights_matrix.append(np.random.normal(scale=0.6, size=(emb_dim,)))

    new_weight = torch.FloatTensor(weights_matrix)
    return new_weight


def makeData(srcFile, labelFile, word_to_id, glove, save_file, hidden_size, if_nn_save=False, if_glove=True, if_nn=True,
             if_shuff=True, if_gender=True):
    Dataset = []
    input = {}

    original_text = []
    original_id = []
    original_label = []
    original_length = []

    original_hidden = []
    original_hidden2 = []
    temp_tensor = torch.FloatTensor(hidden_size)
    if if_nn == True:
        if if_gender == True:
            original_nn = []
        else:
            original_nn = np.load(save_file)
    else:
        original_nn = []
    original_nn_vector = []

    text_line = []
    max_length = 0

    print('Processing %s & %s ...' % (srcFile, labelFile))

    srcSet = srcFile
    labelSet = labelFile

    for i in range(len(srcSet)):
        id_line = []
        temp_nn_vector = []
        temp_nn = []
        temp_hidden = []
        if if_nn == False and if_nn_save == True:
            original_nn.append([])
        srcSet[i] = srcSet[i].strip()
        if if_nn_save == False:
            temp_res = pos_tag(word_tokenize(srcSet[i]))
            for tube in temp_res:
                if tube[1] == 'NN':
                    temp_nn.append(tube[0])
        if if_nn_save == False:
            original_nn.append(temp_nn)
        if if_nn == True:
            for word in original_nn[i]:
                try:
                    temp_nn_vector.append(glove.vectors[glove.stoi[word]])
                except KeyError:
                    a = 1
                    print("Unknown words: ", word)
        original_label.append(int(labelSet[i].strip()))
        if temp_nn_vector == []:
            original_hidden.append([int(labelSet[i].strip())] * 300)
            original_hidden2.append([1 - int(labelSet[i].strip())] * 300)
        else:
            temp_tensor = torch.zeros(300)
            for vector in temp_nn_vector:
                temp_tensor += vector
            temp_tensor = temp_tensor / len(temp_nn_vector)
            original_hidden.append([yi.item() for yi in temp_tensor])
            original_hidden2.append(original_hidden[i])
        if i % 5000 == 0:
            print("now ", i)
        text_line = ["<BOS>"] + srcSet[i].split() + ["<EOS>"]
        original_text.append(text_line)
        original_length.append(len(text_line))
        original_nn_vector.append(temp_nn_vector)

        if len(text_line) > max_length:
            max_length = len(text_line)
        for word in text_line:
            try:
                id = word_to_id[word]
            except KeyError:
                id = 3
            id_line.append(id)
        original_id.append(id_line)
        if if_nn_save == False:
            if i % 1000 == 0:
                print("total: {0} now: {1}".format(len(srcSet), i))
                print("nn: ", original_nn[i])
    if if_nn_save == False:
        save_nn_vector = np.array(original_nn)
        np.save(save_file, save_nn_vector)
    if if_shuff:
        print('... shuffling sentences')
        perm = torch.randperm(len(original_text))
        original_id = [original_id[idx] for idx in perm]
        original_label = [original_label[idx] for idx in perm]
        original_length = [original_length[idx] for idx in perm]
        original_text = [original_text[idx] for idx in perm]
        original_hidden = [original_hidden[idx] for idx in perm]
        original_hidden2 = [original_hidden2[idx] for idx in perm]
    if if_nn == True:
        original_nn = [original_nn[idx] for idx in perm]
        original_nn_vector = [original_nn_vector[idx] for idx in perm]

    print('... pedding')
    for i in range(len(original_text)):
        if original_length[i] < max_length:
            for j in range(max_length - original_length[i]):
                original_text[i].append("<PAD>")
                original_id[i].append(0)
    Dataset = {"text": original_text, "length": original_length, "text_ids": original_id, "labels": original_label,
               "nn": original_nn, "nn_vector": original_nn_vector, "hidden": original_hidden,
               "hidden2": original_hidden2}
    return Dataset, max_length


def makeBatch(Dataset,batch_size):
    Dataset_total = []
    text = []
    length = []
    text_ids = []
    labels = []
    nn = []
    nn_vector = []
    hidden = []
    hidden2 = []
    unk = []
    temp = {"text":text,"length":length,"text_ids":text_ids,"labels":labels,"nn":nn,"nn_vector":nn_vector,"unk":unk,"hidden":hidden,"hidden2":hidden2}
    for i in range(len(Dataset['text'])):
        temp["text"].append(Dataset['text'][i])
        temp["length"].append(Dataset['length'][i])
        temp["text_ids"].append(Dataset['text_ids'][i])
        temp["labels"].append(Dataset['labels'][i])
        temp["nn"].append(Dataset["nn"][i])
        temp["nn_vector"].append(Dataset["nn_vector"][i])
        temp["hidden"].append(Dataset["hidden"][i])
        temp["hidden2"].append(Dataset["hidden2"][i])
        if ((i+1) % batch_size == 0) or (i == len(Dataset['text']) - 1):
            store = {"text":[row for row in temp['text']],"length":[row for row in temp['length']],"text_ids":[row for row in temp['text_ids']],"labels":[row for row in temp['labels']],"nn":[row for row in temp['nn']],"nn_vector":[row for row in temp['nn_vector']],"hidden":[row for row in temp['hidden']],"hidden2":[row for row in temp['hidden2']]}
            Dataset_total.append(store)
            temp['text'].clear()
            temp['length'].clear()
            temp['text_ids'].clear()
            temp['labels'].clear()
            temp['nn'].clear()
            temp['nn_vector'].clear()
            temp['hidden'].clear()
            temp['hidden2'].clear()
    for i in range(len(Dataset_total)):
        raw_inputs = Dataset_total[i]
        input_text_ids = torch.LongTensor(raw_inputs["text_ids"])
        input_labels = torch.LongTensor(raw_inputs["labels"])
        input_length = torch.IntTensor(raw_inputs["length"])
        input_hidden = torch.FloatTensor(raw_inputs['hidden'])
        input_hidden2 = torch.FloatTensor(raw_inputs['hidden2'])
        #input_nn_vector = torch.FloatTensor(raw_inputs["nn_vector"])
        inputs = {"text":raw_inputs["text"],"labels":input_labels,"length":input_length,"text_ids":input_text_ids,"nn":raw_inputs["nn"],"nn_vector":raw_inputs["nn_vector"],"hidden":input_hidden,"hidden2":input_hidden2}
        Dataset_total[i] = inputs
    return Dataset_total


def convert(textArr, objective_truth):
    config = Config()

    gpu = -1
    datapath = "./subj_dataset/"
    '''if torch.cuda.is_available() and gpu == -1:
        gpu = 0
    if gpu != -1:
        torch.cuda.set_device(gpu)
        print("Using GPU: " + str(gpu))'''

    # TODO: lock UI here
    dic_glove = torch_vocab.GloVe(name='twitter.27B', dim=100, cache="./cached-GloVe/27B.glove")
    word_to_id = {}
    id_to_word = {}
    word_to_id, id_to_word = makeVocab(datapath + 'vocab')
    vocab_size = len(word_to_id)
    print("Vocab size", len(word_to_id))

    # Make the weight matrix which will be used as pretrained embedding.
    weights_matrix = make_pretrain_embeddings(dic_glove, id_to_word, config.model['embedder']['dim'])

    glove = torch_vocab.GloVe(name='840B', dim=300, cache="./cached-GloVe/840B.glove")

    # train_nn_file = datapath + "train_nn.npy"
    # dev_nn_file = datapath + "dev_nn.npy"
    test_nn_file= ""

    max_length = 1000
    max_length_dev = 1000

    if objective_truth: t = ["1" for x in range(len(textArr))]
    else: t = ["0" for x in range(len(textArr))]
    Dataset_test, _ = makeData(textArr, t, word_to_id, glove, test_nn_file, hidden_size=config.hidden_size, if_nn=False,
                               if_nn_save=True, if_shuff=False)

    Dataset_test_total = makeBatch(Dataset_test,1)

    model = CtrlGenModel(config, vocab_size, len(textArr), weights_matrix)
    # model = model.cuda()
    if exists("./checkpoint/checkpoint.model"):
        model.load_state_dict(torch.load("./checkpoint/checkpoint.model", map_location=torch.device('cpu')))
    else:
        gdown.download("https://drive.google.com/file/d/1gW6taIS1LQJ71qW0y1tlZkgi55EIvNDi/view?usp=sharing",
                       './checkpoint/checkpoint.model', quiet=False)
    # TODO: unlock UI here

    output = ""
    for mn in range(len(Dataset_test_total)):
        eval_output,soft_outputs,probs,classes,_ = model(Dataset_test_total[mn],max_length)
        result2 = torch.argmax(soft_outputs.transpose(0,1),2)
        p = ""
        q = ""
        m = ""
        #p = [id_to_word[j.item()] for j in eval_data[79]["text_ids"][i][:]]
        for j in Dataset_test_total[mn]["text"][0][1:]:
            if j == "<EOS>":
                break
            p = p+" "+j
        for j in result2[0][:]:
            if j == 2:
                break
            m = m+ " "+ id_to_word[j.item()]
        output += m + "\n"
    return output


if __name__ == "__main__":

    print(convert([input("Enter text: ")],True))
    print("done")
