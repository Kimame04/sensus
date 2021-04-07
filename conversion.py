from os.path import exists

import gdown as gdown
import torch
import torchtext.vocab as torch_vocab
import numpy as np

from conversion_config import Config
from model import CtrlGenModel


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


def convert(textArr):
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

    #train_nn_file = datapath + "train_nn.npy"
    #dev_nn_file = datapath + "dev_nn.npy"

    max_length = 1000
    max_length_dev = 1000

    model = CtrlGenModel(config, vocab_size, len(textArr), weights_matrix)
    #model = model.cuda()
    if exists("./checkpoint/checkpoint.model"):
        model.load_state_dict(torch.load("./checkpoint/checkpoint.model"))
    else:
        gdown.download("https://drive.google.com/file/d/1gW6taIS1LQJ71qW0y1tlZkgi55EIvNDi/view?usp=sharing",
                       './checkpoint/checkpoint.model', quiet=False)
    # TODO: unlock UI here

    output = ""
    for mn in range(len(textArr)):
        eval_output, soft_outputs, probs, classes, _ = model(textArr[mn], max_length)
        result2 = torch.argmax(soft_outputs.transpose(0, 1), 2)
        p = ""
        m = ""
        for j in textArr[mn]:
            p += " " + j
        for j in result2[0][:]:
            if j == 2:
                break
            m = m + " " + id_to_word[j.item()]
        # f1.write(p+'\n')
        # f1.write(m+'\n')
        output += m + "\n"
    return output


if __name__ == "__main__":
    convert([input("Enter text: ")])
