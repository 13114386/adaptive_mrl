from __future__ import unicode_literals, print_function, division
import codecs, json
import pickle as Pickle


def loadFromJson(filename):
    with codecs.open(filename,'r',encoding = 'utf-8') as fp:
        data = json.load(fp, strict = False)
    return data

def saveToJson(filename, data):
    with codecs.open(filename,'w',encoding = 'utf-8') as fp:
        json.dump(data, fp, indent=4)

def saveToPKL(filename, data):
    with open(filename,'wb')as f:
        Pickle.dump(data, f)

def loadFromPKL(filename):
    with open(filename,'rb') as f:
        data = Pickle.load(f)
    return data
