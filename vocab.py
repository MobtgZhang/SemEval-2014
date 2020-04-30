import csv
class Vocab:
    def __init__(self,lower = False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.lower = lower
    def size(self):
        return len(self.idxToLabel)
    def add(self,label):
        if self.lower:
            label = label.lower()
        if label in self.labelToIdx:
            idx = self.labelToIdx[label]
        else:
            idx = len(self.idxToLabel)
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        return idx
    def getIndex(self,key,default = None):
        if self.lower:
            key = key.lower()
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default
    def getLabel(self,idx,default = None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default
    def __len__(self):
        return self.size()
    def __str__(self):
        return str(self.idxToLabel)
    def __add__(self, other):
        vocab = Vocab()
        for item in other.labelToIdx.keys():
            vocab.add(item)
        for item in self.labelToIdx.keys():
            vocab.add(item)
        return vocab
    def convertSentToIdx(self,sentence):
        sent = sentence.split()
        result = []
        for item in sent:
            index = self.getIndex(item)
            result.append(index)
def make_vocab(filename):
    vocab = Vocab()
    with open(filename) as f:
        reader = csv.reader(f,delimiter="\t")
        flag = False
        for row in reader:
            if not flag:
                flag = True
                continue
            sentA = row[1].split()
            sentB = row[2].split()
            for item in sentA:
                vocab.add(item)
            for item in sentB:
                vocab.add(item)
    return vocab
