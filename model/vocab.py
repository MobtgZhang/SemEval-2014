import csv
class Vocab:
    def __init__(self,filename = None,data = None,lower = False):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.special = []
        self.lower = lower
        if data is not None:
            self.addSpecials(data)
        if filename is not None:
            self.loadFile(filename)

    def size(self):
        return len(self.idxToLabel)
    # Load entries from a file.
    def loadFile(self, filename):
        idx = 0
        for line in open(filename, 'r', encoding='utf8', errors='ignore'):
            token = line.rstrip('\n')
            self.add(token)
            idx += 1
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
    def addSpecial(self,label):
        idx = self.add(label)
        self.special += [idx]
    def addSpecials(self,data):
        for label in data:
            self.addSpecial(label)
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
    def convertToIdx(self,labels,unkWord,bosWord=None,eosWord=None):
        vec = []
        if bosWord is not None:
            vec += [self.getIndex(bosWord)]
        unk = self.getIndex(unkWord)
        vec+=[self.getIndex(label,default=unk) for label in labels]
        if eosWord is not None:
            vec+= [self.getIndex(eosWord)]
        return vec
    def convertToLabels(self,idx,stop):
        labels = []
        for item in idx:
            labels+= [self.getLabel(idx)]
            if item == stop:
                break
        return labels
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
