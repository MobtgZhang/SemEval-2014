import os
import csv
class SICKDataset:
    def __init__(self,filename):
        # foramt (sentA,sentB,score,sentiment_label)
        self.data = []
        self.load_file(filename)
    def load_file(self,filename):
        with open(filename,mode="r",encoding="utf-8") as f:
            reader = csv.reader(f,delimiter="\t")
            flag = False
            for row in reader:
                if not flag:
                    flag = True
                    continue
                sentA = row[1]
                sentB = row[2]
                score = float(row[3])
                label = row[4]
                self.data.append((sentA,sentB,score,label))
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)
