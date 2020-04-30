import torch
from tqdm import tqdm
import torch.utils.data
from torch.autograd import Variable
class Trainer:
    def __init__(self,args,model,criterion,optimizer,device):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0
    def train(self,dataset):
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        data_loader = torch.utils.data.DataLoader(
            dataset,batch_size=self.args.batch_size,
            sampler=sampler,
            num_workers=self.args.data_workers,
            pin_memory= self.args.cuda,
        )
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        for item in tqdm(data_loader,desc="Training epoch "+str(self.epoch+1)+''):
            lsents ,rsents,scores = item
            lsents, rsents, scores = Variable(lsents).to(self.device) ,Variable(rsents).to(self.device),Variable(scores).to(self.device)
            output = self.model(lsents,rsents)
            loss = self.criterion(output, scores)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.epoch += 1
        return total_loss / len(dataset)
    def test(self,dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            targets = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                lsent,rsent,score_dist = dataset[idx]
                lsent, rsent, score_dist = Variable(lsent.reshape(1,dataset.seq_len)).to(self.device),\
                                           Variable(rsent.reshape(1,dataset.seq_len)).to(self.device), \
                                           Variable(score_dist).to(self.device)
                output = self.model(lsent,rsent)
                loss = self.criterion(output, score_dist)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                score_dist = score_dist.squeeze().to('cpu')
                predictions[idx] = torch.dot(indices, torch.exp(output))
                targets[idx] = torch.dot(indices,torch.exp(score_dist))
            return total_loss / len(dataset), predictions,targets