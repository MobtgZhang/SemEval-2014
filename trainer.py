import torch
from tqdm import tqdm
import torch.utils.data
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
            exit()