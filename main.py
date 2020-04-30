import os
import logging
import time

import random
import torch
import torch.nn as nn
import torch.optim as optim

from config import parse_args
from download import download_sick,download_wordvecs
from model import SICKDataset
from model import load_word_vectors
from model import RNNSimilarity
from model import Constants
from model import Vocab
from model import Metrics
from model import build_vocab
from trainer import Trainer
from draw import draw
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')
def main():
    args = parse_args()
    # logging defination
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    model_name = time.strftime("%Y%m%d%H%M", time.localtime(time.time()))
    log_dir = os.path.join(os.getcwd(), args.logdir, )
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, model_name + ".log")
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    # download dataset
    download_sick(args.datadir)
    download_wordvecs(args.glove)
    # build vocabulary
    filenames = [os.path.join(args.datadir, "SICK_train.txt"),
                 os.path.join(args.datadir, "SICK_trial.txt"),
                 os.path.join(args.datadir, "SICK_test_annotated.txt")]
    build_vocab(filenames, os.path.join(args.datadir, "vocab.txt"))
    # preparing vocabulary
    vocabulary = Vocab(filename=os.path.join(args.datadir, "vocab.txt"),
                       data=[Constants.PAD_WORD, Constants.UNK_WORD,Constants.BOS_WORD, Constants.EOS_WORD])
    logger.info('==> SICK vocabulary size : %d ' % vocabulary.size())
    # preparing dataset
    train_set = SICKDataset(vocabulary,args.seq_len,args.num_classes,os.path.join(args.datadir,"SICK_train.txt"))
    logger.info('==> Size of train data   : %d ' % len(train_set))
    dev_set = SICKDataset(vocabulary,args.seq_len,args.num_classes,os.path.join(args.datadir, "SICK_trial.txt"))
    logger.info('==> Size of dev data   : %d ' % len(dev_set))
    test_set = SICKDataset(vocabulary,args.seq_len,args.num_classes,os.path.join(args.datadir, "SICK_test_annotated.txt"))
    logger.info('==> Size of test data   : %d ' % len(test_set))

    # preparing model
    model = RNNSimilarity(vocab_size=vocabulary.size(),
                          embedding_dim=args.embedding_dim,
                          mem_dim=args.mem_dim,
                          hid_dim=args.hid_dim,
                          num_layers=args.num_layers,
                          rnn_type=args.rnn_type,
                          dropout=args.dropout,
                          bidirectional=args.bidirectional,
                          seq_len=args.seq_len,
                          num_classes=args.num_classes,
                          sparsity=args.sparse,
                          freeze=args.freeze_embed,
                          name = model_name)
    criterion = nn.KLDivLoss()
    # preparing embeddings

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.datadir, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = load_word_vectors(os.path.join(args.glove, 'glove.840B.300d'))
        logger.info('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocabulary.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocabulary.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocabulary.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    model.word_emb.weight.data.copy_(emb)
    # preparing optimizer
    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr, weight_decay=args.wd)
    else:
        raise TypeError("Unknown optimizer type %s"%str(args.optim))
    metrics = Metrics(args.num_classes)
    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer, device)
    best = -float('inf')
    peason_list = []
    mse_list = []
    loss_list = []
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_set)
        train_loss, train_pred,train_target = trainer.test(train_set)
        dev_loss, dev_pred ,dev_target= trainer.test(dev_set)
        test_loss, test_pred ,test_target= trainer.test(test_set)
        train_pearson = metrics.pearson(train_pred, train_target)
        train_mse = metrics.mse(train_pred,train_target)
        logger.info('==> Epoch {}, Train \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch+1, train_loss, train_pearson, train_mse))
        dev_pearson = metrics.pearson(dev_pred, dev_target)
        dev_mse = metrics.mse(dev_pred, dev_target)
        logger.info('==> Epoch {}, Dev \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch+1, dev_loss, dev_pearson, dev_mse))
        test_pearson = metrics.pearson(test_pred,test_target)
        test_mse = metrics.mse(test_pred,test_target)
        logger.info('==> Epoch {}, Test \tLoss: {}\tPearson: {}\tMSE: {}'.format(epoch+1, test_loss, test_pearson, test_mse))

        # drawing data
        peason_list.append((train_pearson,dev_pearson,test_pearson))
        mse_list.append((train_mse,dev_mse,test_mse))
        loss_list.append((train_loss,dev_loss,test_loss))
        if best < test_pearson:
            best = test_pearson
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'pearson': test_pearson, 'mse': test_mse,
                'args': args, 'epoch': epoch
            }
            logger.info('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.logdir, model.name))
    # draw the picture
    draw(peason_list,"Peason value","peason",model_name,args.logdir)
    draw(mse_list,"MSE value","mse",model_name,args.logdir)
    draw(loss_list,"Loss value","loss",model_name,args.logdir)
if __name__ == '__main__':
    main()