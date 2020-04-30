import argparse
def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch RNNs for Sentence Similarity')
    # data arguments
    parser.add_argument('--datadir', default='data',help='path to dataset')
    parser.add_argument('--glove', default='F:\\natural language processing toolkit\glove\\',help='directory with GLOVE embeddings')
    parser.add_argument('--logdir', default='log',help='path to log files')
    # model arguments
    parser.add_argument('--embedding_dim', default=300, type=int,help='Size of input word vector')
    parser.add_argument('--mem_dim', default=150, type=int,help='Size of RNNs mem dim')
    parser.add_argument('--hid_dim', default=50, type=int,help='Size of classifier MLP')
    parser.add_argument('--num_layers',default=1,type=int,help='Size of RNN num_layers ')
    parser.add_argument('--rnn_type', default="GRU", type=str, help='Type of RNN ')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout of RNN ')
    parser.add_argument('--num_classes', default=5, type=int,help='Number of classes in dataset')
    parser.add_argument('--bidirectional',action='store_true',help='Whether using bidirectional RNNs')
    parser.add_argument('--seq_len',default=40,help='Size of sentence length.')
    parser.add_argument('--freeze_embed', action='store_true',help='Freeze word embeddings')
    # training arguments
    parser.add_argument('--epochs', default=15, type=int,help='number of total epochs to run')
    parser.add_argument('--batchsize', default=25, type=int,help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.01, type=float,metavar='LR', help='initial learning rate')
    parser.add_argument('--wd', default=1e-4, type=float,help='weight decay (default: 1e-4)')
    parser.add_argument('--sparse', action='store_true',help='Enable sparsity for embeddings,incompatible with weight decay')
    parser.add_argument('--optim', default='adagrad',help='optimizer (default: adagrad)')
    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    parser.set_defaults(cuda=True)
    args = parser.parse_args()
    return args