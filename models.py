import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import SAGEConv
from IPython import embed



class TagGNN_QI(nn.Module):
    def __init__(self, args):
        super(TagGNN_QI, self).__init__()
        self.n_query = args.n_query
        self.n_item = args.n_item
        self.n_tag = args.n_tag
        self.initial_weight = args.initial_weight
        self.embedding_dim = args.embedding_dim
        self.weight_decay = args.weight_decay
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

        self.id_embeddings = nn.Embedding(self.n_query + self.n_item + self.n_tag, self.embedding_dim)
        self.word_embeddings = nn.Embedding(args.vocab_size + 1, self.embedding_dim, padding_idx=args.vocab_size)
        
        self.gnn1 = SAGEConv(self.embedding_dim, self.embedding_dim, 'mean', feat_drop=args.drop_rate)
        self.gnn2 = SAGEConv(self.embedding_dim, self.embedding_dim, 'mean', feat_drop=args.drop_rate)
        self.fc = nn.Linear(self.embedding_dim, self.n_tag) # classification layer
        self.init_embeddings()

    def init_embeddings(self):
        nn.init.normal_(self.id_embeddings.weight, std=self.initial_weight)
        nn.init.normal_(self.word_embeddings.weight, std=self.initial_weight)


    def forward(self, g):
        h = torch.sum(self.word_embeddings(g.ndata['wids']), dim=1) / g.ndata['len'].view(-1, 1)
        h = F.leaky_relu(self.gnn1(g, h))
        return self.gnn2(g, h)

    def test_forward(self, g):
        h = self.forward(g)
        logits = self.fc(h[self.n_query : self.n_item + self.n_query])
        return logits
        
    def cal_loss(self, g, train_labels, all_train_items):
        h = self.forward(g)
        logits = self.fc(h[self.n_query : self.n_item + self.n_query])
        loss = self.loss_func(logits, train_labels)

        loss = loss[torch.tensor(all_train_items) - self.n_query].mean()
        
        if self.weight_decay > 0:
            loss += self.weight_decay * self.reg_loss()
        return loss   

    def reg_loss(self):
        return 1 / 2 * torch.sum(self.id_embeddings.weight.data ** 2) + \
            torch.sum(self.word_embeddings.weight.data ** 2)

    # def cal_loss(self, g, train_labels, all_train_items):
    #     h = self.forward(g)
    #     item_embs = h[self.n_query : self.n_item + self.n_query]
    #     tag_embs = h[-self.n_tag : ]
        
    #     logits = item_embs.mm(tag_embs.t())
    #     loss = self.loss_func(logits, train_labels)
    #     loss = loss[torch.tensor(all_train_items) - self.n_query].mean()
    #     return loss   


    