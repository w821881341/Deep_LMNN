# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 10:58:34 2018

@author: nsde
"""
#%%
import torch

#%%
class params:
    pass

#%%
class lmnn_loss(torch.nn.Module):
    def __init__(self, mu, margin):
        super(lmnn_loss, self).__init__()
        self.mu = mu
        self.margin = margin
        
    def forward(self, X, y):
        batch_size = X.shape[0]
        
        # Calcualte distances
        D = self.calcD(X, X)
        
        # Calculate target neighbours
        same_label = (y[:,None] == y[None,:].t())
        same_label = same_label - torch.eye(batch_size)
        tN = torch.nonzero(same_label)
        n_tN = tN.shape[0]
        
        # Calculate all index tuplets
        tN_rep = tN.repeat(1, batch_size).reshape(n_tN*batch_size, 2)
        imp = torch.arange(batch_size).repeat(batch_size).flatten()
        all_comb = torch.cat([tN_rep, imp], dim=1)
        cond = (all_comb[:,1]==all_comb[:,2])
        
        # Gather distances
        D_pull = torch.gather(D, tN)
        D_tn = torch.gather(D, all_comb[:2])
        D_im = torch.gather(D, all_comb[::2])
        
        # Calculate pull and push
        zero = torch.Tensor([0.0])
        pull_loss = D_pull.sum()
        push_loss = (1-cond)*torch.max(zero, self.margin+D_tn-D_im)
        loss = (1-self.mu) * pull_loss + self.mu * push_loss
        return loss
    
    def calcD(self, X1, X2):
        N = X1.shape[0]
        M = X2.shape[0]
        X1_trans = torch.reshape(self.extractor(X1), N, -1)
        X2_trans = torch.reshape(self.extractor(X2), M, -1)
        term1 = torch.norm(X1_trans, dim=2) ** 2
        term2 = torch.norm(X2_trans, dim=2) ** 2
        term3 = -2.0*torch.matmul(X1_trans, X2_trans.t())
        summ = term1[:,None] + term2[None,:] + term3
        return torch.max(0.0, summ)

#%%
class lmnn(object):
    def __init__(self, extractor, gpu=False, dir_loc='./logs'):
        # Set parameters
        self.params = params()
        self.params.dir_loc = dir_loc
        self.params.gpu = gpu
        
        # Set extractor
        assert type(extractor)==torch.nn.Module, \
            ''' Expects the extractor to be a torch.nn.Module class '''
        self.extractor = extractor
    
    #%%
    def compile(self, k=1, learning_rate = 1e-4, mu=0.5, margin=1):
        assert len(self._layers)!=0, '''Layers must be added with the 
                lmnn.add() method before this function is called '''
        self.built = True
        
        # Save parameters
        self.params.k = k
        self.params.lr = learning_rate
        self.params.mu = mu
        self.params.margin = margin
        
        # Set extractor
        self.extractor = torch.nn.Sequential(*self._layers)
        
        # Set distance calculator
        self.calcD = calcD(extractor=self.extractor)
        
        # Loss calculator
        self.loss = lmnn_loss(calcD=self.calcD,
                              mu=self.params.mu, 
                              margin=self.params.margin)
        
        self.optimizer = torch.optim.Adam(self.extractor.parameters(), 
                                          lr=self.params.learning_rate)
    
    #%%    
    def fit(self, Xtrain, ytrain, num_epochs=100, batch_size=50):
        self._assert_if_build()
        
        # Save parameters
        self.params.num_epochs = num_epochs
        self.params.batch_size = batch_size
        
        # Data loader
        train_loader = self._create_data_loader(Xtrain, ytrain)
        
        for i in range(self.params.num_epochs):
            ## Training
            self.extractor.train()
            for i, data in enumerate(train_loader):
                # Get batch data
                Xbatch, ybatch = data
                
                # Zero gradient
                self.optimizer.zero_grad()
                
                # Extract features
                features = self.extractor(Xbatch)
                
                # Compute loss
                loss = self.loss(features, ybatch)
                
                # Backpropagate loss
                loss.backward()
                
                # Gradient update
                optimizer.step()
                
                print(i, loss.items())
                
            ## Evaluation
            self.extractor.eval()
    
    #%% 
    def transform(self, X):
        return self.extractor(X)
       
    #%%
    def predict(self,):
        self._assert_if_build()
    
    #%%
    def evaluate(self,):
        self._assert_if_build()
    
    #%%
    def summary(self):
        print(self.extractor)
    
    #%%
    def _create_data_loader(self, *args):
        dataset = torch.utils.data.TensorDataset(*[torch.Tensor(a) for a in args])
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.params.batch_size)
        return loader
        
    #%%
    def _assert_if_compiled(self):
        assert self.built, '''Model is not compiled, call lmnn.compile() 
                before this function is called '''

#%%
if __name__ == '__main__':
    model = lmnn(extractor=)

