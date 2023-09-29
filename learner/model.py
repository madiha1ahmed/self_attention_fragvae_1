import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention
#from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, embed_size,
                 hidden_size, num_heads, latent_size,
                 dropout, use_gpu):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        #self.hidden_layers = hidden_layers
        self.latent_size = latent_size
        self.use_gpu = use_gpu
        
        # Multi-head self-attention mechanism
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads = num_heads, dropout=dropout)
        self.attention2hidden = nn.Linear(in_features=embed_size, out_features=hidden_size)
        self.hidden2mean = nn.Linear(in_features=hidden_size, out_features=latent_size)
        self.hidden2logv = nn.Linear(in_features=hidden_size, out_features=latent_size)


    def forward(self, inputs, embeddings, lengths):
        '''
        INPUTS:
        inputs: tensor, batch_size x sequence of indices
            Example: ( 0 for '<PAD>', 1 for '<SOS>', 2 for '<EOS>')
            tensor([[    1, 20025,    79,   660,   216],
                    [    1,  8866,  3705,    10,     0],
                    [    1,    20,     5,  1091,     0],
                    [    1,    44,  4866,     3,     0],
                    [    1,    26,   233, 13176,     0],
                    [    1,    31,     5,   356,     0],
                    [    1,     8,  8179,  1517,     0],
                    [    1,   153, 11176,    67,     0]
                    [    1,   153, 11176,    67,     0],
                    [    1,     6,   109,     3,     0],
                    [    1,    11,     4,     0,     0], device='cuda:0')
        embeddings: tensor, batch_size x L x embed_size
        lengths: list, example:  [5, 4, 4, 4, 4, 4, 4, 4, 4, 3]
        OUTPUTS:
        latent_sample: num_layers x batch x latent_size
        # mean, std: same
        '''
        attn_output, _ = self.self_attention(embeddings, embeddings, embeddings) # self-attention
        attn_output = self.attention2hidden(attn_output)

        # The rest of your encoder's forward method
        state = attn_output.mean(dim=1)
        
        mean = self.hidden2mean(state)
        logvar = self.hidden2logv(state)
        std = torch.exp(0.5 * logvar)
        
        z = self.sample_normal(dim=inputs.size(0), use_gpu=self.use_gpu)
        latent_sample = z * std + mean
        return latent_sample, mean, logvar


    #def sample_normal(self, dim):
    #    z = torch.randn((self.hidden_layers, dim, self.latent_size))
    #    return z.cuda() if self.use_gpu else z
    
    def sample_normal(self, dim, use_gpu=False):
        z = torch.randn((dim, self.latent_size))
        return z.cuda() if use_gpu else z

    def init_state(self, dim, use_gpu=False):
        state = torch.zeros((self.hidden_layers, dim, self.hidden_size))
        return state.cuda() if use_gpu else state



class Decoder(nn.Module):
    def __init__(self, embed_size, latent_size, hidden_size, num_heads, dropout, output_size):
        super().__init__()
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        # Multi-head self-attention mechanism
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout)

        self.attention2out = nn.Linear(in_features=embed_size, out_features=output_size) 

    def forward(self, embeddings, state, lengths):
        attn_output, _ = self.self_attention(embeddings, embeddings, embeddings) # self-attention
        output = self.attention2out(attn_output)
        return output, state


class NGMM(nn.Module):
    '''
    Neural Gaussian mixture model.
    '''
    def __init__(self, input_size, num_components, num_layers, hidden_size, output_size):
        super().__init__()
#        self.input_size = config.get('latent_size')
#        self.num_components = config.get('ngmm_num_components')
#        self.num_layers = config.get('ngmm_num_layers')
#        self.hidden_size = config.get('ngmm_hidden_size')
#        self.output_size = config.get('ngmm_output_size')
        self.input_size = input_size
        self.num_components = num_components
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size # number of dimensions in target
        
        self.mlp=nn.ModuleList()
        for i in range(self.num_layers):
            if i==0:
                in_features=self.input_size
            else:
                in_features=self.hidden_size
            out_features=self.hidden_size
            fc=nn.Linear(in_features, out_features)
            self.mlp.append( fc )
            
        self.linear_alpha = nn.Linear(self.hidden_size, self.num_components)
        self.linear_mu = nn.Linear(self.hidden_size, self.num_components*self.output_size)
        self.linear_logsigma = nn.Linear(self.hidden_size, self.num_components*self.output_size)


class MLP(nn.Module):
    '''
    MLP for regression.
    '''
    def __init__(self, input_size, num_layers, hidden_size, output_size, dropout):
        super().__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size # number of dimensions in target
        self.dropout = dropout
        
        self.mlp=nn.ModuleList()
        for i in range(self.num_layers):
            if i==0:
                in_features=self.input_size
            else:
                in_features=self.hidden_size
            if i==self.num_layers-1:
                out_features=self.output_size
            else:
                out_features=self.hidden_size
                
            fc=nn.Linear(in_features, out_features)
            self.mlp.append( fc )
            
            
    def forward(self, z):
        for i in range(self.num_layers):
            if i<self.num_layers-1:
                z = F.dropout(F.relu( self.mlp[i](z) ), p=self.dropout )
                #z = F.relu( self.mlp[i](z) )        
            else:
                z = self.mlp[i](z)
        return z
        

class Frag2Mol(nn.Module):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.input_size = vocab.get_size()
        self.embed_size = config.get('embed_size')
        self.hidden_size = config.get('hidden_size')
        self.hidden_layers = config.get('hidden_layers')
        self.latent_size = config.get('latent_size')
        self.dropout = config.get('dropout')
        self.use_gpu = config.get('use_gpu')
        self.num_heads = config.get('num_heads')
        
        #NGMM hyperparameters
        self.predictor_num_layers = config.get('predictor_num_layers')
        self.predictor_hidden_size = config.get('predictor_hidden_size')
        self.predictor_output_size = config.get('predictor_output_size')

        embeddings = self.load_embeddings()
        self.embedder = nn.Embedding.from_pretrained(embeddings)
        
        #self.latent2rnn = nn.Linear(
        #    in_features=self.latent_size,
        #    out_features=self.hidden_size)
        # I changed the above to this:
        self.latent2rnn = nn.Linear(
            in_features=self.latent_size,
            out_features=self.hidden_layers*self.hidden_size)

        self.encoder = Encoder(
            input_size=self.input_size,
            embed_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_heads = self.num_heads,
            #hidden_layers=self.hidden_layers,
            latent_size=self.latent_size,
            dropout=self.dropout,
            use_gpu=self.use_gpu)

        self.decoder = Decoder(
            embed_size=self.embed_size,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size,
            num_heads = self.num_heads,
            #hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            output_size=self.input_size)
        
        # MLP
        self.mlp = MLP(input_size=self.latent_size, num_layers=self.predictor_num_layers, 
                       hidden_size=self.predictor_hidden_size, output_size=self.predictor_output_size,
                       dropout=self.dropout)

    def forward(self, inputs, lengths):
        # INPUTS:
        # inputs: list or tensor of fragment indices
        # lengths: lengths of molecules
        # OUTPUTS:
        # output: batch x L x output_size
        # mu: batch x latent_size
        # sigma: batch x latent_size
        batch_size = inputs.size(0)
        embeddings = self.embedder(inputs)
        #print('self.training:', self.training) # where is it set?
        embeddings1 = F.dropout(embeddings, p=self.dropout, training=self.training) # where is self.training?
        z, mu, logvar = self.encoder(inputs, embeddings1, lengths)
        #state = self.latent2rnn(z) # z: num_layers x batch x latent_size, state: num_layers x batch x hidden_size
        #state = state.view(self.hidden_layers, batch_size, self.hidden_size) #num_layer x batch x hidden_size, maybe not needed
        
        state = self.latent2rnn(z) # batch x num_layers*latent_size
        #state = torch.tanh(state) # I added this, NOTE: should be consistent with sampler.
        state = state.view(batch_size, self.hidden_layers, self.hidden_size) # batch x num_layers x hidden_size
        state = state.transpose(1,0) # now state is num_layer x batch x hidden_state
        state= state.contiguous()
        #print('state:', state.shape)
        
        embeddings2 = F.dropout(embeddings, p=self.dropout, training=self.training)
        output, state = self.decoder(embeddings2, state, lengths)
        
        # the MLP component
        mlp_pred = self.mlp(z)
        #mlp_pred = 0
        
        return output, mu, logvar, mlp_pred


    def load_embeddings(self):
        filename = f'emb_{self.embed_size}.dat'
        path = self.config.path('config') / filename
        embeddings = np.loadtxt(path, delimiter=",")
        return torch.from_numpy(embeddings).float()


class Loss(nn.Module):
    def __init__(self, config, pad):
        super().__init__()
        self.config = config
        self.pad = pad

    def forward(self, output, target, mu, logvar, epoch, idx, properties, mlp_predicted):
        # INPUTS:
        # output: batch x L x output_size (vocab_size), softmax probs
        # target:  batch X L
        # mu: batch x latent_size
        # sigma: batch x latent_size
        
        batch_size=output.shape[0]
        output = F.log_softmax(output, dim=1)
        
        #print('in loss ...')
        #print('target size:', target.shape)
        #print('output size:', output.shape)

        # flatten all predictions and targets
        target = target.view(-1) # 1d tensor now: batch*L
        output = output.view(-1, output.size(2)) # batch*L x vocab_size
        
        #print('target size:', target.shape)
        #print('output size:', output.shape)

        # create a mask filtering out all tokens that ARE NOT the padding token
        mask = (target > self.pad).float() # 1d vector of integers

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask).item()) # total number of tokens in the 1d target

        # pick the values for the label and zero out the rest with the mask
        output = output[range(output.size(0)), target] * mask # obtain the log-probabilities for each target class

        # compute cross entropy loss which ignores all <PAD> tokens
        #CE_loss = -torch.sum(output) / nb_tokens # this is mean over all all targets
        CE_loss = -torch.sum(output) / batch_size # mean over batch
        #CE_loss = -torch.sum(output)

        # compute KL Divergence
        #KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KL_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean() # mean over batch
        # alpha = (epoch + 1)/(self.config.get('num_epochs') + 1)
        # return alpha * CE_loss + (1-alpha) * KL_loss
 
        # regression loss, I THINK I NEED TO USE MSE INSTEAD

        MSE_loss = F.mse_loss(mlp_predicted, properties, reduction='sum')/batch_size
        #MSE_loss = F.mse_loss(mlp_predicted, properties, reduction='mean')
        #MSE_loss = 0
        #alpha=0.5
        #alpha = np.max( [(epoch-3)/self.config.get('num_epochs'), 0] )
        #alpha = (epoch/self.config.get('num_epochs'))**2
        #if epoch!=0 and (epoch+1)%2 == 0:
        #    alpha = (epoch+1)/self.config.get('num_epochs')
        #else:
        #    beta = 0
        offset_epoch = self.config.get('offset_epoch')
        T = self.config.get('start_epoch') + self.config.get('num_epochs') - offset_epoch
        t = epoch+1 - offset_epoch
        if T<=0:
            T=1
            t=1
        beta=get_beta(self.config.get('k_beta'), self.config.get('a_beta'), self.config.get('l_beta'), self.config.get('u_beta'), T, t)
        #loss = (1-beta) * ((1-alpha)*CE_loss + alpha*KL_loss) + beta*MSE_loss
        #loss = (1-beta) * (CE_loss + alpha*KL_loss) + beta*MSE_loss
        #loss = (1-beta) * (CE_loss + KL_loss) + beta*MSE_loss
        #loss = CE_loss + KL_loss
        #loss = (1-beta) * (CE_loss + alpha*KL_loss)
        #loss = CE_loss + alpha*KL_loss
        
        # increase alpha along with beta
        #if self.config.get('increase_alpha')>1:
            #alpha = self.config.get('increase_alpha')*alpha
        alpha = get_beta(self.config.get('k_alpha'), self.config.get('a_alpha'), self.config.get('l_alpha'), self.config.get('u_alpha'), T, t)
        
        loss = CE_loss + alpha*MSE_loss + beta*KL_loss
        if idx%100 == 0:
            print('CE_loss:',CE_loss.item())
            print('KL_loss:',KL_loss.item())
            print('MSE_loss:', MSE_loss.item())
        
        return loss, CE_loss.item(), MSE_loss.item(), KL_loss.item(), alpha, beta


def mvgaussian(t, mu,sigma):
    '''
    INPUTS:
    t: batch x K where K is the number of dimensions
    mu: batch x K
    sigma: batch x K
    OUTPUTS:
    pdf: batch
    '''
    #sigma_inv = 1/(sigma + 1e-31)
    K=len(mu)
    norm = torch.prod(sigma, dim=1) * torch.sqrt(2*torch.tensor(np.pi)).pow(K)
    expo = torch.exp( -0.5*( ((t-mu).pow(2)/sigma.pow(2)).sum(dim=1) ) )
    return expo/norm
    

def get_beta(k, a, l, u, T, t):
    """
    Compute the value of beta. When k=0, a is the fixed beta value. Usually we let a=1.
    Special cases:
        when a=0: beta=l
        when k=0: beta=max(a, l)
        when k=0, b=0: beta=a
        when k=0, a=0: beta=l
    INPUTS:
        a, T, t: scalars in formula beta=a*np.exp( k*(1-T/t) ) where a>=0, k>=0.
        l: scalar: l>=0, offset as min value of beta, usually we let l=0.
        u: scalar: u>0. max.
    OUTPUTS:
        beta: scalar.
    """
    beta = a*np.exp( k*(1-T/t) )
    beta = np.max( [beta, l] )
    beta = np.min( [beta, u] )
    return beta
    
