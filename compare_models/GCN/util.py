import torch


class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)
def pad_with_last_val(vect,k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect
class Classifier(torch.nn.Module):
    def __init__(self,args):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()


        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = args.cls_in_feats,
                                                       out_features =8),
                                       activation,
                                       torch.nn.Linear(in_features = 8,
                                                       out_features = 2))

    def forward(self,x):
        return self.mlp(x)
