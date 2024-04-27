from model.basic_module import *
import model.Config as cfg

class LRCore(nn.Module):
    def __init__(self):
        super(LRCore, self).__init__()
        self.down_1=basic_encoder(n_sample=1024,n_group=32,inc=3,outc=64,
                                  RIF=True,CA=True,LA=True,gp=2,first=True,Pree=True,Pose=True)
        self.down_2=basic_encoder(n_sample=256,n_group=16,inc=64,outc=256,
                                  RIF=True,CA=True,LA=True,gp=4,first=False,Pree=True,Pose=True)
        self.pooling=GeM_pool()

    def forward(self,xyz):
        xyz,feature=self.down_1(xyz,None)
        xyz,feature=self.down_2(xyz,feature)
        out=self.pooling(feature)
        return out

if __name__ == '__main__':
    print('running some test...')
    from data.dataset_utils import make_dataloaders
    model_test=LRCore().cuda()
    test_loader=make_dataloaders()
    for batch, positives_mask, negatives_mask in test_loader['train']:
        n_positives = torch.sum(positives_mask).item()
        n_negatives = torch.sum(negatives_mask).item()
        batch=batch.cuda()
        tt=model_test(batch)
        tt=0