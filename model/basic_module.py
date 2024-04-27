from model.util import *

class basic_encoder(nn.Module):
    def __init__(self,n_sample,n_group,inc,outc,dilation=2,RIF=True,CA=True,LA=True,gp=2,first=False,Pree=True,Pose=True):
        super(basic_encoder, self).__init__()
        self.n_sample=n_sample
        self.n_group=n_group
        self.dilation=dilation
        self.first=first
        self.RIF=RIF
        if first:
            if RIF:
                input_c = 7
            else:
                input_c = 3
        else:
            if RIF:
                input_c=inc+7
            else:
                input_c=inc+3
        self.Stem=SharedMLP(in_channels=input_c,out_channels=outc,with_bn=True,mode='2d')
        self.Pree=Pree
        if self.Pree:
            self.Pre=ResidualMLP(outc,mode='2d')
        self.Pose=Pose
        if self.Pose:
            self.Pos=ResidualMLP(outc,mode='1d')
        else:
            self.Pos=SharedMLP(outc,outc,mode='1d')
        self.CA=CA
        self.LA=LA
        if CA:
            self.Channel_attention = ECALayer(channels=outc)
        if LA:
            self.Spatial_attentione = G_Transformer(channels=outc,gp=gp)

    def _encoding(self,feature):
        new_feature=self.Stem(feature)
        if self.Pree:
            Pre_feature=self.Pre(new_feature)
            Pre_feature=torch.max(Pre_feature,2)[0]
        else:
            Pre_feature = torch.max(new_feature, 2)[0]
        if self.CA and not self.LA:
            att_feature=self.Channel_attention(Pre_feature)
        elif not self.CA and self.LA:
            att_feature = self.Spatial_attentione(Pre_feature)
        elif self.CA and self.LA:
            channel_att_feature=self.Channel_attention(Pre_feature)
            spatial_att_feature=self.Spatial_attentione(Pre_feature)
            att_feature=channel_att_feature+spatial_att_feature
        else:
            att_feature=Pre_feature
        if self.Pose:
            Pos_feature=self.Pos(att_feature)
            Pos_feature=Pos_feature+Pre_feature
        else:
            Pos_feature = att_feature
        return Pos_feature

    def forward(self,xyz,feature,idx=None):
       # feature B,N,
        B,N,_=xyz.shape
        new_xyz, new_feature, _ = sample_and_group(self.n_sample, None, self.n_group, xyz, feature,idx=idx,
                                                   returnfps=True, knn=True, dialated_ratio=self.dilation,RIF=self.RIF)
        new_feature=self._encoding(new_feature)
        return new_xyz, new_feature
