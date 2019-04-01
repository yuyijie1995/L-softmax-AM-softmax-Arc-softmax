import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon
import math

class Arc_Loss(gluon.nn.HybridBlock):
    def __init__(self,numclasses,feature_dim,margin=2,scale=30,**kwargs):
        super(Arc_Loss,self).__init__(**kwargs)
        self._num_classes=numclasses
        self._feature_dim=feature_dim
        self._margin=margin
        self._scale=scale
        self.weight=self.params.get('weight',shape=(feature_dim,numclasses),init=mx.init.Xavier())


    def hybrid_forward(self, F, x, y,weight):
        x_norm=F.L2Normalization(x,mode='instance')
        W_norm=F.L2Normalization(weight,mode='instance')
        fc=F.dot(x_norm,W_norm,transpose_b=True)
        if y is None:
            return fc
        target_logits=F.pick(fc,y)
        theta=F.arccos(target_logits)
        marginal_target_logit=F.cos(theta+self._margin)
        one_hot=F.one_hot(y,depth=self._num_classes,on_value=1.0,off_value=0.0)
        fc_label=fc+F.broadcast_mul(one_hot,F.expand_dims(marginal_target_logit-target_logits,1))#no mask
        output=fc_label*self._scale
        return output
