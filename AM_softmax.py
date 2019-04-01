import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon,autograd
import math

class AM_Loss(gluon.nn.HybridBlock):
    def __init__(self,numclasses,feature_dim,margin,scale=30,**kwargs):
        super(AM_Loss,self).__init__(**kwargs)
        self._numclasses=numclasses
        self._feature_dim=feature_dim
        self._margin=margin
        self._scale=scale
        self.weight=self.params.get('weight',shape=(feature_dim,numclasses),init=mx.init.Xavier())



    def hybrid_forward(self, F, x, y, weight):
        norm_x=F.norm(x,axis=1)
        x_norm = F.L2Normalization(x, mode='instance')
        norm_w = F.norm(weight, axis=0)
        w_norm = weight / norm_w
        if y is None:
            output=F.dot(x_norm,w_norm,transpose_b=True)
            return output

        batch_size=x.shape[0]
        mask=F.zeros((batch_size,self._numclasses))
        mask[range(batch_size),y]=1

        output=F.dot(x_norm,w_norm,transpose_b=True)
        scale_output=self._scale*output
        cos_target=F.pick(output,y)/norm_x/norm_w
        expand_cos_target=F.expand_dims(cos_target,1)
        cos_t=(expand_cos_target-self._margin)*self._scale
        mask_cos_t=mask*cos_t
        logits=scale_output-scale_output*mask+mask_cos_t
        return logits





