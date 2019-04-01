import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
from mxnet import gluon,autograd
import math

class L_M_Loss(gluon.nn.HybridBlock):
    def __init__(self,numclass,feature_dim,margin,beta,beta_min,scale,**kwargs):
        super(L_M_Loss,self).__init__(**kwargs)
        self._num_class=numclass
        self._feature_dim=feature_dim
        self._margin=margin
        self._beta=beta
        self._beta_min=beta_min
        self._scale=scale
        self._divisor=math.pi/margin
        self.c_map=[]
        self.k_map=[]
        self.weight=self.params.get('weight',shape=(feature_dim,numclass),init=mx.init.Xavier())
        c_m_n=lambda m,n :math.factorial(n)/(math.factorial(m)*math.factorial(n-m))
        for i in range(margin+1):
            self.c_map.append(c_m_n(i,margin))
            self.k_map.append(math.cos(i*math.pi/margin))


    def find_k1(self,cos_t):
        acos=mx.nd.arccos(cos_t)
        k=(acos/self._divisor).floor()
        return k

    def calc_cos_mt(self,cos_t):
        cos_mt=0
        sin2_t =1-cos_t*cos_t
        flag=-1
        for p in range(self._margin/2+1):
            flag=flag*(-1)
            cos_mt=cos_mt+flag*self.c_map[2*p]*nd.power(cos_t,self._margin-2*p)*nd.power(sin2_t,p)
        return cos_mt

    def hybrid_forward(self, F, x, y,weight):

        if y is None:
            output=F.dot(x,weight,transpose_b=True)
            return output
        batch_size=y.shape[0]
        mask=F.zeros((batch_size,self._num_class))
        mask[range(batch_size),y]=1

        output=F.dot(x,weight,transpose_b=True)
        if y is None:
            return output
        w_norm=F.norm(weight,axis=1)
        x_norm=F.norm(x,axis=1)
        f=F.pick(output,y)
        w_norm_label=F.take(w_norm,y)
        cos_t=f/(w_norm_label*x_norm)
        k=self.find_k1(cos_t)
        cos_mt = self.calc_cos_mt(cos_t)
        f_i_j = (F.power(-1, k) * cos_mt - 2 * k) * (w_norm_label * x_norm)
        #output[range(batch_size),y]=f_i_j
        expand_f_i_j=F.expand_dims(f_i_j,1)
        mask_f_i_j=mask*expand_f_i_j
        logits=output-output*mask+mask_f_i_j
        return logits





