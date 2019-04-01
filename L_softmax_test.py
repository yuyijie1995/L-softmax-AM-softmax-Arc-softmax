import argparse
import logging
from mxnet import gluon,autograd
from mxnet.gluon import nn
import mxnet as mx
from mxnet.gluon.data.vision import datasets,transforms
from mxnet.gluon import model_zoo
from Large_margin_loss import L_M_Loss
from gluoncv.model_zoo import get_model


def parse_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--num-workers',type=int,default=0)
    parser.add_argument('--save-prefix',type=str,default='output/resnet18')
    parser.add_argument('--save-step',type=int,default=5)
    parser.add_argument('--use-hybrid',type=bool,default=False)
    parser.add_argument('--lr-decay-epoch',type=list,default=[150,255])
    parser.add_argument('--lr-decay',type=float,default=0.1)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--optimizer',type=str,default='sgd')
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--wd',type=float,default=5e-4)
    parser.add_argument('--margin',type=int,default=2,help='margin in LS loss')
    parser.add_argument('--beta',type=float,default=1)
    parser.add_argument('--beta_min',type=float,default=0)
    parser.add_argument('--scale',type=float,default=1)
    parser.add_argument('--gpus',type=str,default='0,1,2,3')
    args=parser.parse_args()
    return args

def transform():
    transformer=transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.13,0.31)])
    return transformer

def cifar10Data(args):
    cifar_train = datasets.CIFAR10(root='data/', train=True).transform_first(transform())
    cifar_val = datasets.CIFAR10(root='data/', train=False).transform_first(transform())
    train_data = gluon.data.DataLoader(cifar_train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers)
    val_data = gluon.data.DataLoader(cifar_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    return train_data, val_data

def acc(output,label):
    pre_label=output.argmax(axis=1)
    return (pre_label==label.astype('float32')).mean().asscalar()

def train(args,train_data,val_data,net,ctx):
    lr_decay=args.lr_decay
    lr_decay_epoch=args.lr_decay_epoch
    soft_max_loss=gluon.loss.SoftmaxCrossEntropyLoss()
    L_loss=L_M_Loss(10,10,args.margin,args.beta,args.beta_min,args.scale)
    L_loss.initialize(mx.init.Xavier(),ctx=ctx)
    params=net.collect_params()
    params.update(L_loss.collect_params())
    trainer=gluon.Trainer(params=params,optimizer=args.optimizer,optimizer_params={
        'learning_rate':args.lr,'momentum':args.momentum,'wd':args.wd
    })
    epoch_index=0
    for epoch in range(args.epochs):
        lr_decay_count=0
        train_loss=0.0
        train_acc=0.0
        val_acc=0.0
        if epoch==lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count+=1
        for data,label in train_data:
            with autograd.record():
                output1=net(data)
                output=L_loss(output1,label)
                loss=soft_max_loss(output,label)
                loss.backward()
                train_loss+=loss.mean().asscalar()
            trainer.step(args.batch_size)
            #print(train_loss)
        for data,label in val_data:
            output2=net(data)
            output3=L_loss(output2,None)
            val_acc+=acc(output3,label)
        epoch_index+=1
        print('Epoch {}:Loss {:.4f},Train accuracy {:.4f},Val accuracy {:.4f}'.format(epoch,train_loss/len(train_data),
                                                                                                      train_acc/len(train_data),
                                                                                                      val_acc/len(val_data),
                                                                                                      ))




def main():
    args=parse_arguments()
    train_data,val_data=cifar10Data(args)
    net=get_model('cifar_resnet20_v1',classes=10)
    ctx=mx.cpu()
    net.initialize(mx.init.Xavier(),ctx=ctx)
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler=logging.FileHandler('train.log')
    logger.addHandler(file_handler)
    logger.info(args)
    train(args,train_data,val_data,net,ctx)


if __name__=='__main__':
    main()




