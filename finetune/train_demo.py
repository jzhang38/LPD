from fewshot_re_kit.data_loader import FewRelDataset, FewRelPredictionDataset, collate_fn_prediction, get_loader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder
from models.proto import Proto
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import torch
import random
import torch.utils.data as data
from tqdm import tqdm
def predict(model,N, K, Q,eval_iter,ckpt=None): 
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")
        
        model.eval()
        
            
        

        print("----start prediction-----")
        result = []
        with torch.no_grad():
            for support, query in tqdm(eval_iter):
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    
                logits, pred = model(support, query, N, K, Q)
                result.extend(pred.tolist())
        return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='train_wiki',
            help='train file')
    parser.add_argument('--val', default='val_wiki',
            help='val file')
    parser.add_argument('--test', default='test_wiki',
            help='test file')

    parser.add_argument('--trainN', default=10, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=5, type=int,
            help='K shot')
    parser.add_argument('--Q', default=5, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=10000, type=int,
            help='num of iters in training')
    parser.add_argument('--val_iter', default=1000, type=int,
            help='num of iters in validation')
    parser.add_argument('--test_iter', default=10000, type=int,
            help='num of iters in testing')
    parser.add_argument('--val_step', default=2000, type=int,
           help='val after training how many iters')
    parser.add_argument('--model', default='proto',
            help='model name')
    parser.add_argument('--encoder', default='cnn',
            help='encoder: bert or roberta')
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=2e-5, type=float,
           help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
           help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')

    parser.add_argument('--grad_iter', default=1, type=int,
           help='accumulate gradient every x iterations')
 
    
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
           help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
           help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
           help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
           help='only test')

    parser.add_argument('--pretrain_ckpt', default=None,
            help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--seed', default=42,type=int,
            help='seed')
    parser.add_argument('--path', default=None,
            help='path to ckpt')
    parser.add_argument('--mode', default="CM",
            help='mode {CM, OC, OM}')
    
    
    parser.add_argument('--do_prediction', action='store_true',
            )
    parser.add_argument('--label_mask_prob', default=1.0, type=float,
            )
    parser.add_argument("--prediction_save_path", type=str,
                    )

    parser.add_argument("--debug",action="store_true",
                    )
    opt = parser.parse_args()

   

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    
    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    
    
    if encoder_name == 'bert':
        pretrain_ckpt = opt.pretrain_ckpt or 'bert-base-uncased'

        sentence_encoder = BERTSentenceEncoder(
                    pretrain_ckpt,
                    max_length,
                    opt.path,
                    opt.mode)
    elif encoder_name == 'roberta':
        pretrain_ckpt = opt.pretrain_ckpt or 'roberta-base'

        sentence_encoder = RobertaSentenceEncoder(
                    pretrain_ckpt,
                    max_length)
    else:
        raise NotImplementedError
    
    if model_name == 'proto':
        model = Proto(sentence_encoder, hidden_size=opt.hidden_size)
    else:
        raise NotImplementedError


    if opt.do_prediction:
        print("-------------do prediction-------------")
        ckpt = opt.load_ckpt
        if not os.path.isfile(opt.load_ckpt) :
            assert False
        if ckpt != None:
            try: 
                state_dict = torch.load(ckpt)['state_dict']
                print("Successfully loaded checkpoint '%s'" % ckpt)
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)

                for name, param in own_state.items():
                    if name not in state_dict:
                        print(name)
            except:
                print('save ckpt not succesfully loaded')
                assert False
        
        prediction_dataset = FewRelPredictionDataset(opt.test, sentence_encoder)
        prediction_data_loader = data.DataLoader(dataset=prediction_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=8,
            collate_fn=collate_fn_prediction)
  
        result = predict(model,N, K, Q,prediction_data_loader,ckpt=opt.load_ckpt)
        json.dump(result,  sys.stdout)
        os.makedirs(opt.prediction_save_path,exist_ok=True) 
        with open(opt.prediction_save_path + "pred-%d-%d.json" %(N, K),'w') as f:
            json.dump(result, f)
        
        print('------done-----------')
        return

    train_data_loader = get_loader(opt, opt.train, sentence_encoder,
                N=trainN, K=K, Q=Q,  batch_size=batch_size, label_mask_prob=opt.label_mask_prob)
    val_data_loader = get_loader(opt, opt.val, sentence_encoder,
                N=N, K=K, Q=Q,  batch_size=batch_size, label_mask_prob=0.0)
    test_data_loader = get_loader(opt, opt.test, sentence_encoder,
                N=N, K=K, Q=Q,  batch_size=batch_size, label_mask_prob=0.0)


    pytorch_optim = optim.SGD

  
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
        
    prefix = '-'.join([model_name, encoder_name, opt.train, opt.val, str(N), str(K)])

    
   
    
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta']:
            bert_optim = True
        else:
            bert_optim = False

        framework.train(model, prefix, batch_size, trainN, N, K, Q,
                pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                 val_step=opt.val_step, fp16=opt.fp16, 
                train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim, lr=opt.lr)
    else:
        ckpt = opt.load_ckpt
    
    if opt.only_test:
        acc = framework.eval(model, batch_size, N, K, Q, opt.test_iter, ckpt=ckpt)
        print("RESULT: %.2f" % (acc * 100))
        if opt.load_ckpt:
            name = os.path.basename(opt.load_ckpt)
        else:
            path = os.path.dirname(opt.path)
            name = os.path.basename(path)
        with open("accuracy/"+name + "%dN-%dK"%(N, K), "w") as f:
            f.write("Seed %d RESULT: %.2f" % (opt.seed, acc * 100))
    


if __name__ == "__main__":
    main()
