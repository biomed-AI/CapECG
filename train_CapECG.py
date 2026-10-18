
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr,spearmanr
from torch import optim
from Dataloader import SNP_ECG_dataset
from models_upload import CapECG
from collections import Counter
from sklearn.metrics import r2_score
import os
from sklearn.model_selection import KFold
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def regularization_loss(model, factor, p=2):
    reg_loss = torch.tensor(0.,)
    for name, w in model.named_parameters():
        if 'weight' in name: 
            reg_loss = reg_loss + torch.norm(w, p)
    reg_loss = factor * reg_loss
    return reg_loss

def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() 
    return l1_reg

def l2_reg_all(model, reg_type=None):
    l1_reg = None
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.pow(W,2).sum()
        else:
            l1_reg = l1_reg + torch.pow(W,2).sum() 
    return l1_reg


def rmse(target,predict):
    return np.sqrt(((predict - target) ** 2).mean())
 
def R2(target, predict):
    SSE = sum((target - predict) ** 2)
    SST = sum((target - (target).mean()) ** 2)
    return 1 - (SSE / SST)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def priority(x):
    return int(x.split('V')[-1])

def train_model(args):
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs_num=args.epochs_num
    batch_size = args.batch_size
    step_print = 200
    num_workers = args.num_workers
    lr = args.lr
    lambda_reg = 0.1

    feature = args.ECG_trait
    train_npy_path = args.train_geno_path
    test_npy_path = args.test_geno_path
    train_pheno_path = args.train_pheno_path
    test_pheno_path = args.test_pheno_path

    train_npy = np.load(train_npy_path)
    test_npy = np.load(test_npy_path)
    train_pheno_df = pd.read_csv(train_pheno_path, sep='\t')
    test_pheno_df = pd.read_csv(test_pheno_path, sep='\t')

    train_dataset = SNP_ECG_dataset(train_npy, train_pheno_df, feature)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
        num_workers=num_workers)
    test_dataset = SNP_ECG_dataset(test_npy, test_pheno_df, feature)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
        num_workers=num_workers)

    input_dim = train_npy.shape[2]
    input_len = train_npy.shape[1]
    output_dim = 1

    model = CapECG(input_len, input_dim, output_dim, args).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    mseloss = nn.MSELoss()

    for epoch in range(epochs_num):
        model.train()  
        loss_sum = 0
        for step,(batch_x,batch_y, batch_discrete_y) in enumerate(tqdm(train_dataloader)):
            batch_x = batch_x.type(torch.FloatTensor).to(device)
            batch_y = batch_y.reshape(-1,1)
            batch_y = batch_y.type(torch.FloatTensor).to(device)
            batch_discrete_y = batch_discrete_y.type(torch.FloatTensor).to(device)

            pred_y, atta_map, x_gen, pred_y2 = model(batch_x, batch_discrete_y)
            loss = mseloss(pred_y,batch_y) + mseloss(batch_x.reshape(batch_x.size(0), -1),x_gen.reshape(x_gen.size(0), -1)) + mseloss(pred_y2,batch_y)*0.1
            #loss = mseloss(pred_y,batch_y)*0.8 + mseloss(batch_x.reshape(batch_x.size(0), -1),x_gen.reshape(x_gen.size(0), -1))*0.2 + mseloss(pred_y2,batch_y)*0.1
            #loss_reg = l2_reg_all(model) * lambda_reg
            #loss_reg = regularization_loss(model, factor=0.5, p=2)
            loss_sum += loss.item()
            #loss = loss + loss_reg

            optimizer.zero_grad()      
            loss.backward()  
            optimizer.step()

        if step%1 == 0:
            print("Epoch: %d, loss: %.6f"%(epoch, loss_sum/step))

        model.eval()
        for step,(batch_x,batch_y, batch_discrete_y) in enumerate(tqdm(test_dataloader)):
            batch_x = batch_x.type(torch.FloatTensor).to(device)
            batch_y = batch_y.reshape(-1,1)
            pred_y, atta_map, x_gen, pred_y2 = model(batch_x)
            pred_y = pred_y.cpu().detach().numpy()
            batch_y = batch_y.cpu().detach().numpy()
            if step==0:
                pred_y_all = pred_y
                gt_y_all = batch_y
            else:
                pred_y_all = np.concatenate((pred_y_all, pred_y))
                gt_y_all = np.concatenate((gt_y_all, batch_y))

        pcc, pcc_p = pearsonr(pred_y_all.reshape(-1),gt_y_all.reshape(-1))
        spc, spc_p = spearmanr(pred_y_all.reshape(-1),gt_y_all.reshape(-1))
        rmse_s = rmse(gt_y_all.reshape(-1), pred_y_all.reshape(-1))
        r2_s = r2_score(gt_y_all.reshape(-1), pred_y_all.reshape(-1))

        print(pearsonr(pred_y_all.reshape(-1),gt_y_all.reshape(-1)), 
        spearmanr(pred_y_all.reshape(-1),gt_y_all.reshape(-1)),
        rmse(gt_y_all.reshape(-1), pred_y_all.reshape(-1)),
        r2_score(gt_y_all.reshape(-1), pred_y_all.reshape(-1)))

    torch.save(model.state_dict(), args.save_model_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--ECG_trait", type=str)
    parse.add_argument("--train_geno_path", type=str)
    parse.add_argument("--test_geno_path", type=str)
    parse.add_argument("--train_pheno_path", type=str)
    parse.add_argument("--test_pheno_path", type=str)
    parse.add_argument("--save_model_path", type=str)
    parse.add_argument("--device", type=str, default = "cuda:0") 
    parse.add_argument("--epochs_num", type=int, default = 50)
    parse.add_argument("--batch_size", type=int, default = 256)
    parse.add_argument("--num_workers", type=int, default = 1)
    parse.add_argument("--lr", type=float, default = 5e-4)
    parse.add_argument("--hidden_dim", type=int, default = 32)
    parse.add_argument("--num_layers", type=int, default = 4)

    args = parse.parse_args()
    train_model(args)





