import os
import subprocess
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import argparse

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def return_int_list(intput_list):
    output_list = []
    for a in intput_list:
        #a = a.split('\n')[0] 
        if a!='NA':
            output_list.append(int(a))
        else:
            output_list.append(0)
    return output_list

def raw2npy(rawdata_path, geno_path):
    genotype_data = []
    n=0
    max_n=40085
    for line in open(rawdata_path):
        line = line.split('\n')[0]
        FID = line.split('\t')[0]
        if n!=0:
            genotype_data.append(return_int_list(line.split(' ')[6:]))
        n+=1
        # if n%(int(max_n/100))==0:
        #     print('\r',int(n/(int(max_n/100))), end='', flush=True)
    np.save(geno_path, np.array(genotype_data))

def clump_getSNP_LD(feature, GWAS_path, save_dir):
    print(feature)
    GWAS_df1 = pd.read_csv(GWAS_path, sep='\t', error_bad_lines=False)

    mkdir(save_dir)

    clump_dir = os.path.join(save_dir, 'clumping')
    mkdir(clump_dir)
    mkdir(os.path.join(save_dir, 'SNP'))
    clump_path = os.path.join(clump_dir, feature)
    r2 = 0.1
    order = '/bigdat2/user/linsy/bigdat1/zhanghy/soft/plink_1.9/plink \
            --bfile /bigdat2/user/linsy/bigdat1/zhanghy/no_mhc/eur/eur_nomhc \
            --clump-p1 0.001 --clump-p2 1 --clump-r2 '+str(r2)+' --clump-kb 1000 \
            --clump '+GWAS_path+' \
            --clump-snp-field SNP --clump-field P_BOLT_LMM_INF \
            --out '+clump_path
    os.system(order)

    clump_df = pd.read_csv(clump_path+'.clumped', sep='\s+')
    
    n=0
    all_SNP_list = []
    for b in tqdm(clump_df.index.tolist()):
        SNP_df = clump_df.loc[b,'SNP']
        chr_df2 = clump_df.loc[b,'CHR']
        BP_df2 = clump_df.loc[b,'BP']
        P_df2 = clump_df.loc[b,'P']

        SP2_list = clump_df.loc[b,'SP2'].split(',')
        SP2_list = [a.split('(')[0] for a in SP2_list]+[SNP_df]
        all_SNP_list = list(set(all_SNP_list) | set(SP2_list))
        
        GWAS_df1_target1 = GWAS_df1.loc[GWAS_df1['SNP'].isin(SP2_list)]

        SNP_save_dir = os.path.join(save_dir, 'SNP', SNP_df)
        mkdir(SNP_save_dir)
        save_path = os.path.join(SNP_save_dir, SNP_df+'.csv')
        if os.path.exists(save_path):
            continue
        GWAS_df1_target1.to_csv(save_path, index=False, sep='\t')

def LD_PCA(feature, save_dir, bfile):
    for rsid in tqdm(os.listdir(os.path.join(save_dir, 'SNP'))):
        rsid_path = os.path.join(save_dir, 'SNP', rsid, rsid+'.csv')

        df = pd.read_csv(rsid_path, sep='\t').loc[:,['SNP']]
        SNP_path = os.path.join(save_dir, 'SNP', rsid, rsid+'.SNP')
        df.to_csv(SNP_path, index=False, header=None)
        pca_path = os.path.join(save_dir, 'SNP', rsid, rsid+'_PCA10')
        raw_path = os.path.join(save_dir, 'SNP', rsid, rsid+'')
        npy_path = os.path.join(save_dir, 'SNP', rsid, rsid+'.npy')
        pca_path2 = os.path.join(save_dir, 'SNP', rsid, rsid+'_PCA10.npy')

        order = '/bigdat2/user/linsy/bigdat1/zhanghy/soft/plink_1.9/plink \
            --bfile '+bfile+' \
            --extract '+SNP_path+' \
            --export A \
            --out '+raw_path
        os.system(order)
        if not os.path.exists(raw_path+'.raw'):
            continue
        raw2npy(raw_path+'.raw', npy_path)

        SNP_ori_npy = np.load(npy_path)
        if SNP_ori_npy.shape[1]<5:
            component = 1
        elif SNP_ori_npy.shape[1]<11:
            component = 5
        else:
            component = 10
        pca = PCA(n_components=component)
        SNP_pca = pca.fit_transform(SNP_ori_npy)
        np.save(pca_path2, SNP_pca)

def PCA_split_train_test_padding(feature, save_dir, train_FID_path, test_FID_path, bfile):
    print(feature)
    mkdir(os.path.join(save_dir, 'npy_data'))
    train_pca_all_path = os.path.join(save_dir, 'npy_data', 'train.npy')
    test_pca_all_path = os.path.join(save_dir, 'npy_data', 'test.npy')

    train_pca_npy_list = []
    test_pca_npy_list = []
    n=0
    snp_path = os.path.join(save_dir, 'SNP_list.csv')
    snp_df = pd.DataFrame(columns=['SNP'])
    snp_list = []
    for rsid in tqdm(os.listdir(os.path.join(save_dir, 'SNP'))):

        snp_list.append(rsid)
        pca_path2 = os.path.join(save_dir, 'SNP', rsid, rsid+'_PCA10.npy')
        if not os.path.exists(pca_path2) or rsid=='all':
            continue
        
        pca_npy = np.load(pca_path2)
        test_FID_df = pd.read_csv(test_FID_path, header=None, sep=' ')
        test_FID_list = test_FID_df.loc[:,0].tolist()
        train_FID_df = pd.read_csv(train_FID_path, header=None, sep=' ')
        train_FID_list = train_FID_df.loc[:,0].tolist()

        all_FID_df = pd.read_csv(bfile+'.fam', header=None, sep=' ')
        train_FID_index = all_FID_df.loc[all_FID_df[0].isin(train_FID_list)].index.tolist()
        test_FID_index = all_FID_df.loc[all_FID_df[0].isin(test_FID_list)].index.tolist()
        
        train_pca_path = os.path.join(save_dir, 'SNP', rsid, rsid+'_train.npy')
        test_pca_path = os.path.join(save_dir, 'SNP', rsid, rsid+'_test.npy')
        train_pca_npy = pca_npy[train_FID_index]
        test_pca_npy = pca_npy[test_FID_index]
        np.save(train_pca_path, train_pca_npy)
        np.save(test_pca_path, test_pca_npy)

        # train_pca_path = os.path.join(save_dir, rsid, rsid+'_train.npy')
        # test_pca_path = os.path.join(save_dir, rsid, rsid+'_test.npy')
        train_pca_npy = np.load(train_pca_path)
        test_pca_npy = np.load(test_pca_path)

        target_shape = (train_pca_npy.shape[0], 10)
        padding = [(0, target_shape[i] - train_pca_npy.shape[i]) for i in range(len(target_shape))]
        train_pca_npy = np.pad(train_pca_npy, padding, 'constant')

        target_shape = (test_pca_npy.shape[0], 10)
        padding = [(0, target_shape[i] - test_pca_npy.shape[i]) for i in range(len(target_shape))]
        test_pca_npy = np.pad(test_pca_npy, padding, 'constant')
        train_pca_npy = np.expand_dims(train_pca_npy, axis=1)
        test_pca_npy = np.expand_dims(test_pca_npy, axis=1)
        train_pca_npy_list.append(train_pca_npy)
        test_pca_npy_list.append(test_pca_npy)
        # if n==0:
        #     train_pca_all_npy = train_pca_npy
        #     test_pca_all_npy = test_pca_npy
        # else:
        #     train_pca_all_npy = np.concatenate((train_pca_all_npy,train_pca_npy),axis=1)
        #     test_pca_all_npy = np.concatenate((test_pca_all_npy,test_pca_npy),axis=1)
        #     #exit()
        n+=1
    snp_df['SNP'] = snp_list
    snp_df.to_csv(snp_path, index=False, header=None, sep=' ')

    train_pca_all_npy = np.concatenate(train_pca_npy_list,axis=1)
    test_pca_all_npy = np.concatenate(test_pca_npy_list,axis=1)
    print(train_pca_all_npy.shape, test_pca_all_npy.shape)
    np.save(train_pca_all_path, train_pca_all_npy)
    np.save(test_pca_all_path, test_pca_all_npy)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--ECG_trait", type=str, default = "V79")
    parse.add_argument("--GWAS_path", type=str, default = "/bigdat2/user/linsy/bigdat1/linsy/bfile/ECG/data/GWAS/ori_ECG_GWAS/british_V79.txt")
    parse.add_argument("--save_dir", type=str, default = "./data/V79")
    parse.add_argument("--train_FID_path", type=str, default = "/bigdat2/user/linsy/bigdat1/linsy/CMR/data/CMR_train_FID3.txt")
    parse.add_argument("--test_FID_path", type=str, default = "/bigdat2/user/linsy/bigdat1/linsy/CMR/data/CMR_test_FID3.txt")
    parse.add_argument("--bfile", type=str, default = "/bigdat2/user/linsy/bigdat1/linsy/CMR/data/bfile/brt_40k/brt_CMR_40k_QC")
    args = parse.parse_args()
    #clump_getSNP_LD(args.ECG_trait, args.GWAS_path, args.save_dir)
    #LD_PCA(args.ECG_trait, args.save_dir)

    PCA_split_train_test_padding(args.ECG_trait, args.save_dir, args.train_FID_path, args.test_FID_path, args.bfile)

test_FID_path = '/bigdat2/user/linsy/bigdat1/linsy/CMR/data/CMR_test_FID3.txt'
train_FID_path = '/bigdat2/user/linsy/bigdat1/linsy/CMR/data/CMR_train_FID3.txt'
bfile_path = '/bigdat2/user/linsy/bigdat1/linsy/CMR/data/bfile/brt_40k/brt_CMR_40k_QC'
