## NOTE: this script is mainly used for explanation. One needs to run step by step ##

################## Step 1.  Data Quality Control ###################
plink --bfile QCedSNPs --geno 0.0 --maf 0.01 --hwe 1e-5 midp include-nonctrl  --make-bed --out QCedSNPs.qc1

plink --bfile QCedSNPs.qc1 --het --test-missing midp --pfilter 1e-4 --make-bed --out QCedSNPs.qc2

################## Step 2.  Split data into training-test set  ###################

plink --bfile QCedSNPs.qc2 --keep train_FID.txt --make-bed --out QCedSNPs.qc2.train

plink --bfile QCedSNPs.qc2 --keep test_FID.txt --make-bed --out QCedSNPs.qc2.test

################## Step 3.  GWAS for training data ###################

./BOLT-LMM_v2.3.5/bolt \
--bfile=QCedSNPs.qc2.train --LDscoresFile=file_ld \
--lmm \
--phenoFile=file_pheno --phenoCol=trait \
--covarFile=file_pheno --qCovarCol=age --covarCol=sex --covarCol=center --covarCol=batch --covarMaxLevels=120 \
--modelSnps=file_modsnp \
--statsFile=file_out #QCedSNPs.qc2.train_GWAS.trait

################## Step 4.  Perform LD-PCA analysis ###################

python LD_PCA.py --ECG_trait trait --GWAS_path QCedSNPs.qc2.train_GWAS.trait --save_dir ./data/trait --train_FID_path train_FID.txt --test_FID_path test_FID.txt --bfile QCedSNPs.qc2