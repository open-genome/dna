```
git clone git@github.com:gersteinlab/dna.git
cd dna
mkdir data
cd data
wget https://bydna.s3.us-east-2.amazonaws.com/bert_hg38.tar.gz
tar -xzvf bert_hg38.tar.gz
wget https://bydna.s3.us-east-2.amazonaws.com/dnabert2_bin.tar.gz
tar -xzvf dnabert2_bin.tar.gz
```

Change the dataset.text_file for configs/experiment/xxx/xxx.yaml

### RUN
```
conda create -n dna python==3.8
conda activate dna
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pip install setuptools==69.5.1 
pip install pytorch-lightning==1.8.6
pip install packaging
pip install flash_attn==1.0.7 --no-build-isolation
python train.py experiment='dnabert2/dnabert2_hg38_pretrain' wandb=null
```
