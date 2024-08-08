```
git clone git@github.com:gersteinlab/dna.git
cd dna
wget https://bydna.s3.us-east-2.amazonaws.com/data.tar.gz
tar -xzvf data.tar.gz
```

Change the dataset.text_file for configs/experiment/xxx/xxx.yaml

### RUN
```
conda create -n dna python==3.8
conda activate dna
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install setuptools==69.5.1
pip install flash_attn==1.0.7 --no-build-isolation
python train.py experiment='dnabert2/dnabert2_pretrain'
```
