```
git clone git@github.com:gersteinlab/dna.git
cd dna
mkdir data
aws s3 sync  s3://bydna/ ./
```

Change the dataset.text_file for configs/experiment/xxx/xxx.yaml

### RUN
```
conda create -n dna python==3.8
conda activate dna
pip install -r requirement.txt
pip install flash_attn==1.0.7 --no-build-isolation
python train.py experiment='dnabert2/dnabert2_pretrain'
```
