```
git clone git@github.com:gersteinlab/dna.git
cd dna
mkdir data
aws s3 sync  s3://bydna/ ./
```

Change the dataset.text_file for configs/experiment/xxx/xxx.yaml

### RUN
```
python train.py experiment='dnabert2/dnabert2_pretrain'
```
