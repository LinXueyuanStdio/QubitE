
```
# WN18
bash runs.sh train Rotate3D wn18 0 0 512 256 1000 12.0 1.0 0.0001 80000 8 0 2 --disable_adv 

# FB15k
bash runs.sh train Rotate3D FB15k 0 0 1024 256 1000 24.0 0.5 0.00005 150000 8 0 2

# WN18RR
bash runs.sh train Rotate3D wn18rr 0 0 512 256 500 6.0 1.0 0.00005 80000 8 0.1 1 --disable_adv

# FB15k-237
bash runs.sh train Rotate3D FB15k-237 0 0 1024 256 1000 12.0 1.0 0.00005 100000 8 0 2
bash runs.sh train Rotate3D FB15k-237 0 0 256 256 500 12.0 1.0 0.00005 100000 16 0 2
git pull && CUDA_VISIBLE_DEVICES=3 bash runs.sh train Rotate3D FB15k-237 0 0 1024 256 200 12.0 1.0 0.00005 100000 8 0 2
```