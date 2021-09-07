# PAIRED - KFOLDS on combined training set
# Note: Transfer to server on "Default", do not use "Binary"
{
    # BASE
    sudo CUDA_VISIBLE_DEVICES=0 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
    --data_dir data --model_name bert-base-cased --output_dir "outs/embed/01" --max_seq_length 350 \
    --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 916 \
    --do_train --folds 3 --overwrite_output_dir
} &
{
    # BASE
    sleep 500 &&
    sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
    --data_dir data --model_name bert-base-cased --output_dir "outs/embed/01" --max_seq_length 350 \
    --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 703 \
    --do_train --folds 3 --overwrite_output_dir
}  &
{
    # BASE
    sleep 500 &&
    sudo CUDA_VISIBLE_DEVICES=2 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
    --data_dir data --model_name bert-base-cased --output_dir "outs/embed/01" --max_seq_length 350 \
    --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 443 \
    --do_train --folds 3 --overwrite_output_dir
} &
{
    # BASE
    sleep 500 &&
    sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
    --data_dir data --model_name bert-base-cased --output_dir "outs/embed/01" --max_seq_length 350 \
    --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 229 \
    --do_train --folds 3 --overwrite_output_dir
} &
{
    # BASE
    sleep 500 &&
    sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
    --data_dir data --model_name bert-base-cased --output_dir "outs/embed/01" --max_seq_length 350 \
    --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 585 \
    --do_train --folds 3 --overwrite_output_dir
}