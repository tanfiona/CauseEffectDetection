{
    # EMBED, ADD_FEATS, GRAPHLSTM
    sleep 5 &&
    sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
    --data_dir data --model_name bert-large-cased --output_dir "outs/embed/08" --max_seq_length 350 \
    --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
    --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose embed \
    --graph_hidden 1024 --graph_out 512 --add_feats --add_pos --use_graphlstm
}

# # TRAIN ONE on combined training set
# # Note: Transfer to server on "Default", do not use "Binary"
# {
#     # BASE
#     sudo CUDA_VISIBLE_DEVICES=0 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/embed/01" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir
# } &
# {
#     # EMBED
#     sleep 300 &&
#     sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/embed/02" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose embed \
#     --graph_hidden 1024 --graph_out 512
# } & 
# {
#     # EMBED, ADD_FEATS
#     sleep 300 &&
#     sudo CUDA_VISIBLE_DEVICES=2 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/embed/03" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 3000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose embed \
#     --graph_hidden 1024 --graph_out 512 --add_feats
# } & 
# {
#     # EMBED, ADD_FEATS, GRAPHLSTM
#     sleep 300 &&
#     sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/embed/04" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose embed \
#     --graph_hidden 1024 --graph_out 512 --add_feats --use_graphlstm
# }

# ### ADD_POS ###
# {
#     # BASE
#     sudo CUDA_VISIBLE_DEVICES=0 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/embed/05" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir --add_pos
# } &  
# {
#     # EMBED
#     sleep 5 &&
#     sudo CUDA_VISIBLE_DEVICES=1 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/embed/06" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose embed \
#     --graph_hidden 1024 --graph_out 512 --add_pos
# } & 
# {
#     # EMBED, ADD_FEATS
#     sleep 5 &&
#     sudo CUDA_VISIBLE_DEVICES=2 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/embed/07" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose embed \
#     --graph_hidden 1024 --graph_out 512 --add_feats --add_pos
# } & 
# {
#     # EMBED, ADD_FEATS, GRAPHLSTM
#     sleep 5 &&
#     sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/embed/08" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose embed \
#     --graph_hidden 1024 --graph_out 512 --add_feats --add_pos --use_graphlstm
# }



# {
#     # POST, ADD_POS, EMBED, ADD_FEATS
#     sleep 5 &&
#     sudo CUDA_VISIBLE_DEVICES=2 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/post/07" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose post \
#     --graph_hidden 1024 --graph_out 512 --add_feats --add_pos
# } & 
# {
#     # CLF, ADD_POS, EMBED, ADD_FEATS
#     sleep 5 &&
#     sudo CUDA_VISIBLE_DEVICES=2 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
#     --data_dir data --model_name bert-base-cased --output_dir "outs/classifier/07" --max_seq_length 350 \
#     --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
#     --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose classifier \
#     --graph_hidden 1024 --graph_out 512 --add_feats --add_pos
# }