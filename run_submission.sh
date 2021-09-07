{
    # ADD POS
    # EMBED, ADD_FEATS, GRAPHLSTM
    sleep 5 &&
    sudo CUDA_VISIBLE_DEVICES=3 /home/fiona/anaconda3/envs/torchgeom/bin/python3 run.py \
    --data_dir data --model_name bert-large-cased --output_dir "outs/embed/08" --max_seq_length 350 \
    --num_train_epochs 10 --per_gpu_train_batch_size 4 --save_steps 5000 --seed 123 \
    --do_train --do_predict --overwrite_output_dir --use_graph  --graph_purpose embed \
    --graph_hidden 1024 --graph_out 512 --add_feats --add_pos --use_graphlstm
}