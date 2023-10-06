export load_path=REPLACE_WITH_PACE_PATH
export dim=64
export image_size=128
export timesteps=1000
export sampling_timesteps=250
export objective=pred_v
export beta_schedule=cosine
export channels=3
export resnet_block_groups=8
export learned_sinusoidal_dim=16
export attn_dim_head=32         # DOESN'T MATTER, we are not using self-attention
export attn_heads=1             # DOESN'T MATTER, we are not using self-attention
export batch_size=32
export lr=0.0008
export num_steps=700000
export gradient_accumulate_every=2
export ema_decay=0.995
export save_and_sample_every=70000


python \
    train.py \
    --dim $dim \
    --image_size $image_size \
    --timesteps $timesteps \
    --sampling_timesteps $sampling_timesteps \
    --objective $objective \
    --beta_schedule $beta_schedule \
    --channels $channels \
    --resnet_block_groups $resnet_block_groups \
    --learned_sinusoidal_dim $learned_sinusoidal_dim \
    --attn_dim_head $attn_dim_head \
    --attn_heads $attn_heads \
    --load_path $load_path \
    --batch_size $batch_size \
    --lr $lr \
    --num_steps $num_steps \
    --gradient_accumulate_every $gradient_accumulate_every \
    --ema_decay $ema_decay \
    --save_and_sample_every $save_and_sample_every \
    --wandb         # disable this one (comment it out if this is not allowed in PACE)