export PATH="bin:$PATH" # for docking use

CUDA_LAUNCH_BLOCKING=1 python run_rl.py --name='pocket0_denovo_vanilla' \
                       --exp_root='/mnt/2tb/experiments/freed/crem' \
                       --pocket_id=0 \
                       --load=0 --train=1 --has_feature=1 \
                       --name_full_load='' \
                       --min_action=1 --max_action=4 \
                       --graph_emb=1 --gcn_aggregate='sum' --gcn_type='GCN'\
                       --seed=141 --target='usp7' \
                       --update_after=3000 --start_steps=4000 --update_every=256 --init_alpha=1.0 \
                       --target_entropy=3. \
                       --desc='ecfp' \
                       --rl_model='sac' \
                       --gpu_id=1 --emb_size=64 --tau=1e-1 --batch_size=256 > /mnt/2tb/experiments/freed/logs/logs_crem/pocket0_denovo_vanilla.txt
