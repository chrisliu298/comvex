# %%shell
for i in $(seq 0 1 99)
do
rm -rf model_ckpt/ wandb/ lightning_logs/
python3 train.py \
    --project_name="ep-network" \
    --emb_dim=16 \
    --max_epochs=1000 \
    --patience=20 \
    --verbose=0 \
    --wandb
done