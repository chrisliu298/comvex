for seed in $(seq 1 1 250)
do
echo "seed = $seed"
rm *.zip
rm -rf weights/ wandb/ lightning_logs/
python3 train.py \
    --project_name="mnist-cnn-collect" \
    --dataset="mnist" \
    --model_name="cnn-16x16x16" \
    --model_ckpt_path="weights" \
    --seed=$seed \
    --hw=28 \
    --channels=1 \
    --wandb
zip -qr weights$seed.zip weights/
cp *.zip /content/drive/Shareddrives/Embedding/mnist-cnn-zip/
done
for seed in $(seq 1 1 100)
do
echo "seed = $seed"
rm *.zip
rm -rf weights/ wandb/ lightning_logs/
python3 train.py \
    --project_name="cifar10-cnn-collect" \
    --dataset="cifar10" \
    --model_name="cnn-16x16x16" \
    --model_ckpt_path="weights" \
    --seed=$seed \
    --hw=32 \
    --channels=3 \
    --wandb
zip -qr weights$seed.zip weights/
cp *.zip /content/drive/Shareddrives/Embedding/cifar10-cnn-zip/
done