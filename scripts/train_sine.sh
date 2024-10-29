export PATH=/usr/local/nvidia/bin:$PATH

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=3
#export NCCL_IB_HCA=mlx5_2,mlx5_5  # A100
export NCCL_IB_HCA=$(ibv_devices |tail -1 |awk '{print $1}') # V100
export NCCL_DEBUG=info
export OMP_NUM_THREADS=4
# ulimit -l 131072
export JOB_NAME=$(cat /etc/hostname | cut -d '-' -f 1,2,3)
#export MASTER_FILE=/home/yangliu/master_ip.${JOB_NAME}

# trap finish EXIT INT TERM

#export MASTER_ADDR=$(cat ${MASTER_FILE})
echo "master_ip: $MASTER_ADDR"
DIST_URL="tcp://$MASTER_ADDR:60900"


lr=1e-4
bs=2
uf=4
gpus=8
ep=50

model=vit_large
model_path=models/dinov2_vitl14_pretrain.pth

SUB_DIR=pt_sine_lr${lr}_bs${bs}_uf${uf}_gpu${gpus}_ep${ep}
echo ${SUB_DIR}

python tools/train.py \
    --num-gpus ${gpus} \
    --num-machines ${WORLD_SIZE} \
    --machine-rank ${RANK} \
    --dist-url ${DIST_URL} \
    --batch_size ${bs} \
    --epochs ${ep} \
    --update_freq ${uf} \
    --dinov2-size ${model} \
    --dinov2-weights ${model_path} \
    --save_ckpt_freq 5 \
    --dataset "pano_seg||ins_seg" \
    --sample_rate "2,5" \
    --pano_seg_data "coco||ade20k" \
    --pano_sample_rate "1,1" \
    --ins_seg_data "coco||o365" \
    --ins_sample_rate "1,3" \
    --transformer_num_queries 200 \
    --transformer_fusion_layer_depth 1 \
    --crop_ratio 0.5 \
    --load_dir "" \
    --output_dir ./outputs/pt_dir/${SUB_DIR} \
    --log_dir ./outputs/pt_dir/${SUB_DIR} \
    --num_workers 2 \
    --auto_resume \
    --lr ${lr}