# pertrub 不一定 就 optimize pose 了
# 同样，不一定只有perturb才需要optimize，有时候在colmap基础上也要optimzie
# 这俩选项要一起用才好

CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_optimize_pose_0.0001 \
    --perturb \
    --optimize_pose


CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5679 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/gt


CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5679 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_no_optimize
    --perturb



CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5679 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_no_optimize \
    --perturb 

### perturbed_optimize_noise_001_001
CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5679 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_optimize_noise_001_001_log_warp_error \
    --perturb \
    --optimize_pose \
    --noise_r 0.01 \
    --noise_t 0.01

### perturbed_optimize_noise_0001_0001
CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5679 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_optimize_noise_0001_0001 \
    --perturb \
    --optimize_pose \
    --noise_r 0.001 \
    --noise_t 0.001

### perturbed_optimize_noise_01_01
CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_optimize_noise_01_01 \
    --perturb \
    --optimize_pose \
    --noise_r 0.1 \
    --noise_t 0.1

CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5679 render.py \
    -m output/perturbed_optimize_noise_001_001 \
    --iteration -1

CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5679 metrics.py \
    -m output/perturbed_optimize_noise_001_001

# perturbed_no_optimize_noise_0001_0001
CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_no_optimize_noise_0001_0001 \
    --perturb \
    --noise_r 0.001 \
    --noise_t 0.001

# perturbed_no_optimize_noise_001_001
CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_no_optimize_noise_001_001_logs \
    --perturb \
    --noise_r 0.01 \
    --noise_t 0.01

CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 render.py \
    -m output/perturbed_no_optimize_noise_001_001 \
    --iteration -1

CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 metrics.py \
    -m output/perturbed_no_optimize_noise_001_001

# perturbed_no_optimize_from_identity
CUDA_VISIBLE_DEVICES=0 python -m debugpy --listen 5678 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_no_optimize_from_identity \
    --perturb \
    --noise_r 0.01 \
    --noise_t 0.01 \
    --identity

# perturbed_optimize_from_identity
CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 5679 train.py \
    --eval \
    -s nerf_synthetic/lego \
    -m output/perturbed_optimize_from_identity \
    --perturb \
    --noise_r 0.01 \
    --noise_t 0.01 \
    --identity \
    --optimize_pose