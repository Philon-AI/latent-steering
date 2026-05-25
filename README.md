# Latent Steering

> Steering action experts with future state representations.

This repository contains the reference implementation of **Latent Steering**, a training paradigm in which a flow-matching action expert is conditioned on a compressed encoding of *future* frames rather than on natural-language instructions. By giving the future-stream pooler a much smaller token budget than the past-stream pooler, the steering signal is forced to carry roughly the information content of a short instruction, while remaining fully self-supervised.

The full technical report is included in this repository as [`technical-report.pdf`](technical-report.pdf).

## Prerequisites

Install Miniconda:

```bash
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash
```

Install AzCopy:

```bash
wget -q https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
tar -xf azcopy.tar.gz
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/local/bin/
```

Install ffmpeg:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

## Setup

Create the environment:

```bash
conda create -n latent-steering python=3.12 -y
conda activate latent-steering
```

Install latent-steering:

```bash
git clone https://github.com/Philon-AI/latent-steering
cd latent-steering && pip install -r requirements.txt && cd ..
```

Install V-JEPA2:

```bash
git clone https://github.com/Philon-AI/vjepa2
cd vjepa2 && pip install -e . && cd ..
```

## Pretrained Checkpoint

Download the V-JEPA2 pretrained checkpoint to `latent-steering/pretrained`:

```bash
wget -q https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt -O latent-steering/pretrained/vjepa2_1_vitl_dist_vitG_384.pt
```

## Dataset

```bash
azcopy copy "https://philonaimds.blob.core.windows.net/gold/MDS_LeRobot_10000.tar?sp=r&st=2026-04-28T04:29:34Z&se=2026-04-28T12:44:34Z&spr=https&sv=2025-11-05&sr=b&sig=NsFjlGiJHbBZI9I%2FtKWGWCt43cXAnAKU2uQwea0E5SA%3D" "MDS_LeRobot_10000.tar"
tar -xf MDS_LeRobot_10000.tar
mv MDS_LeRobot latent_steering_LeRobot
```

The dataset directory `latent_steering_LeRobot` is what [`configs/default.yaml`](configs/default.yaml) expects under `dataset_dir`.

## Train

```bash
cd latent-steering
OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 train.py
```

## Evaluate

```bash
cd latent-steering
python eval.py --checkpoint <dcp_dir>
```

## Known Issues

- [ ] `dataset.py`: flip (img + action), crop augmentations, ...

## Citation

```bibtex
@techreport{philon2026latentsteering,
  title  = {Latent Steering: Steering Action Experts with Future State Representations},
  author = {Philon},
  year   = {2026},
  type   = {Technical Report},
  url    = {https://github.com/Philon-AI/latent-steering},
}
```
