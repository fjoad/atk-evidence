#!/bin/bash
#SBATCH -p gpu-all
#SBATCH --gres=gpu:1

cd /export/home/fjoad/atk-evidence
"$@"
