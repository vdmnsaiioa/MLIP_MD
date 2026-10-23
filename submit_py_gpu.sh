#!/bin/bash
#SBATCH --job-name=mace_lco_lpo
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s234633@dtu.dk
#SBATCH --partition=sm3090el8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=07-00:00:00

#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=errs/slurm_%j_errors.log

# Activate environment
source /home/energy/s234633/anaconda3/etc/profile.d/conda.sh
conda activate vasp_stuff

cd /home/energy/s234633/LCO_LPO_thesis || exit 1

# MP_API_KEY should be exported before submitting the job
export MP_API_KEY="pSyZlL5P0Pl3HiHMoNUuTRtmpBPyDRGO"

# GPU diagnostics
nvidia-smi

#python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

#python scripts/relax_lco_lpo_surfaces.py --lco-thickness 15 --lpo-thickness 15 --vacuum 20

#python scripts/match_relaxed_surfaces.py mh1_surfaces --max-strain 0.01 --max-atoms 1000

#python scripts/scan_interface_registry.py mh1_interfaces/001_LCO_113_s1-bot__LPO_110_s2-bot_432at --grid 6 6 --gaps 1.5 2.0 2.5 3.0 --min-contact 1.6

#python scripts/relax_interface_scan.py mh1_interfaces/001_LCO_113_s1-bot__LPO_110_s2-bot_432at/registry_scan

python scripts/md_interface.py mh1_interfaces/001_LCO_113_s1-bot__LPO_110_s2-bot_432at/registry_scan --repeat 2 1