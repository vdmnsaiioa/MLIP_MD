#!/bin/bash
#SBATCH --job-name=electra_MM
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s234633@dtu.dk  # The default value is the submitting user.
#SBATCH --partition=xeon24el9_test
#SBATCH --nodes=1
#SBATCH --time=00-00:10:00
#SBATCH --ntasks=1
##SBATCH --gres=gpu:1

#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=errs/slurm_%j_errors.log

# Debugging flags (optional)

source /home/energy/s234633/anaconda3/etc/profile.d/conda.sh
conda activate vasp_stuff
cd /home/energy/s234633/LCO_LPO_thesis || exit 1

export MP_API_KEY="pSyZlL5P0Pl3HiHMoNUuTRtmpBPyDRGO"
#python3 -m scripts.lattice_match
python scripts/relaxed_surfaces_mace_polar.py --model polar-1-m --device cuda \
    --lco-thickness 15 --lpo-thickness 15 --vacuum 20
