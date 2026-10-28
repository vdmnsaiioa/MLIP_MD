#!/bin/bash
#SBATCH --job-name=md_interface
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

# MD of a relaxed interface (scripts/md_interface.py), then the analysis (analyse_md.py).
# The interface comes from a run of mp_to_interf_relax.sh, which may have used another model.
# Resubmit the same job to continue after the wall-time limit; finished phases are skipped.

# ===== model and interface =====================================================================
MODEL=uma-omat              # pretrained model for the MD, as in mp_to_interf_relax.sh: a short name
                            #   (mace-mh, uma-omat, esen-omat, eqv2-omat, sevennet-ompa, grace-omat) or a
                            #   published name, e.g. EquiformerV2-31M-OMAT24, 7net-omni, GRACE-2L-OMAT-medium-base
CHECKPOINT=                 # optional local checkpoint file for MODEL's package
HEAD=                       # optional head/task/modal (e.g. mpa for SevenNet)
CUEQ=false                  # true: cuEquivariance kernels (MACE, SevenNet; must be installed),
                            #   less GPU memory: advised for SevenNet
INTERFACES=uma_interfaces   # interfaces folder of the run with the relaxed scan (e.g. mh1_interfaces)
INTERFACE=                  # empty = smallest lattice mismatch (as in mp_to_interf_relax.sh);
                            #   or e.g. 001_LCO_113_s1-bot__LPO_110_s2-bot
OUTDIR=                     # empty = <interface folder>/md_<MODEL> (md_uma-omat for MODEL=uma-s-1p2p1)

# ===== MD ======================================================================================
STAGES="anneal production"  # anneal, production or both
REPEAT="2 1"                # in-plane supercell
FIX_BOTTOM=2.0              # freeze LCO atoms within this distance of its bottom (A)
TIMESTEP=1.0                # fs
INTERVAL=100                # steps between saved frames and log lines
T_LOW=300                   # anneal: T_LOW -> ANNEAL_T -> T_LOW (K)
ANNEAL_T=900
ANNEAL_EQUIL_PS=10          # ps at T_LOW, heating, hold at ANNEAL_T, cooling
HEAT_PS=20
HOLD_PS=100
COOL_PS=50
TEMPERATURES="600 700 800 900"   # production temperatures (K)
EQUIL_PS=20                 # Langevin equilibration at each T (ps)
PROD_PS=500                 # production at each T (ps)
THERMOSTAT=bussi            # bussi | langevin (production)
TAUT=100                    # Bussi time constant (fs)
FRICTION=0.01               # Langevin friction (1/fs): anneal, equilibration
FROM=annealed               # production starts from the annealed structure, or "relaxed" (scan structure)

# ===== environment =============================================================================
shopt -s nocasematch       # conda environment of each package
case $MODEL in
    mace-*)                         ENV=vasp_stuff ;;
    uma-*)                          ENV=uma ;;
    esen-*|eqv2-*|equiformerv2-*)   ENV=fairchem1 ;;
    sevennet-*|7net-*)              ENV=sevenn ;;
    grace-*|grace_*)                ENV=grace ;;
    *) echo "no conda environment for MODEL=$MODEL: add it to the case list in this script"; exit 1 ;;
esac
shopt -u nocasematch
source /home/energy/s234633/anaconda3/etc/profile.d/conda.sh
conda activate $ENV
cd /home/energy/s234633/LCO_LPO_thesis || exit 1
export HF_HUB_OFFLINE=1
nvidia-smi

python scripts/md_interface.py --interfaces $INTERFACES --interface "$INTERFACE" --outdir "$OUTDIR" \
    --model $MODEL --checkpoint "$CHECKPOINT" --head "$HEAD" $([ "$CUEQ" = true ] && echo --cueq) \
    --stages $STAGES --repeat $REPEAT --fix-bottom $FIX_BOTTOM \
    --timestep $TIMESTEP --interval $INTERVAL --friction $FRICTION \
    --T-low $T_LOW --anneal-T $ANNEAL_T --anneal-equil-ps $ANNEAL_EQUIL_PS --heat-ps $HEAT_PS \
    --hold-ps $HOLD_PS --cool-ps $COOL_PS \
    --temperatures $TEMPERATURES --equil-ps $EQUIL_PS --prod-ps $PROD_PS --from $FROM \
    --thermostat $THERMOSTAT --taut $TAUT \
    --analyse
