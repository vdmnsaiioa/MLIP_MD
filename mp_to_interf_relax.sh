#!/bin/bash
#SBATCH --job-name=mp_to_interf
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

# Steps 1-4 of the LCO | LPO workflow with one model (scripts/mp_to_interf_relax.py):
#   1 bulk + slabs from MP, relaxed -> 2 lattice matching -> 3 registry/gap scan of one interface
#   -> 4 relaxation of the scan structures (work of adhesion).   MD: md_interface.sh
# Every step resumes, so the job can simply be resubmitted. A run with other slab or interface
# settings needs its own RUN_NAME.

# ===== model and run ===========================================================================
MODEL=uma-omat             # mace-mh (MACE-MH-1, omat_pbe head) | uma-omat (UMA-S 1.2.1, omat task)
RUN_NAME=                  # folders <RUN_NAME>_surfaces, <RUN_NAME>_interfaces; empty = mh1 / uma
EARLY_STOP=None            # None = all 4 steps; 1, 2 or 3 = stop after that step

# ===== step 1: bulk and slabs ==================================================================
LCO_ID=mp-22526            # Materials Project ids (MP_API_KEY in ~/.bashrc; cells already saved in
LPO_ID=mp-13725            #   <RUN_NAME>_surfaces/bulk are reused without asking MP)
LCO_FILE=                  # or local bulk files instead of MP, e.g. the cells of the MACE run:
LPO_FILE=                  #   mh1_surfaces/bulk/LCO_mp-22526_input_conv.vasp, .../LPO_mp-13725_input_conv.vasp
N_FACETS=8                 # facets per material, best Donnay-Harker ranked, Miller indices up to MAX_INDEX
MAX_INDEX=4
LCO_MILLERS=auto           # auto = ranked list, or explicit facets, e.g. "001 104 012"
LPO_MILLERS=auto
EXTRA_LCO_MILLERS=         # facets added to the list, e.g. "100 214"
EXTRA_LPO_MILLERS=
LCO_THICKNESS=15           # minimum slab thickness (A)
LPO_THICKNESS=15
SLAB_VACUUM=20             # vacuum of the slab cells (A)
FTOL=0.1                   # atoms closer than this along the normal form one layer (A)
BONDS="P-O:2.0"            # slabs cutting these bonds are not used ("none" = keep all)
KEEP_BROKEN=false
BULK_FMAX=0.005            # force/stress convergence, bulk (eV/A)
SLAB_FMAX=0.02             # force/stress convergence, slabs (eV/A)
SLAB_MAX_STEPS=2000
SLAB_FIX_BOTTOM=0          # freeze slab atoms within this distance of the bottom (A); 0 = none

# ===== step 2: lattice matching ================================================================
MAX_STRAIN=0.01            # mismatch tolerance: largest principal strain of LPO on LCO (0.01 = 1 %)
MAX_ATOMS=1000             # largest interface
MATCHES_PER_PAIR=2         # cells per slab pair and orientation (smallest, then lower-strain ones)
SEPARATION=2.0             # LCO-LPO gap of POSCAR_interface (A); the scan sets its own gaps
LCO_VACUUM=10              # vacuum below LCO / above LPO in the interface cell (A)
LPO_VACUUM=10
INCLUDE_UNCONVERGED=false  # also use slabs whose relaxation did not converge

# ===== step 3: registry and gap scan ===========================================================
INTERFACE=                 # empty = smallest lattice mismatch; or e.g. 001_LCO_113_s1-bot__LPO_110_s2-bot
GRID="6 6"                 # lateral shifts N1 x N2 over the minimal shift cell
GAPS="1.5 2.0 2.5 3.0"     # LCO-LPO gaps (A)
MIN_CONTACT=1.6            # skip structures with a closer LCO-LPO contact (A)
SHIFT_CELL=minimal         # minimal | lpo | interface

# ===== step 4: interface relaxations ===========================================================
INTERFACE_FMAX=0.02        # force convergence (eV/A)
INTERFACE_MAX_STEPS=2000
INTERFACE_FIX_BOTTOM=0     # freeze LCO atoms within this distance of its bottom (A); 0 = none
TOP=0                      # relax only the TOP lowest single points; 0 = all

# ===== environment =============================================================================
case $MODEL in
    mace-mh)  ENV=vasp_stuff ;;
    uma-omat) ENV=uma ;;
    *) echo "set the conda environment for MODEL=$MODEL in this script"; exit 1 ;;
esac
source /home/energy/s234633/anaconda3/etc/profile.d/conda.sh
conda activate $ENV
cd /home/energy/s234633/LCO_LPO_thesis || exit 1
export HF_HUB_OFFLINE=1    # UMA weights are downloaded once on the login node, not in the job
nvidia-smi

python scripts/mp_to_interf_relax.py \
    --model $MODEL --run-name "$RUN_NAME" --early-stop $EARLY_STOP \
    --lco-id $LCO_ID --lpo-id $LPO_ID --lco-file "$LCO_FILE" --lpo-file "$LPO_FILE" \
    --n-facets $N_FACETS --max-index $MAX_INDEX \
    --lco-millers $LCO_MILLERS --lpo-millers $LPO_MILLERS \
    --extra-lco-millers $EXTRA_LCO_MILLERS --extra-lpo-millers $EXTRA_LPO_MILLERS \
    --lco-thickness $LCO_THICKNESS --lpo-thickness $LPO_THICKNESS --slab-vacuum $SLAB_VACUUM \
    --ftol $FTOL --bonds $BONDS --keep-broken $KEEP_BROKEN \
    --bulk-fmax $BULK_FMAX --slab-fmax $SLAB_FMAX --slab-max-steps $SLAB_MAX_STEPS \
    --slab-fix-bottom $SLAB_FIX_BOTTOM \
    --max-strain $MAX_STRAIN --max-atoms $MAX_ATOMS --matches-per-pair $MATCHES_PER_PAIR \
    --separation $SEPARATION --lco-vacuum $LCO_VACUUM --lpo-vacuum $LPO_VACUUM \
    --include-unconverged $INCLUDE_UNCONVERGED \
    --interface "$INTERFACE" --grid $GRID --gaps $GAPS --min-contact $MIN_CONTACT --shift-cell $SHIFT_CELL \
    --interface-fmax $INTERFACE_FMAX --interface-max-steps $INTERFACE_MAX_STEPS \
    --interface-fix-bottom $INTERFACE_FIX_BOTTOM --top $TOP
