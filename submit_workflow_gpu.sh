#!/bin/bash
#SBATCH --job-name=lco_lpo_mlip
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

# LCO | LPO workflow with a selectable model. Set MODEL and STEP here, or when submitting:
#   sbatch --export=ALL,MODEL=uma,STEP=surfaces submit_workflow_gpu.sh
# Steps, in this order (each one needs the output of the one before):
#   surfaces -> match -> scan -> relax -> md -> analyse

MODEL=${MODEL:-uma}      # uma   : UMA-S 1.2.1 (uma-s-1p2p1), omat task -> uma_surfaces, uma_interfaces
                         # mace  : MACE-MH-1, omat_pbe head              -> mh1_surfaces, mh1_interfaces
                         # uma-m : UMA-M 1.1 (uma-m-1p1), omat task     -> umam_surfaces, umam_interfaces
STEP=${STEP:-surfaces}   # surfaces | match | scan | relax | md | analyse
PAIR=${PAIR:-LCO_113_s1-bot__LPO_110_s2-bot}   # interface for scan/relax/md/analyse (001 of the MACE run)
TEMPS=${TEMPS:-}         # md only: e.g. TEMPS=800 runs just that temperature (needs the anneal done)

case $MODEL in
    mace)  ENV=vasp_stuff; PREFIX=mh1;  MODEL_ARGS="--backend mace" ;;
    uma)   ENV=uma;        PREFIX=uma;  MODEL_ARGS="--backend uma" ;;
    uma-m) ENV=uma;        PREFIX=umam; MODEL_ARGS="--backend uma --model uma-m-1p1" ;;
    *) echo "unknown MODEL=$MODEL (mace, uma or uma-m)"; exit 1 ;;
esac

source /home/energy/s234633/anaconda3/etc/profile.d/conda.sh
conda activate $ENV
cd /home/energy/s234633/LCO_LPO_thesis || exit 1
export HF_HUB_OFFLINE=1  # UMA weights are downloaded once on the login node, not in the job

SURF=${PREFIX}_surfaces
IFACES=${PREFIX}_interfaces
REF=mh1_surfaces         # first run: its MP cells, facets and settings are reused for every model
IFACE=$(ls -d $IFACES/[0-9][0-9][0-9]_${PAIR}* 2>/dev/null | head -1)
need_iface() {
    [ -n "$IFACE" ] || { echo "no $PAIR interface in $IFACES/ - run STEP=match first or pick PAIR from $IFACES/interfaces.csv"; exit 1; }
    echo "interface: $IFACE"
}

echo "MODEL=$MODEL  STEP=$STEP  conda env: $ENV"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '- GPU available:', torch.cuda.is_available())"

case $STEP in
    surfaces)  # 1. relax bulk (atoms + cell) and all slabs (atoms + in-plane lattice) -> $SURF
        if [ -f $REF/pairs.csv ]; then
            INPUT="--like $REF"   # same MP cells, facets and settings as the first run
        else                      # no earlier run: query MP (needs mp-api and MP_API_KEY in ~/.bashrc)
            INPUT="--lco-id mp-22526 --lpo-id mp-13725 --lco-thickness 15 --lpo-thickness 15 --vacuum 20"
        fi
        # + LCO (0 1 2) and (0 2 1), which the first facet list missed (MACE: added to mh1_surfaces in place)
        python scripts/relax_lco_lpo_surfaces.py $MODEL_ARGS $INPUT --extra-lco-millers 012 021 --outdir $SURF ;;
    match)     # 2. lattice matching of the relaxed slabs, <= 1 % strain, <= 1000 atoms (no model) -> $IFACES
        python scripts/match_relaxed_surfaces.py $SURF --max-strain 0.01 --max-atoms 1000 --outdir $IFACES ;;
    scan)      # 3. registry (lateral shift) x gap structures of the interface (no model)
        need_iface
        python scripts/scan_interface_registry.py $IFACE --grid 6 6 --gaps 1.5 2.0 2.5 3.0 --min-contact 1.6 ;;
    relax)     # 4. single points + relaxations of all scan structures, work of adhesion -> results.csv
        need_iface
        python scripts/relax_interface_scan.py $IFACE/registry_scan $MODEL_ARGS ;;
    md)        # 5. anneal, then NVT at 600-900 K from the lowest-energy structure (resubmit to continue)
        need_iface
        python scripts/md_interface.py $IFACE/registry_scan $MODEL_ARGS --repeat 2 1 \
            ${TEMPS:+--stages production --temperatures $TEMPS} ;;
    analyse)   # 6. stability and Li transport from the MD (no model)
        need_iface
        python scripts/analyse_md.py $IFACE/md ;;
    *) echo "unknown STEP=$STEP (surfaces, match, scan, relax, md, analyse)"; exit 1 ;;
esac
