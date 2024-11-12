#!/bin/sh 

### General options 
### -- specify queue -- 
#BSUB -q gpua10
### -- set the job Name -- 
#BSUB -J Proposal_Analysis
### -- ask for number of cores (default: 1) -- 
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s243446@dtu.dk
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 
module load python3/3.12.4
source /zhome/91/9/214141/default_venv/bin/activate

# here follow the commands you want to execute with input.in as the input file

##nvidia-smi
torchrun --standalone --nproc_per_node=1 region_proposal_analysis.py