#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load singularity/2.6.1
singularity exec --nv viscon.img sh runvistest.sh
