#!/bin/bash
#$ -l rt_G.large=1
#$ -l h_rt=42:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load singularity/2.6.1
singularity exec --nv simnet.img sh runsimnettrain.sh
