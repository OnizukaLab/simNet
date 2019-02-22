#!/bin/bash
#$ -l rt_G.large=1
#$ -ar 916
#$ -m e
#$ -cwd

source /etc/profile.d/modules.sh
module load singularity/2.6.1
singularity exec --nv simnet.img sh runsimnettrain.sh
