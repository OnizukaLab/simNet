#!/bin/bash
#$ -l rt_F=1
#$ -ar 916
#$ -m e
#$ -cwd

source /etc/profile.d/modules.sh
module load singularity/2.6.1
singularity exec --nv simnet_java8.img sh runsimnettest.sh
