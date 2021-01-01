#!/bin/sh

source /u/local/Modules/default/init/modules.sh
module load python/3.6.1

echo "Task id is $SGE_TASK_ID"

python3 main.py $SGE_TASK_ID