#!/bin/bash -l

#SBATCH --job-name=ensemble_forecasters
#SBATCH --output={logs_folder}/output_%j.txt
#SBATCH --error={logs_folder}/error_%j.txt
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_tasks_per_node}
#SBATCH --ntasks-per-socket=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00

cd {current_dir}
eval "$(micromamba shell hook --shell bash)"
micromamba activate ensemble_venv
module load mpi/OpenMPI/4.0.5-GCC-10.2.0
mpirun -n {world_size} python ensemble_of_forecasters.py {num_forecasters} {data_folder}