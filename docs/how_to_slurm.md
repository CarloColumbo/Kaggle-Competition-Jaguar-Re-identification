### Help for SLURM
- saccount: Get account name
- squeue --m: Get my jobs
- scancel id: cancel job
- salloc -A sci-demelo-computer-vision: alloc job
- salloc -A sci-demelo-computer-vision --partition=gpu-interactive --mem=16gb --cpus-per-task=2 --time=08:00:00 --gpus=1
- jupyter notebook --port 8888: start jupyter
- ssh karl.schuetz@gx03.hpc.sci.hpi.de -N -L 8888:localhost:8888: forward port