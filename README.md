# beamforming

Experiment with beam forming algorithms for CHORD

## Use on Graham

```sh
rsync -CPr $HOME/src/chord graham.computecanada.ca:work/src

module load StdEnv/2020
module load arch/avx512
module load cuda/11.0

srun --account=def-eschnett --gres=gpu:t4:1 --pty bash
```

## Use on Symmetry

```sh
rsync -CPr $HOME/src/chord symmetry:src

module load cuda
module load gcc/10
module load slurm

srun --partition=gpudebugq --gpus=1 --pty bash
```
