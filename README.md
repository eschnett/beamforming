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

```
$ ./cpu
beamforming.cpu
Setting up input data...
Forming beams...
Elapsed time: 175.684 seconds
Calculating checksum...
Checksum: 0x4de6498f
Checksum matches.
Done.
```

```
$ ./cpu2
beamforming.cpu2
Setting up input data...
Forming beams...
f=0
f=1
f=2
f=3
f=4
f=5
f=6
f=7
f=8
f=9
f=10
f=11
f=12
f=13
[...]
```
very slow.

```
$ ./cuda
beamforming.cuda
Setting up input data...
Forming beams...
Elapsed time: 0.047045 seconds
Calculating checksum...
Checksum: 0x4de6498f
Checksum matches.
Done.
```

`cuda2` is unfinished. it is an attempt to represent complex numbers
differently to simplify code.
