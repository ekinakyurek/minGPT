# minGPT DeepSpeed

Modified [Andrej's](https://github.com/karpathy/minGPT) and [William's](https://github.com/williamFalcon/minGPT) awesome code to provide a benchmark script and show how to train a model from scratch using DeepSpeed.

### Usage

```
pip install -r requirements.txt
```

### Train Billion Parameter Models using DeepSpeed

In the below examples batch size is set to 1 to try reduce VRAM as much as possible, but you can scale that with your compute. In the below case we could scale the batch size significantly to fill the left over GPU memory.

For 20B/45B parameter models, you'll need a reasonable amount of CPU RAM as we offload partitions to the CPU. For the 45B parameter model, you'll need around 1TB of CPU memory which is the default for the p4d.24xlarge instance in AWS (roughly 9 dollars an hour for a spot instance).

##### 453M (Requires around 3GiB per 8 GPUs, 10GiB for 1 GPU)

In this example we use Stage 2 by default, since it's faster (remains the same speed parity as DDP, but with more memory savings with multiple GPUs).

If you want even more savings with multiple GPUs, swap `--strategy deepspeed_stage 3`.

```bash
python train.py --n_layer 4 --n_head 16 --n_embd 3072 --gpus 1 --precision 16 --batch_size 1 --strategy deepspeed_stage_2
```

##### 1.7B (Requires around 2GiB per 8 GPUs, 5.1GiB for 1 GPU)
```bash
python train.py --n_layer 15 --n_head 16 --n_embd 3072 --gpus 8 --precision 16 --batch_size 1 --strategy deepspeed_stage_3
```

##### ~10B (Requires around 6GiB per 8 GPUs, 26GiB for 1 GPU)
```bash
python train.py --n_layer 13 --n_head 16 --n_embd 8192 --gpus 8 --precision 16 --batch_size 1 --strategy deepspeed_stage_3
```

##### ~20B (Requires around 8GiB per 8 GPUs, OOM for 1 GPU, offloading onto ~500GB of CPU RAM)
```bash
python train.py --n_layer 25 --n_head 16 --n_embd 8192 --gpus 8 --precision 16 --batch_size 1 --strategy deepspeed_stage_3
```

##### ~45B (Requires around 14GiB per 8 GPUs, 26GiB for 1 GPU, offloading onto ~950GB of CPU RAM)
```bash
python train.py --n_layer 56 --n_head 16 --n_embd 8192 --gpus 8 --precision 16 --batch_size 1 --strategy deepspeed_stage_3
```

### Model Loading and Evaluation
The best model is checkpointed during the training process and stored by default in "checkpoints" directory. With DeepSpeed the model checkpoints are saved as directories, which can cause some issues when trying to load model/trainers from Pytorch Lightning checkpoints. To properly restore the model and run trainer tests, call the evaluate.py file with similar arguments to the train script: 

```bash
python evaluate.py --gpus 1 --precision 16 --batch_size 1 --strategy deepspeed_stage_2
```

This will first convert the model checkpoint directory into a single model .pt file, then load the trainer using deepspeed_stage_2, and run the test set. For simplicity of this example, the test set is identical to the training set.  

### Benchmark Results

#### Maximum DeepSpeed!

Largest model I could fit onto the server. 

##### DeepSpeed ZeRO Stage 3

```
~20B
python benchmark.py --n_layer 21 --n_head 16 --n_embd 8192 --gpus 8 --precision 16 --limit_train_batches 120 --batch_size 1

Average Epoch time: 45.65 seconds
Average Peak CUDA memory: 36086.14MiB
```

##### DeepSpeed ZeRO Stage 3 Offload

```
~45B
python benchmark.py --n_layer 56 --n_head 16 --n_embd 8192 --gpus 8 --precision 16 --limit_train_batches 120 --cpu_offload

Average Epoch time: 3039.97 seconds
Average Peak CUDA memory: 14186.86MiB
CPU Virtual Memory:  used = 867.61 GB, percent = 91.8%
```

#### Smaller Model Comparison, DDP vs DeepSpeed

We collected results using a model size that fit training with DDP (roughly 1.6B parameters). 

This benchmark simulates the improvement in memory when training larger models, which is useful for users that do not have access to high memory GPUs.

A technical note: when using DeepSpeed, I noticed that for the first 20 batches, the optimizer step were skipped as infs were detected. 
Also as the memory decreases **we can achieve the same speed by increasing the batch size as DDP**. The reason we do not test this as we're simulating low resource environments to see the lowest VRAM memory usage possible. 

Command:
```bash
1.7B
python benchmark.py --n_layer 15 --n_head 16 --n_embd 3072 --gpus 8 --precision 16 --limit_train_batches 128 --batch_size 1
```

##### DDP
```
Average Epoch time: 43.69 seconds
Average Peak CUDA memory: 36148.55MiB
```

##### DeepSpeed ZeRO Stage 3 + FusedAdam
```
Average Epoch time: 45.05 seconds
Average Peak CUDA memory: 7796.98MiB
```

##### DeepSpeed ZeRO Stage 3 Offload + DeepSpeedCPUAdam
```
Average Epoch time: 248.39 seconds
Average Peak CUDA memory: 4976.98MiB
```

##### DeepSpeed ZeRO Stage 3 Offload + DeepSpeedCPUAdam + Activation Checkpointing
```
Average Epoch time: 256.91 seconds
Average Peak CUDA memory: 2192.98MiB
```
