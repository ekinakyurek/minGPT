# minGPT Fully Sharded Example

Modified [Andrej's](https://github.com/karpathy/minGPT) and [William's](https://github.com/williamFalcon/minGPT) awesome code to create a simple example and benchmarking script.

### Usage

```
pip install -r requirements.txt
```

```
python benchmark.py --strategy fsdp --gpus 8
```