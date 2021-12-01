import math
import os
from argparse import ArgumentParser
import pathlib
from pytorch_lightning import Trainer
from mingpt.model import GPT
from pathlib import Path
from train import CharDataset
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import seed_everything

# here we convert the directory into a single model weight checkpoint inside the directory
# https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#collating-single-file-checkpoint-for-deepspeed-zero-stage-3
# Note: trainer.fit(ckpt_path=SINGLE_FILE) will not work. Instead, you have to point at the checkpoint directory
def convert_directory_chkpt_to_file(mpath):
    from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
    dir_path = Path(mpath)
    model_path = Path(mpath, 'model_checkpoint.pt')
    if dir_path.is_dir() and not model_path.exists():
        # convert directory to single model pytorch 
        convert_zero_checkpoint_to_fp32_state_dict(dir_path, model_path)
    return model_path

if __name__ == '__main__':
    seed_everything(42)
    
    import os
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--checkpoint', default="./checkpoints/last.ckpt", type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    args = parser.parse_args()

    print("loading model checkpoint")
    chk_path = convert_directory_chkpt_to_file(args.checkpoint)
    
    # this loads the model
    model = GPT.load_from_checkpoint(chk_path)
    print("finished loading model with config: ", model.config)

    # this loads the trainer
    trainer = Trainer.from_argparse_args(args)

    if not os.path.exists("input.txt"):
        os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

    text = open('input.txt', 'r').read()  # don't worry we won't run out of file handles
    test_dataset = CharDataset(text, model.config.block_size)  # one line of poem is roughly 50 characters
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # attempt to run test with existing model
    trainer.test(model, dataloaders=test_loader)
    
    