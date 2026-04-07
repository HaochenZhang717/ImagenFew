import os, sys
import torch
import numpy as np
import torch.multiprocessing
import logging
from metrics import evaluate_model_uncond
from utils.loggers import NeptuneLogger, WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from data_provider.data_provider import data_provider
from utils.utils_args import parse_args_uncond

from distributed.distributed import is_main_process, Disributed
import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.multiprocessing.set_sharing_strategy('file_system')
from data_provider.combined_datasets import dataset_list
from importlib import import_module

def main(args):
    # Set up basic attributes
    args.finetune = not args.pretrain
    args.train_on_datasets = [dataset for dataset in dataset_list if dataset in args.train_on_datasets]
    
    # Model name and directory
    name = create_model_name_and_dir(args)

    # set-up wandb logger. switch to your desired logger
    with CompositeLogger([WandbLogger(project=args.wandb_project, name=args.run_id), PrintLogger()]) if args.wandb and is_main_process() \
            else CompositeLogger([PrintLogger()]) as logger:
        
        if args.finetune:
            args.tags.append('finetune')
        else:
            args.tags.append('pretrain')
        
        # log config and tags
        log_config_and_tags(args, logger, name, len(args.train_on_datasets) > 1)

        # Setup Data
        dataset_loader, samplers, trainsets, metadatas = data_provider(args)
        args.n_classes = dataset_loader.num_datasets
        if len(args.datasets) > 1:
            logging.info(f'all datasets are ready - Total number of sequences: {sum([len(trainset) for keya, trainset in trainsets.items()])}')
        else:
            logging.info(args.datasets[0]['name'] + ' dataset is ready.')

        # Setup handler
        handler = import_module(args.handler).Handler(args=args, rank=args.device)

        # print model parameters
        print_model_params(logger, handler.model)

        # --- train model ---
        start_epoch = getattr(handler, 'resume_epoch', 0) + 1
        if start_epoch > 1:
            logging.info(f"Resuming training from epoch {start_epoch}")
        for epoch in range(start_epoch, args.epochs):
            handler.model.train()
            handler.epoch = epoch
            # logger.log('train/epoch', epoch, step=epoch)

            if args.ddp:
                dist.barrier()
                for key, sampler in samplers.items():
                    sampler.set_epoch(epoch)
                dist.barrier()

            # --- train loop ---
            handler.train_iter(dataset_loader, logger)

            if epoch % args.logging_iter == 0:
                handler.best_score = getattr(handler, 'best_score', float('inf'))
                handler.save_model(args.log_dir)

            if args.ddp:
                dist.barrier()

        logging.info(f"{'Finetune' if args.finetune else 'Pretrain'} is complete")

if __name__ == '__main__':
    args = parse_args_uncond()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.ddp:
        Disributed(main).run(args)
    else:
        args.gpu_num = 1
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        main(args)
