import os
import sys
import logging

import numpy as np
import torch
import torch.multiprocessing
import torch.distributed as dist

from data_provider.data_provider import data_provider
from distributed.distributed import is_main_process, Disributed
from utils.loggers import WandbLogger, PrintLogger, CompositeLogger
from utils.utils import create_model_name_and_dir, print_model_params, log_config_and_tags
from utils.utils_args import parse_args_uncond

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
torch.multiprocessing.set_sharing_strategy("file_system")

from data_provider.combined_datasets import dataset_list
from importlib import import_module


def main(args):
    args.finetune = not args.pretrain
    args.train_on_datasets = [dataset for dataset in dataset_list if dataset in args.train_on_datasets]

    create_model_name_and_dir(args)

    with (
        CompositeLogger([WandbLogger(project=args.wandb_project, name=args.run_id), PrintLogger()])
        if args.wandb and is_main_process()
        else CompositeLogger([PrintLogger()])
    ) as logger:
        if args.finetune:
            args.tags.append("finetune")
        else:
            args.tags.append("pretrain")

        log_config_and_tags(args, logger, None, len(args.train_on_datasets) > 1)

        dataset_loader, samplers, trainsets, metadatas = data_provider(args)
        args.n_classes = dataset_loader.num_datasets
        if len(args.datasets) > 1:
            logging.info(
                "all datasets are ready - Total number of sequences: %s",
                sum(len(trainset) for trainset in trainsets.values()),
            )
        else:
            logging.info("%s dataset is ready.", args.datasets[0]["name"])

        handler = import_module(args.handler).Handler(args=args, rank=args.device)
        print_model_params(logger, handler.model)

        best_score = getattr(handler, "best_score", float("inf"))
        start_epoch = getattr(handler, "resume_epoch", 0) + 1
        if start_epoch > 1:
            logging.info("Resuming training from epoch %s", start_epoch)

        for epoch in range(start_epoch, args.epochs):
            handler.model.train()
            handler.epoch = epoch
            handler.best_score = best_score
            logger.log("train/epoch", epoch, step=epoch)

            if args.ddp:
                dist.barrier()
                for sampler in samplers.values():
                    sampler.set_epoch(epoch)
                dist.barrier()

            handler.train_iter(dataset_loader, logger)

            if epoch % args.logging_iter == 0 and is_main_process():
                handler.best_score = best_score
                handler.save_model(args.log_dir)

            if args.ddp:
                dist.barrier()

        logging.info("%s is complete", "Finetune" if args.finetune else "Pretrain")


if __name__ == "__main__":
    args = parse_args_uncond()
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if args.ddp:
        Disributed(main).run(args)
    else:
        args.gpu_num = 1
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        main(args)
