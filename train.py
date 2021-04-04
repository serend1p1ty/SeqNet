import argparse
import datetime
import os.path as osp
import time

import torch
import torch.utils.data

from datasets.cuhk_sysu import CUHKSYSU
from datasets.prw import PRW
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch
from models.seqnet import SeqNet
from utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from utils.transforms import build_transforms
from utils.utils import (
    init_distributed_mode,
    mkdir,
    resume_from_ckpt,
    save_on_master,
    set_random_seed,
)


def build_dataset(dataset_name, root, verbose=True):
    assert dataset_name in ["CUHK-SYSU", "PRW"]
    fn = CUHKSYSU if dataset_name == "CUHK-SYSU" else PRW
    train_transforms = build_transforms(is_train=True)
    test_transforms = build_transforms(is_train=False)
    train_set = fn(root, train_transforms, "train")
    gallery_set = fn(root, test_transforms, "gallery")
    query_set = fn(root, test_transforms, "query")
    if verbose:
        train_set.print_statistics()
        gallery_set.print_statistics()
        query_set.print_statistics()
    return train_set, gallery_set, query_set


def collate_fn(batch):
    return tuple(zip(*batch))


def main(args):
    init_distributed_mode(args)
    print(f"Called with args: {args}")

    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    # different processes should be independent of each other,
    # so set different random seeds for them
    set_random_seed(cfg.SEED + args.rank)

    print("Loading data")
    train_set, gallery_set, query_set = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)

    if cfg.INPUT.ASPECT_RATIO_GROUP_FACTOR_TRAIN >= 0:
        group_ids = create_aspect_ratio_groups(
            train_set, k=cfg.INPUT.ASPECT_RATIO_GROUP_FACTOR_TRAIN
        )
        train_batch_sampler = GroupedBatchSampler(
            train_sampler, group_ids, cfg.INPUT.BATCH_SIZE_TRAIN
        )
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, cfg.INPUT.BATCH_SIZE_TRAIN, drop_last=True
        )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_sampler=train_batch_sampler,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    gallery_loader = torch.utils.data.DataLoader(
        gallery_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    query_loader = torch.utils.data.DataLoader(
        query_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print("Creating model")
    model = SeqNet(cfg)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model_without_ddp)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model_without_ddp, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(cfg, model, optimizer, train_loader, device, epoch, tfboard)
        lr_scheduler.step()

        # if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
        #     evaluate_performance(
        #         model,
        #         gallery_loader,
        #         query_loader,
        #         device,
        #         use_gt=cfg.EVAL_USE_GT,
        #         use_cache=cfg.EVAL_USE_CACHE,
        #         use_cbgm=cfg.EVAL_USE_CBGM,
        #     )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    args = parser.parse_args()
    main(args)
