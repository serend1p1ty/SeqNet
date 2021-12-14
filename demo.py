import argparse
from glob import glob

import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import functional as F

from defaults import get_default_cfg
from models.seqnet import SeqNet
from utils.utils import resume_from_ckpt


def visualize_result(img_path, detections, similarities):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(plt.imread(img_path))
    plt.axis("off")
    for detection, sim in zip(detections, similarities):
        x1, y1, x2, y2 = detection
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#4CAF50", linewidth=3.5
            )
        )
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1)
        )
        ax.text(
            x1 + 5,
            y1 - 18,
            "{:.2f}".format(sim),
            bbox=dict(facecolor="#4CAF50", linewidth=0),
            fontsize=20,
            color="white",
        )
    plt.tight_layout()
    fig.savefig(img_path.replace("gallery", "result"))
    plt.show()
    plt.close(fig)


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)

    print("Creating model")
    model = SeqNet(cfg)
    model.to(device)
    model.eval()

    resume_from_ckpt(args.ckpt, model)

    query_img = [F.to_tensor(Image.open("demo_imgs/query.jpg").convert("RGB")).to(device)]
    query_target = [{"boxes": torch.tensor([[0, 0, 466, 943]]).to(device)}]
    query_feat = model(query_img, query_target)[0]

    gallery_img_paths = sorted(glob("demo_imgs/gallery-*.jpg"))
    for gallery_img_path in gallery_img_paths:
        print(f"Processing {gallery_img_path}")
        gallery_img = [F.to_tensor(Image.open(gallery_img_path).convert("RGB")).to(device)]
        gallery_output = model(gallery_img)[0]
        detections = gallery_output["boxes"]
        gallery_feats = gallery_output["embeddings"]

        # Compute pairwise cosine similarities,
        # which equals to inner-products, as features are already L2-normed
        similarities = gallery_feats.mm(query_feat.view(-1, 1)).squeeze()

        visualize_result(gallery_img_path, detections.cpu().numpy(), similarities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    with torch.no_grad():
        main(args)
