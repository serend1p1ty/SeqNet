import torch
from PIL import Image

from utils.utils import create_small_table


class BaseDataset:
    """
    Base class of person search dataset.
    """

    def __init__(self, root, transforms, split):
        self.root = root
        self.transforms = transforms
        self.split = split
        assert self.split in ("train", "gallery", "query")
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """
        For each image, load its annotation that is a dictionary with the following keys:
            img_name (str): image name
            img_path (str): image path
            boxes (np.array[N, 4]): ground-truth boxes in (x1, y1, x2, y2) format
            pids (np.array[N]): person IDs corresponding to these boxes
            cam_id (int): camera ID (only for PRW dataset)
        """
        raise NotImplementedError

    def __getitem__(self, index):
        anno = self.annotations[index]
        img = Image.open(anno["img_path"]).convert("RGB")
        boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(anno["pids"], dtype=torch.int64)
        target = {"img_name": anno["img_name"], "boxes": boxes, "labels": labels}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.annotations)

    def print_statistics(self):
        """
        Print dataset statistics.
        """
        num_imgs = len(self.annotations)
        num_boxes = 0
        pid_set = set()
        for anno in self.annotations:
            num_boxes += anno["boxes"].shape[0]
            for pid in anno["pids"]:
                pid_set.add(pid)
        statistics = {
            "dataset": self.name,
            "split": self.split,
            "num_images": num_imgs,
            "num_boxes": num_boxes,
        }
        if len(pid_set) != 1:
            pid_list = sorted(list(pid_set))
            unlabeled_pid = pid_list[-1]
            pid_list = pid_list[:-1]  # remove unlabeled pid
            num_pids, min_pid, max_pid = len(pid_list), min(pid_list), max(pid_list)
            statistics.update(
                {
                    "num_labeled_pids": num_pids,
                    "min_labeled_pid": int(min_pid),
                    "max_labeled_pid": int(max_pid),
                    "unlabeled_pid": int(unlabeled_pid),
                }
            )
        print(f"=> {self.name}-{self.split} loaded:\n" + create_small_table(statistics))
