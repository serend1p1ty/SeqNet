import torch
import torch.nn.functional as F
from torch import autograd, nn

from utils.utils import all_gather


class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum = ctx.saved_tensors

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, lut, cq, torch.tensor(header), torch.tensor(momentum))


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0

    def forward(self, inputs, roi_labels):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_labels)
        labels = targets - 1  # background label = -1

        inds = labels >= 0
        labels = labels[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        # Gather the batch data in all GPUs to calculate OIM loss
        # Otherwise, the lut and cq of each GPU will be different.
        device = inputs.device
        inputs = all_gather(inputs)
        inputs = torch.cat([input.to(device) for input in inputs], dim=0)
        labels = all_gather(labels)
        labels = torch.cat([label.to(device) for label in labels], dim=0)

        projected = oim(inputs, labels, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        projected *= self.oim_scalar

        self.header_cq = (
            self.header_cq + (labels >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        loss_oim = F.cross_entropy(projected, labels, ignore_index=5554)
        return loss_oim
