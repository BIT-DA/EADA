import torch
import torch.nn as nn


def bound_max_loss(energy, bound):
    """
    return the loss value of max(0, \mathcal{F}(x) - \Delta )
    """
    energy_minus_bound = energy - bound
    energy_minus_bound = torch.unsqueeze(energy_minus_bound, dim=1)
    zeros = torch.zeros_like(energy_minus_bound)
    for_select = torch.cat((energy_minus_bound, zeros), dim=1)
    selected = torch.max(for_select, dim=1).values

    return selected.mean()


class FreeEnergyAlignmentLoss(nn.Module):
    """
    free energy alignment loss
    """

    def __init__(self, cfg):
        super(FreeEnergyAlignmentLoss, self).__init__()
        assert cfg.TRAINER.ENERGY_BETA > 0, "beta for energy calculate must be larger than 0"
        self.beta = cfg.TRAINER.ENERGY_BETA

        self.type = cfg.TRAINER.ENERGY_ALIGN_TYPE

        if self.type == 'l1':
            self.loss = nn.L1Loss()
        elif self.type == 'mse':
            self.loss = nn.MSELoss()
        elif self.type == 'max':
            self.loss = bound_max_loss

    def forward(self, inputs, bound):
        mul_neg_beta = -1.0 * self.beta * inputs
        log_sum_exp = torch.logsumexp(mul_neg_beta, dim=1)
        free_energies = -1.0 * log_sum_exp / self.beta

        bound = torch.ones_like(free_energies) * bound
        loss = self.loss(free_energies, bound)

        return loss


class NLLLoss(nn.Module):
    """
    NLL loss for energy based model
    """

    def __init__(self, cfg):
        super(NLLLoss, self).__init__()
        assert cfg.TRAINER.ENERGY_BETA > 0, "beta for energy calculate must be larger than 0"
        self.beta = cfg.TRAINER.ENERGY_BETA

    def forward(self, inputs, targets):
        indices = torch.unsqueeze(targets, dim=1)
        energy_c = torch.gather(inputs, dim=1, index=indices)

        all_energy = -1.0 * self.beta * inputs
        free_energy = -1.0 * torch.logsumexp(all_energy, dim=1, keepdim=True) / self.beta

        nLL = energy_c - free_energy

        return nLL.mean()
