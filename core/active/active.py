import random
import math
import numpy as np
import torch


def RAND_active(tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality):
    length = len(tgt_unlabeled_ds.samples)
    index = random.sample(range(length), round(totality * active_ratio))

    active_samples = tgt_unlabeled_ds.samples[index]

    tgt_selected_ds.add_item(active_samples)
    tgt_unlabeled_ds.remove_item(index)

    return active_samples


def EADA_active(tgt_unlabeled_loader_full, tgt_unlabeled_ds, tgt_selected_ds, active_ratio, totality, model, cfg):
    model.eval()
    first_stat = list()
    with torch.no_grad():
        for _, data in enumerate(tgt_unlabeled_loader_full):
            tgt_img, tgt_lbl = data['img'], data['label']
            tgt_path, tgt_index = data['path'], data['index']
            tgt_img, tgt_lbl = tgt_img.cuda(), tgt_lbl.cuda()

            tgt_out = model(tgt_img, return_feat=False)

            # MvSM of each sample
            # minimal energy - second minimal energy
            min2 = torch.topk(tgt_out, k=2, dim=1, largest=False).values
            mvsm_uncertainty = min2[:, 0] - min2[:, 1]

            # free energy of each sample
            output_div_t = -1.0 * tgt_out / cfg.TRAINER.ENERGY_BETA
            output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
            free_energy = -1.0 * cfg.TRAINER.ENERGY_BETA * output_logsumexp

            for i in range(len(free_energy)):
                first_stat.append([tgt_path[i], tgt_lbl[i].item(), tgt_index[i].item(),
                                   mvsm_uncertainty[i].item(), free_energy[i].item()])

    first_sample_ratio = cfg.TRAINER.FIRST_SAMPLE_RATIO
    first_sample_num = math.ceil(totality * first_sample_ratio)
    second_sample_ratio = active_ratio / cfg.TRAINER.FIRST_SAMPLE_RATIO
    second_sample_num = math.ceil(first_sample_num * second_sample_ratio)

    # the first sample using \mathca{F}, higher value, higher consideration
    first_stat = sorted(first_stat, key=lambda x: x[4], reverse=True)
    second_stat = first_stat[:first_sample_num]

    # the second sample using \mathca{U}, higher value, higher consideration
    second_stat = sorted(second_stat, key=lambda x: x[3], reverse=True)
    second_stat = np.array(second_stat)

    active_samples = second_stat[:second_sample_num, 0:2, ...]
    candidate_ds_index = second_stat[:second_sample_num, 2, ...]
    candidate_ds_index = np.array(candidate_ds_index, dtype=np.int)

    tgt_selected_ds.add_item(active_samples)
    tgt_unlabeled_ds.remove_item(candidate_ds_index)

    return active_samples
