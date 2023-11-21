import argparse
import os
import shutil
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import AUROC
from constant import RESIZE_SHAPE, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from model.metrics import AUPRO
from model.model_utils import l2_norm

warnings.filterwarnings("ignore")

def evaluate(args, category, twostream, segmentation_net):
    twostream.eval()
    segmentation_net.eval()
    with torch.no_grad():
        dataset = MVTecDataset(
            is_train=False,
            mvtec_dir=args.mvtec_path + category + "/test/",
            resize_shape=RESIZE_SHAPE,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
        )

        # w/o FA
        tlm_AUPRO = AUPRO().cuda()
        tlm_AUROC = AUROC().cuda()
        tlm_detect_AUROC = AUROC().cuda()
        # with FA
        fa_AUPRO = AUPRO().cuda()
        fa_AUROC = AUROC().cuda()
        fa_detect_AUROC = AUROC().cuda()

        for _, sample_batched in enumerate(dataloader):
            img = sample_batched["img"].cuda()
            mask = sample_batched["mask"].to(torch.int64).cuda()

            pfeature, dfeature = twostream(img)

            outputs_plain = [
                l2_norm(output_p.detach()) for output_p in pfeature
            ]
            outputs_denoising = [
                l2_norm(output_d.detach()) for output_d in dfeature
            ]

            output = torch.cat(
                [
                    F.interpolate(
                        -output_p * output_d,
                        size=outputs_denoising[0].size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for output_p, output_d in zip(outputs_plain, outputs_denoising)
                ],
                dim=1,
            )

            output_fa = segmentation_net(output)

            output_tlm_list = []
            for output_p, output_d in zip(outputs_plain, outputs_denoising):
                a_map = 1 - torch.sum(output_p * output_d, dim=1, keepdim=True)
                output_tlm_list.append(a_map)

            output_tlm = torch.cat(
                [
                    F.interpolate(
                        output_tlm_instance,
                        size=outputs_denoising[0].size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for output_tlm_instance in output_tlm_list
                ],
                dim=1,
            )  # [N, 3, H, W]
            output_tlm = torch.prod(output_tlm, dim=1, keepdim=True)

            output_fa = F.interpolate(
                output_fa,
                size=mask.size()[2:],
                mode="bilinear",
                align_corners=False,
            )

            output_tlm = F.interpolate(
                output_tlm, size=mask.size()[2:], mode="bilinear", align_corners=False
            )

            mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]
            output_fa_sample, _ = torch.sort(
                output_fa.view(output_fa.size(0), -1),
                dim=1,
                descending=True,
            )
            output_fa_sample = torch.mean(
                output_fa_sample[:, : args.T], dim=1
            )
            output_de_st_sample, _ = torch.sort(
                output_tlm.view(output_tlm.size(0), -1), dim=1, descending=True
            )
            output_de_st_sample = torch.mean(output_de_st_sample[:, : args.T], dim=1)

            tlm_AUPRO.update(output_tlm, mask)
            tlm_AUROC.update(output_tlm.flatten(), mask.flatten())
            tlm_detect_AUROC.update(output_de_st_sample, mask_sample)

            fa_AUPRO.update(output_fa, mask)
            fa_AUROC.update(output_fa.flatten(), mask.flatten())
            fa_detect_AUROC.update(output_fa_sample, mask_sample)

        aupro_de_st, auc_de_st, auc_detect_de_st = (
            tlm_AUPRO.compute(),
            tlm_AUROC.compute(),
            tlm_detect_AUROC.compute(),
        )
        aupro_seg, auc_seg, auc_detect_seg = (
            fa_AUPRO.compute(),
            fa_AUROC.compute(),
            fa_detect_AUROC.compute(),
        )

        tlm_AUPRO.reset()
        tlm_AUROC.reset()
        tlm_detect_AUROC.reset()
        fa_AUPRO.reset()
        fa_AUROC.reset()
        fa_detect_AUROC.reset()
        return aupro_de_st, auc_de_st, auc_detect_de_st,aupro_seg, auc_seg, auc_detect_seg
