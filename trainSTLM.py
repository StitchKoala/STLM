import argparse
import os
import shutil
import warnings
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from constant import RESIZE_SHAPE, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from evalSTLM import evaluate
from model.losses import cosine_similarity_loss, focal_loss, l1_loss
from mob_sam import Batch_Sam,Batch_SamE,SegmentationNet
from model.model_utils import ASPP, BasicBlock, l2_norm, make_layer

warnings.filterwarnings("ignore")

def train(args, category, rotate_90=False, random_rotate=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize
    sam_checkpoint = "./weights/mobile_sam.pt"
    model_type = "vit_t"
    sam_mode = "train"
    twostream = Batch_SamE(sam_checkpoint,model_type,sam_mode,device)

    sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam_mode = "eval"
    fix_teacher = Batch_Sam(sam_checkpoint,model_type,sam_mode,device)

    segmentation_net = SegmentationNet(512).cuda()

    # Define optimizer
    tlm_optimizer = torch.optim.Adam(twostream.parameters(), betas=(0.5, 0.999), lr=0.0005)
    seg_optimizer = torch.optim.SGD(
        [
            {"params": segmentation_net.res.parameters(), "lr": args.lr_res},
            {"params": segmentation_net.head.parameters(), "lr": args.lr_seghead},
        ],
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    dataset = MVTecDataset(
        is_train=True,
        mvtec_dir=args.mvtec_path + category + "/train/good/",
        resize_shape=RESIZE_SHAPE,
        dtd_dir=args.dtd_path,
        rotate_90=rotate_90,
        random_rotate=random_rotate,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    global_step = 0
    flag = True

    while flag:
        for _, data in enumerate(dataloader):
            twostream.train()
            segmentation_net.train()
            tlm_optimizer.zero_grad()
            seg_optimizer.zero_grad()
            img_origin = data["img_origin"].to(device)
            img_pseudo = data["img_aug"].to(device)
            mask = data["mask"].to(device)

            pfeature1, pfeature2 = fix_teacher(img_pseudo)
            dfeature1, dfeature2 = fix_teacher(img_origin)
            Tpfeature = [pfeature1, pfeature2]
            Tdfeature = [dfeature1, dfeature2]
            Pfeature, Dfeature = twostream(img_pseudo)

            outputs_Tplain = [
                l2_norm(output_p.detach()) for output_p in Tpfeature
            ]
            outputs_Tdenoising = [
                l2_norm(output_d.detach()) for output_d in Tdfeature
            ]

            outputs_Splain = [
                l2_norm(output_p) for output_p in Pfeature
            ]
            outputs_Sdenoising = [
                l2_norm(output_d) for output_d in Dfeature
            ]

            output_pain_list = []
            for output_t, output_s in zip(outputs_Tplain, outputs_Splain):
                a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
                output_pain_list.append(a_map)

            output_denoising_list = []
            for output_t, output_s in zip(outputs_Tdenoising, outputs_Sdenoising):
                a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
                output_denoising_list.append(a_map)

            output = torch.cat(
                [
                    F.interpolate(
                        -output_p * output_d,
                        size=outputs_Sdenoising[0].size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for output_p, output_d in zip(outputs_Splain, outputs_Sdenoising)
                ],
                dim=1,
            )

            output_segmentation = segmentation_net(output)

            mask = F.interpolate(
                mask,
                size=output_segmentation.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            mask = torch.where(
                mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask)
            )

            cosine_loss = cosine_similarity_loss(output_pain_list) + cosine_similarity_loss(output_denoising_list)

            focal_loss = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss = l1_loss(output_segmentation, mask)
            seg_loss = focal_loss + l1_loss

            total_loss_val = seg_loss + cosine_loss
            total_loss_val.backward()
            tlm_optimizer.step()
            seg_optimizer.step()

        global_step += 1

        if global_step % args.eval_per_steps == 0:
            aupro_tlm, auc_tlm, auc_detect_tlm,aupro_fa, auc_fa, auc_detect_fa = evaluate(args, category, twostream, segmentation_net)

        if global_step >= args.steps:
            flag = False
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="./datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="./datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--run_name_head", type=str, default="STLM")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=16)
    # parser.add_argument("--lr_de_st", type=float, default=0.4)
    parser.add_argument("--lr_res", type=float, default=0.1)
    parser.add_argument("--lr_seghead", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--de_st_steps", type=int, default=1000
    )  # steps of training the denoising student model
    parser.add_argument("--eval_per_steps", type=int, default=5)
    parser.add_argument("--log_per_steps", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=4)  # for focal loss
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument(
        "--custom_training_category", action="store_true", default=False
    )
    parser.add_argument("--no_rotation_category", nargs="*", type=str, default=list())
    parser.add_argument(
        "--slight_rotation_category", nargs="*", type=str, default=list()
    )
    parser.add_argument("--rotation_category", nargs="*", type=str, default=list())

    args = parser.parse_args()

    if args.custom_training_category:
        no_rotation_category = args.no_rotation_category
        slight_rotation_category = args.slight_rotation_category
        rotation_category = args.rotation_category
        # check
        for category in (
            no_rotation_category + slight_rotation_category + rotation_category
        ):
            assert category in ALL_CATEGORY
    else:
        no_rotation_category = [
            "capsule",
            "metal_nut",
            "pill",
            "toothbrush",
            "transistor",
            "screw",
            "grid",
        ]
        slight_rotation_category = [
            "wood",
            "zipper",
            "cable",
            "transistor",
            "screw",
            "grid",
        ]
        rotation_category = [
            "bottle",
            "grid",
            "hazelnut",
            "leather",
            "tile",
            "carpet",
            "screw",
            "transistor",
        ]

    with torch.cuda.device(args.gpu_id):
        for obj in no_rotation_category:
            print(obj)
            args.run_name_head = f"{args.run_name_head}_no_rotation"
            train(args, obj)

        for obj in slight_rotation_category:
            print(obj)
            args.run_name_head = f"{args.run_name_head}_slight_rotation"
            train(args, obj, rotate_90=False, random_rotate=5)

        for obj in rotation_category:
            print(obj)
            args.run_name_head = f"{args.run_name_head}_rotation"
            train(args, obj, rotate_90=True, random_rotate=5)
