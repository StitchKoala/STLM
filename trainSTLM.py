
def train(args, category, rotate_90=False, random_rotate=0):
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_checkpoint = "./weights/mobile_sam.pt"
    model_type = "vit_t"
    sam_mode = "train"
    twostream = Batch_SamE(sam_checkpoint,model_type,sam_mode,device)

    sam_checkpoint = "./weights/mobile_sam.pt"
    model_type = "vit_t"
    sam_mode = "eval"
    fix_teacher = Batch_Sam(sam_checkpoint, model_type, sam_mode, device)

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