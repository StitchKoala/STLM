
def train(args, category, rotate_90=False, random_rotate=0):
    # Define optimizer
    nor_optimizer = torch.optim.Adam(twostream.parameters(), betas=(0.5, 0.999), lr=0.0005)
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
    total_loss = 0

    flag = True

    while flag:
        for _, data in enumerate(dataloader):
            twostream.train()
            segmentation_net.train()
            nor_optimizer.zero_grad()
            seg_optimizer.zero_grad()
            img_origin = data["img_origin"].to(device)
            img_pseudo = data["img_aug"].to(device)
            mask = data["mask"].to(device)

            tfeature1, tfeature2 = teacher(img_origin)
            tfeature = [tfeature1, tfeature2]
            pfeature, dfeature = twostream(img_pseudo)

            outputs_T = [
                l2_norm(output_s.detach()) for output_s in tfeature
            ]

            outputs_p = [
                l2_norm(output_t) for output_t in pfeature
            ]
            outputs_d = [
                l2_norm(output_s) for output_s in dfeature
            ]

            output_ano_list = []
            for output_t, output_s in zip(outputs_T, outputs_d):
                a_map = 1 - torch.sum(output_s * output_t, dim=1, keepdim=True)
                output_ano_list.append(a_map)

            output = torch.cat(
                [
                    F.interpolate(
                        -output_t * output_s,
                        size=outputs_d[0].size()[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    for output_t, output_s in zip(outputs_p, outputs_d)
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

            cosine_loss = cosine_similarity_loss1(output_ano_list)

            focal_loss_val = focal_loss(output_segmentation, mask, gamma=args.gamma)
            l1_loss_val = l1_loss(output_segmentation, mask)
            seg_loss = focal_loss_val + l1_loss_val
            total_loss_val = seg_loss + cosine_loss
            total_loss_val.backward()
            nor_optimizer.step()
            seg_optimizer.step()

        global_step += 1

        if global_step % args.eval_per_steps == 0:
            aupro_de_st, auc_de_st, auc_detect_de_st,aupro_seg, auc_seg, auc_detect_seg = evaluate(args, category, mob_sam, segmentation_net, global_step)

        if global_step >= args.steps:
            flag = False
            break