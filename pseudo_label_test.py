import argparse
import os
from collections import OrderedDict
from PIL import Image
from utils.camutils_CD import cam_to_label, multi_scale_cam
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from utils import evaluate_CD, imutils
from tqdm import tqdm
from datasets import weaklyCD
from models.PAR import PAR
from utils import evaluate_CD
from models.model_ACWCD import ACWCD
from utils.camutils_CD import (cam_to_label, multi_scale_cam_with_change_attn, propagte_cam_with_change_attn,
                               align_ref_cam, align_initial_cam, multi_scale_cam, align_ref_cam_)
parser = argparse.ArgumentParser()
# DSIFN/CLCD.yaml
parser.add_argument("--config",default='configs/BCD.yaml',type=str,
                    help="config")
parser.add_argument("--save_dir", default="./results/BCD", type=str, help="save_dir")
parser.add_argument("--eval_set", default="test", type=str, help="eval_set")
# model_path
parser.add_argument("--model_path", default=r"E:\1\2024-09-18-16-44/acwcd_iter_20000.pth", type=str, help="model_path")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--bkg_score", default=0.50, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=256, type=int, help="resize the long side (256 or 512)")

def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w

def test(model, dataset, test_scales=1.0):
    preds, gts, pseudo_labels = [], [], []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=False)

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda(0)
        for idx, data in tqdm(enumerate(data_loader)):
            ### 注意此处cls_label ###
            name, inputs_A, inputs_B, labels, cls_label = data
            inputs_A = inputs_A.cuda()
            inputs_B = inputs_B.cuda()
            inputs_denorm_A = imutils.denormalize_img(inputs_A.clone())
            inputs_denorm_B = imutils.denormalize_img(inputs_B.clone())
            inputs_denorm = torch.absolute(inputs_denorm_A - inputs_denorm_B)

            b, c, h, w = inputs_A.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            cls, segs, change_attn = model(inputs_A, inputs_B)

            _, _, h, w = inputs_A.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_A = F.interpolate(inputs_A, size=(_h, _w), mode='bilinear', align_corners=False)

            _, _, h, w = inputs_B.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_B = F.interpolate(inputs_B, size=(_h, _w), mode='bilinear', align_corners=False)

            #######

            _cams = multi_scale_cam(model, inputs_A, inputs_B, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)


            H, W = get_down_size(ori_shape=(h, w))
            infer_mask = np.eye(H * W)

            valid_cam_resized = F.interpolate(resized_cam, size=(H, W), mode='bilinear', align_corners=False)
            after_walk_cam = propagte_cam_with_change_attn(valid_cam_resized, ct=change_attn, mask=infer_mask, cls_labels=cls_label,
                                                bkg_score=cfg.cam.bkg_score)
            after_walk_cam = F.interpolate(after_walk_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)

            par = PAR(num_iter=10, dilations=cfg.dataset.dilations)
            par = par.cuda()
            refined_after_walk_cam = align_ref_cam_(par, inputs_denorm, cams=after_walk_cam)
            final_pseudo_labels = refined_after_walk_cam.argmax(dim=1)

            gts += list(labels.cpu().numpy().astype(np.int16))
            pseudo_labels += list(final_pseudo_labels.cpu().numpy().astype(np.int16))

            cam_path = args.save_dir + '/pseudo-label-prediction/' + name[0] + '.png'
            cam_img = Image.fromarray((final_pseudo_labels.squeeze().cpu().numpy() * 255).astype(np.uint8))
            cam_img.save(cam_path)

            ### FN and FP color ###
            cam = final_pseudo_labels.squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            # Create RGB image from labels
            label_rgb = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
            label_rgb[labels == 0] = [0, 0, 0]  # Background (black)
            label_rgb[labels == 1] = [255, 255, 255]  # Foreground (white)

            # Mark FN pixels as blue
            fn_pixels = np.logical_and(cam == 0, labels == 1)  # False Negatives
            label_rgb[fn_pixels] = [0, 0, 255]  # Blue

            # Mark FP pixels as red
            fp_pixels = np.logical_and(cam == 1, labels == 0)  # False Positives
            label_rgb[fp_pixels] = [255, 0, 0]  # Red

            # Save the labeled image
            label_with_fn_fp_path = args.save_dir + '/pseudo-label-prediction_color/' + name[0] + '.png'
            label_with_fn_fp_img = Image.fromarray(label_rgb)
            label_with_fn_fp_img.save(label_with_fn_fp_path)

        return inputs_A, inputs_B, gts, pseudo_labels


def main(cfg):
    test_dataset = weaklyCD.CDDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='test',
        aug=False,
        num_classes=cfg.dataset.num_classes,
    )

    acwcd = ACWCD(
        backbone=cfg.backbone.config,
        stride=cfg.backbone.stride,
        num_classes=cfg.dataset.num_classes,
        embedding_dim=256,
        pretrained=True,
        pooling=args.pooling,
    )


    trained_state_dict = torch.load(args.model_path, map_location="cpu")

    new_state_dict = OrderedDict()
    for k, v in trained_state_dict.items():
        if 'diff.0.bias' in k:
            k = k.replace('diff.0.bias', 'diff.bias')
        if 'diff.0.weight' in k:
            k = k.replace('diff.0.weight', 'diff.weight')
        new_state_dict[k] = v

    acwcd.load_state_dict(state_dict=new_state_dict, strict=True)  # True
    acwcd.eval()

    inputs_A, inputs_B, gts, pseudo_labels= test(model=acwcd, dataset=test_dataset, test_scales=[1.0])
    torch.cuda.empty_cache()

    pseudo_labels_score = evaluate_CD.scores(gts, pseudo_labels)

    print("pseudo labels score:")
    print(pseudo_labels_score)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    cfg.cam.bkg_score = args.bkg_score
    print(cfg)
    print(args)

    args.save_dir = os.path.join(args.save_dir, args.eval_set)

    os.makedirs(args.save_dir + "/pseudo-label-prediction", exist_ok=True)
    os.makedirs(args.save_dir + "/pseudo-label-prediction_color", exist_ok=True)
    main(cfg=cfg)

