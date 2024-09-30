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
parser.add_argument("--model_path", default=r"E:\1\2024-09-14-15-28/acwcd_iter_20000.pth", type=str, help="model_path")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--bkg_score", default=0.50, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=256, type=int, help="resize the long side (256 or 512)")

def test(model, dataset, test_scales=1.0):
    _preds, _gts = [], []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.test.batch_size, shuffle=False, num_workers=2,
                                              pin_memory=False)

    with torch.no_grad(), torch.cuda.device(0):
        model.cuda(0)
        for idx, data in tqdm(enumerate(data_loader)):

            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.cuda()
            inputs_B = inputs_B.cuda()
            b, c, h, w = inputs_A.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            _, _, h, w = inputs_A.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_A = F.interpolate(inputs_A, size=(_h, _w), mode='bilinear', align_corners=False)

            _, _, h, w = inputs_B.shape
            ratio = args.resize_long / max(h, w)
            _h, _w = int(h * ratio), int(w * ratio)
            inputs_B = F.interpolate(inputs_B, size=(_h, _w), mode='bilinear', align_corners=False)

            #######

            segs_list = []
            inputs_cat_A = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
            inputs_cat_B = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)
            _, segs_cat, _ = model(inputs_cat_A, inputs_cat_B)
            segs = segs_cat[0].unsqueeze(0)

            _segs = (segs_cat[0, ...] + segs_cat[1, ...].flip(-1)) / 2
            segs_list.append(_segs)

            _, _, h, w = segs_cat.shape

            for s in test_scales:
                if s != 1.0:
                    _inputsA = F.interpolate(inputs_A, scale_factor=s, mode='bilinear', align_corners=False)
                    _inputsB = F.interpolate(inputs_B, scale_factor=s, mode='bilinear', align_corners=False)
                    inputs_catA = torch.cat([_inputsA, _inputsA.flip(-1)], dim=0)
                    inputs_catB = torch.cat([_inputsB, _inputsB.flip(-1)], dim=0)

                    _, segs_cat, _ = model(inputs_catA, inputs_catB)

                    _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                    _segs = (_segs_cat[0, ...] + _segs_cat[1, ...].flip(-1)) / 2
                    segs_list.append(_segs)

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
            seg_preds = torch.argmax(resized_segs, dim=1)

            _preds += list(seg_preds.cpu().numpy().astype(np.int16))
            _gts += list(labels.cpu().numpy().astype(np.int16))

            _preds_path = args.save_dir + '/seg-prediction/' + name[0] + '.png'

            _preds_img = Image.fromarray((seg_preds.squeeze().cpu().numpy() * 255).astype(np.uint8))

            _preds_img.save(_preds_path)

            ### FN and FP color ###
            preds = seg_preds.squeeze().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()

            # Create RGB image from labels
            label_rgb1 = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
            label_rgb1[labels == 0] = [0, 0, 0]  # Background (black)
            label_rgb1[labels == 1] = [255, 255, 255]  # Foreground (white)

            # Mark FN pixels as blue
            fn_pixels1 = np.logical_and(preds == 0, labels == 1)  # False Negatives
            label_rgb1[fn_pixels1] = [0, 0, 255]  # Blue

            # Mark FP pixels as red
            fp_pixels1 = np.logical_and(preds == 1, labels == 0)
            label_rgb1[fp_pixels1] = [255, 0, 0]  # Red

            # Save the labeled image
            label_with_fn_fp_path_preds = args.save_dir + '/seg-prediction-color/' + name[0] + '.png'
            label_with_fn_fp_img_preds = Image.fromarray(label_rgb1)
            label_with_fn_fp_img_preds.save(label_with_fn_fp_path_preds)

        return inputs_A, inputs_B, _gts, _preds


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

    inputs_A, inputs_B, _gts, _preds = test(model=acwcd, dataset=test_dataset, test_scales=[1, 0.5, 0.75])
    torch.cuda.empty_cache()

    preds_score = evaluate_CD.scores(_gts, _preds)

    print(" preds score:")
    print( preds_score)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    cfg.cam.bkg_score = args.bkg_score
    print(cfg)
    print(args)

    args.save_dir = os.path.join(args.save_dir, args.eval_set)

    os.makedirs(args.save_dir + "/seg-prediction", exist_ok=True)
    os.makedirs(args.save_dir + "/seg-prediction-color", exist_ok=True)


    main(cfg=cfg)
