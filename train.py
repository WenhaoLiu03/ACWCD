import argparse
import datetime
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import weaklyCD
from utils import evaluate_CD, imutils
from utils.AverageMeter import AverageMeter
from utils.losses import cploss
from utils.camutils_CD import (cam_to_label, multi_scale_cam_with_change_attn, propagte_cam_with_change_attn,
                               align_ref_cam, align_initial_cam, multi_scale_cam, align_ref_cam_)
from utils.optimizer import PolyWarmupAdamW
from models.model_ACWCD import ACWCD
from models.PAR import PAR
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
# DSIFN/CLCD.yaml
parser.add_argument("--config",
                    default='configs/BCD.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--crop_size", default=256, type=int, help="crop_size")
parser.add_argument('--pretrained', default= True, type=bool, help="pretrained")
parser.add_argument('--checkpoint_path', default= False, type=str, help="checkpoint_path" )
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument('--backend', default='nccl')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta * scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)

def get_down_size(ori_shape=(512, 512), stride=16):
    h, w = ori_shape
    _h = h // stride + 1 - ((h % stride) == 0)
    _w = w // stride + 1 - ((w % stride) == 0)
    return _h, _w

def validate(model=None, data_loader=None, cfg=None):
    preds, gts, pseudo_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs_A, inputs_B, labels, cls_label = data

            inputs_A = inputs_A.cuda()
            inputs_B = inputs_B.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()
            b, c, h, w = inputs_A.shape
            inputs_denorm_A = imutils.denormalize_img(inputs_A.clone())
            inputs_denorm_B = imutils.denormalize_img(inputs_B.clone())
            inputs_denorm = torch.absolute(inputs_denorm_A - inputs_denorm_B)

            par = PAR(num_iter=10, dilations=cfg.dataset.dilations)
            par = par.cuda()

            cls, segs, change_attn = model(inputs_A, inputs_B)
            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            _cams = multi_scale_cam(model, inputs_A, inputs_B, cfg.cam.scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)

            H, W = get_down_size(ori_shape=(h, w))
            infer_mask = np.eye(H * W)
            valid_cam_resized = F.interpolate(resized_cam, size=(H, W), mode='bilinear', align_corners=False)

            after_walk_cam = propagte_cam_with_change_attn(valid_cam_resized, ct=change_attn, mask=infer_mask, cls_labels=cls_label,
                                                bkg_score=0.50)
            after_walk_cam = F.interpolate(after_walk_cam, size=labels.shape[1:], mode="bilinear", align_corners=False)

            refined_after_walk_cam = align_ref_cam_(par, inputs_denorm, cams=after_walk_cam)

            final_pseudo_labels = refined_after_walk_cam.argmax(dim=1)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))

            gts += list(labels.cpu().numpy().astype(np.int16))
            pseudo_labels += list(final_pseudo_labels.cpu().numpy().astype(np.int16))

    seg_score = evaluate_CD.scores(gts, preds)
    pseudo_labels_score = evaluate_CD.scores(gts, pseudo_labels)

    model.train()
    return seg_score, pseudo_labels_score, labels


def normalize(after_walk_cam):

    after_walk_cam_max = after_walk_cam.max(dim=1, keepdim=True)[0]

    after_walk_cam_f = after_walk_cam_max.view(after_walk_cam_max.size(0), -1)

    min_vals = after_walk_cam_f.min(dim=1, keepdim=True)[0].unsqueeze(1).unsqueeze(2)
    max_vals = after_walk_cam_f.max(dim=1, keepdim=True)[0].unsqueeze(1).unsqueeze(2)

    normalized_after_walk_cam = (after_walk_cam_max - min_vals) / (max_vals - min_vals)

    return normalized_after_walk_cam

def train(cfg):
    num_workers = 10

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = weaklyCD.ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = weaklyCD.CDDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)
    device = torch.device('cuda')

    acwcd = ACWCD(
        backbone=cfg.backbone.config,
        stride=cfg.backbone.stride,
        num_classes=cfg.dataset.num_classes,
        embedding_dim=256,
        pretrained=args.pretrained,
        pooling=args.pooling,
    )


    param_groups = acwcd.get_param_groups()
    par = PAR(num_iter=10, dilations=cfg.dataset.dilations)

    acwcd.to(device)
    par.to(device)

    infer_size = int((cfg.dataset.crop_size * max(cfg.cam.scales)) // 16)
    mask_infer = np.eye(infer_size * infer_size)

    writer = SummaryWriter(cfg.work_dir.logger_dir)
    print('writer:',writer)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,  ## freeze norm layers
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()

    bkg_cls = torch.ones(size=(cfg.train.batch_size, 1))

    best_F1_seg = 0.0
    best_F1_pseudo_labels = 0.0
    best_iter_seg = 0.0
    best_iter_pseudo_labels = 0.0
    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs_A, inputs_B, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs_A, inputs_A, cls_labels, img_box = next(train_loader_iter)

        inputs_A = inputs_A.to(device, non_blocking=True)
        inputs_B = inputs_B.to(device, non_blocking=True)
        inputs_denorm_A = imutils.denormalize_img(inputs_A.clone())
        inputs_denorm_B = imutils.denormalize_img(inputs_B.clone())
        inputs_denorm = torch.absolute(inputs_denorm_A - inputs_denorm_B)

        cls_labels = cls_labels.to(device, non_blocking=True)

        cls, segs, _ = acwcd(inputs_A, inputs_B, seg_detach=args.seg_detach)

        # Generation of Initial CAM and Change Attention
        cams, change_attn = multi_scale_cam_with_change_attn(acwcd, inputs_A=inputs_A, inputs_B=inputs_B, scales=cfg.cam.scales)
        valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True, cfg=cfg)
        valid_cam_resized = F.interpolate(valid_cam, size=(infer_size, infer_size), mode='bilinear', align_corners=False)

        # Random Walk Propagation
        # AR Module Propagation under CP Constraints
        # Final Pseudo-Label Generation
        after_walk_cam = propagte_cam_with_change_attn(valid_cam_resized, ct=change_attn.detach().clone(), mask=mask_infer,
                                                       cls_labels=cls_labels, bkg_score=cfg.cam.bkg_score)
        after_walk_cam = F.interpolate(after_walk_cam, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        bkg_cls = bkg_cls.to(cams.device)
        _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)
        refined_after_walk_cam = align_ref_cam(par, inputs_denorm, cams=after_walk_cam, labels=_cls_labels, img_box=img_box)

        final_pseudo_labels = refined_after_walk_cam.argmax(dim=1)

        # Visualization of Propagated Threshold CAM and Initial CAM
        refined_after_walk_cam1 = refined_after_walk_cam[:, 0, :, :].unsqueeze(1)
        refined_after_walk_cam2 = refined_after_walk_cam[:, 1, :, :].unsqueeze(1)

        refined_after_walk_cam1 = normalize(refined_after_walk_cam1)
        refined_after_walk_cam2 = normalize(refined_after_walk_cam2)

        initial_pseudo_labels = align_initial_cam(par, inputs_denorm, cams=cams, cls_labels=cls_labels, cfg=cfg, img_box=img_box)

        segs = F.interpolate(segs, size=initial_pseudo_labels.shape[1:], mode='bilinear', align_corners=False)

        if n_iter <= 8000:
            final_pseudo_labels = initial_pseudo_labels


        seg_loss = F.cross_entropy(segs, final_pseudo_labels.type(torch.long), ignore_index=255)

        # Image-level classification loss
        lp_loss = F.binary_cross_entropy_with_logits(cls, cls_labels)

        segs_end = torch.argmax(segs, dim=1)

        # Change prior loss(CP_loss)
        cp_loss1 = cploss(cls_labels, final_pseudo_labels, alpha1=0.9, alpha2=0.9) # Calculation of Pseudo-Labels
        cp_loss2 = cploss(cls_labels, segs_end, alpha1=0.9, alpha2=0.9) # Calculation of Segmentation Results

        if n_iter <= cfg.train.cam_iters:
            loss = 1.0 * lp_loss + 0.0 * cp_loss1 + 0.0 * cp_loss2 + 0.0 * seg_loss
        else:
            loss = 1.0 * lp_loss + 1.0 * cp_loss1 + 1.0 * cp_loss2 + 0.1 * seg_loss

        avg_meter.add({'lp_loss': lp_loss.item(), 'cp_loss1': cp_loss1.item(), 'seg_loss': seg_loss.item(), 'cp_loss2': cp_loss2.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)

            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs, dim=1, ).cpu().numpy().astype(np.int16)
            pseudo_label = pseudo_label.cpu().numpy().astype(np.int16)
            final_gts = final_pseudo_labels.cpu().numpy().astype(np.int16)

            logging.info(
                "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; lp_loss: %.4f, cp_loss1: %.4f, seg_loss: %.4f, cp_loss2: %.4f" % (
                    n_iter + 1, delta, eta, cur_lr, avg_meter.pop('lp_loss'), avg_meter.pop('cp_loss1'), avg_meter.pop('seg_loss'), avg_meter.pop('cp_loss2')))

            grid_imgs_A, grid_cam_A = imutils.tensorboard_image(imgs=inputs_A.clone(), cam=valid_cam)
            grid_imgs_B, grid_cam_B = imutils.tensorboard_image(imgs=inputs_B.clone(), cam=valid_cam)

            _, grid_ref_after_walk_cam1 = imutils.tensorboard_image(imgs=inputs_B.clone(), cam=refined_after_walk_cam1)
            _, grid_ref_after_walk_cam2 = imutils.tensorboard_image(imgs=inputs_B.clone(), cam=refined_after_walk_cam2)

            grid_labels = imutils.tensorboard_label(labels=pseudo_label)
            grid_preds = imutils.tensorboard_label(labels=preds)
            grid_final_gt = imutils.tensorboard_label(labels=final_gts)

            writer.add_image("train/images_A"+str(img_name), grid_imgs_A, global_step=n_iter)
            writer.add_image("train/images_B"+str(img_name), grid_imgs_B, global_step=n_iter)
            writer.add_image("cam/valid_cams_A", grid_cam_A, global_step=n_iter)
            writer.add_image("cam/valid_cams_B", grid_cam_B, global_step=n_iter)
            writer.add_image("train/preds_cam", grid_labels, global_step=n_iter)
            writer.add_image("train/preds", grid_preds, global_step=n_iter)
            writer.add_image("train/final_gts", grid_final_gt, global_step=n_iter)
            writer.add_image("cam/refined_after_walk_cam_back", grid_ref_after_walk_cam1, global_step=n_iter)
            writer.add_image("cam/refined_after_walk_cam_fore", grid_ref_after_walk_cam2, global_step=n_iter)

            writer.add_scalars('train/loss', {"lp_loss": lp_loss.item(), "cp_loss1": cp_loss1.item(), "seg_loss": seg_loss.item(),"cp_loss2": cp_loss2.item()},
                               global_step=n_iter)

        if (n_iter + 1) % cfg.train.eval_iters == 0:

            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "acwcd_iter_%d.pth" % (n_iter + 1))
            logging.info('CD Validating...')
            torch.save(acwcd.state_dict(), ckpt_name)
            seg_score, pseudo_labels_score, _ = validate(model=acwcd, data_loader=val_loader, cfg=cfg)  # _ 为 labels

            if seg_score['f1'][1] > best_F1_seg:
                best_F1_seg = seg_score['f1'][1]
                best_iter_seg = n_iter + 1

            if pseudo_labels_score['f1'][1] > best_F1_pseudo_labels:
                best_F1_pseudo_labels = pseudo_labels_score['f1'][1]
                best_iter_pseudo_labels = n_iter + 1

            logging.info("pseudo_labels_score: %s, \n[best_iter]: %d", pseudo_labels_score, best_iter_pseudo_labels)
            logging.info("seg_score: %s, \n[best_iter]: %d", seg_score, best_iter_seg)

    return True

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)


    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp + '.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)

