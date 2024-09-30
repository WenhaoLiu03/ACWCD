import torch
import torch.nn.functional as F
import numpy as np


def cam_to_label(cam, img_box=None, ignore_mid=False, cfg=None, cls_label=None):
    b, c, h, w = cam.shape
    cam_value, _pseudo_label = cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= cfg.cam.bkg_score] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= cfg.cam.bkg_score] = 0
    pseudo_label = torch.ones_like(_pseudo_label)

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return cam, pseudo_label

def multi_scale_cam(model, inputs_A, inputs_B, scales):
    # cam_list = []
    b, c, h, w = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        _cam, _ = model(inputs_A_cat, inputs_B_cat, cam_only=True)  # _cam: torch.Size([8, 1, 16, 16])

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs_A = F.interpolate(inputs_A, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_A_cat = torch.cat([_inputs_A, _inputs_A.flip(-1)], dim=0)
                _inputs_B = F.interpolate(inputs_B, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_B_cat = torch.cat([_inputs_B, _inputs_B.flip(-1)], dim=0)

                _cam, _ = model(inputs_A_cat, inputs_B_cat, cam_only=True)  # _cam, _,_

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5
    return cam


def multi_scale_cam_with_change_attn(model, inputs_A, inputs_B, scales):
    cam_list, change_attn = [], []
    b, c, h, w = inputs_A.shape
    with torch.no_grad():
        inputs_A_cat = torch.cat([inputs_A, inputs_A.flip(-1)], dim=0)
        inputs_B_cat = torch.cat([inputs_B, inputs_B.flip(-1)], dim=0)

        _cam, _change_attn =  model(inputs_A_cat, inputs_B_cat, cam_only=True)
        change_attn.append(_change_attn)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]

        for s in scales:
            if s != 1.0:
                _inputs_A = F.interpolate(inputs_A, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_A_cat = torch.cat([_inputs_A, _inputs_A.flip(-1)], dim=0)
                _inputs_B = F.interpolate(inputs_B, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_B_cat = torch.cat([_inputs_B, _inputs_B.flip(-1)], dim=0)

                _cam, _change_attn = model(inputs_A_cat,inputs_B_cat, cam_only=True)
                change_attn.append(_change_attn)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

    max_change_attn = change_attn[np.argmax(scales)]
    return cam, max_change_attn

def propagte_cam_with_change_attn(cams, ct=None, mask=None, cls_labels=None, bkg_score=None):

    b,_,h,w = cams.shape

    bkg = torch.ones(size=(b,1,h,w))*bkg_score
    bkg = bkg.to(cams.device)

    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    cams_with_bkg = torch.cat((bkg, cams), dim=1)

    cams_rw = torch.zeros_like(cams_with_bkg)

    b, c, h, w = cams_with_bkg.shape
    n_pow = 2
    n_log_iter = 0

    if mask is not None:
        for i in range(b):
            ct[i, mask==0] = 0

    ct = ct.detach() ** n_pow
    ct = ct / (torch.sum(ct, dim=1, keepdim=True) + 1e-1) ## avoid nan

    for i in range(n_log_iter):
        ct = torch.matmul(ct, ct)

    for i in range(b):
        _cams = cams_with_bkg[i].reshape(c, -1)
        valid_key = torch.nonzero(cls_labels[i,...])[:,0]
        _cams = _cams[valid_key,...]
        _cams = F.softmax(_cams, dim=0)
        _ct = ct[i]

        _cams_rw = torch.matmul(_cams, _ct)

        cams_rw[i, valid_key,:] = _cams_rw.reshape(-1, cams_rw.shape[2], cams_rw.shape[3])

    return cams_rw

def align_ref_cam(ref_mod=None, images=None, labels=None, cams=None, img_box=None):
    refined_cams = torch.zeros_like(cams)

    cls_label = labels

    for idx, coord in enumerate(img_box):
        _images = images[[idx], :, coord[0]:coord[1], coord[2]:coord[3]]

        _, _, h, w = _images.shape
        _images_ = F.interpolate(_images, size=[h // 2, w // 2], mode="bilinear", align_corners=False)

        valid_key = torch.nonzero(cls_label[idx, ...])[:, 0]
        valid_cams = cams[[idx], :, coord[0]:coord[1], coord[2]:coord[3]][:, valid_key, ...]

        _refined_cams = ref_mod(_images_, valid_cams)
        _refined_cams = F.interpolate(_refined_cams, size=_images.shape[2:], mode="bilinear", align_corners=False)

        refined_cams[idx, valid_key, coord[0]:coord[1], coord[2]:coord[3]] = _refined_cams[0, ...]

    return refined_cams

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label

def align_initial_cam(ref_mod=None, images=None, cams=None, cls_labels=None, cfg=None, img_box=None,
                            down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_c = torch.ones(size=(b, 1, h, w)) * cfg.cam.bkg_score
    bkg_c = bkg_c.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w))
    refined_label = refined_label.to(cams.device)
    refined_label_c = refined_label.clone()

    cams_with_bkg_c = torch.cat((bkg_c, cams), dim=1)
    _cams_with_bkg_c = F.interpolate(cams_with_bkg_c, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_c = _cams_with_bkg_c[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_c = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_c,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_c[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_c[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_c.clone()

    return refined_label

def align_ref_cam_(ref_mod=None, images=None, cams=None):  # 处理所有通道
    b, _, h, w = images.shape

    _images_downsampled = F.interpolate(images, size=[h // 2, w // 2], mode="bilinear", align_corners=False)

    _refined_cams = ref_mod(_images_downsampled, cams)

    refined_cams = F.interpolate(_refined_cams, size=[h, w], mode="bilinear", align_corners=False)

    return refined_cams


