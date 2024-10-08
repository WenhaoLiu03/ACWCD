import torch
import torch.nn.functional as F

def Luc(Predn_final, yn_cls, alpha1=1.0):

    change_pixel_count = (Predn_final.gt(0)).sum(dim=(1, 2))

    total_pixels = 256 * 256
    change_pixel_ratio = change_pixel_count.float() / total_pixels

    loss_coefficient = change_pixel_ratio + 1
    luc = alpha1 * loss_coefficient * (change_pixel_count > 0).float() * (1 - yn_cls.squeeze(1))


    return luc

def Lc(Predn_final, yn_cls, alpha2=1.0):

    change_pixel_count = (Predn_final.gt(0)).sum(dim=(1, 2))


    lc = alpha2 * (change_pixel_count == 0).float() * yn_cls.squeeze(1)

    return lc

def cploss(yn_cls, Predn_final, alpha1=0.9, alpha2=0.9):

    luc = Luc(Predn_final, yn_cls, alpha1)

    lc = Lc(Predn_final, yn_cls, alpha2)

    cp_loss = (yn_cls * lc + (1 - yn_cls) * luc).mean()

    return cp_loss
