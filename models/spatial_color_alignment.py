import torch
import torch.nn.functional as F

def match_colors(im_ref, im_q, im_test):

    im_ref_mean_re = im_ref.view(*im_ref.shape[:2], -1)
    im_q_mean_re = im_q.view(*im_q.shape[:2], -1)

    # Estimate color transformation matrix by minimizing the least squares error
    c_mat_all = []
    for ir, iq in zip(im_ref_mean_re, im_q_mean_re):
        c = torch.lstsq(ir.t(), iq.t())
        c = c.solution[:im_ref_mean_re.size(1)]
        c_mat_all.append(c)

    c_mat = torch.stack(c_mat_all, dim=0)
    # Apply the transformation to test image
    im_test_re = im_test.view(*im_test.shape[:2], -1)
    im_t_conv = torch.matmul(im_test_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_t_conv = im_t_conv.view(im_test.shape)

    return im_t_conv

def color_correction(gt, in_put, output, scale_factor=4):
    ds_gt = F.interpolate(gt, scale_factor=1.0 / scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True)
    output_cor = match_channel_colors(ds_gt, in_put, output)
    return output_cor

def match_channel_colors(im_ref, im_q, im_test):

    im_ref_reshape = im_ref.view(*im_ref.shape[:2], -1)
    im_q_reshape = im_q.view(*im_q.shape[:2], -1)
    im_test_reshape = im_test.view(*im_test.shape[:2], -1)
    # Estimate color transformation matrix by minimizing the least squares error

    im_t_conv_list = []
    for i in range(im_ref.size(1)):
        c_mat_all = []
        for ir_batch, iq_batch in zip(im_ref_reshape[:, i:i+1, :], im_q_reshape[:, i:i+1, :]):
            c = torch.lstsq(ir_batch.t(), iq_batch.t())
            c = c.solution[:1]
            c_mat_all.append(c)

        c_mat = torch.stack(c_mat_all, dim=0)
        # Apply the transformation to test image
        im_t_conv = torch.matmul(im_test_reshape[:, i:i+1, :].permute(0, 2, 1), c_mat).permute(0, 2, 1)
        im_t_conv = im_t_conv.view(*im_t_conv.shape[:2], *im_test.shape[-2:])
        im_t_conv_list.append(im_t_conv)

    im_t_conv = torch.cat(im_t_conv_list, dim=1)

    return im_t_conv
