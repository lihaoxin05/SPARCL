import torch


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert feat.size()[0] == 3
    assert (isinstance(feat, torch.FloatTensor)) or (
        isinstance(feat, torch.cuda.FloatTensor)
    )
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (
        source_f - source_f_mean.expand_as(source_f)
    ) / source_f_std.expand_as(source_f)
    source_f_cov_eye = torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3).cuda()

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (
        target_f - target_f_mean.expand_as(target_f)
    ) / target_f_std.expand_as(target_f)
    target_f_cov_eye = torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3).cuda()

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)), source_f_norm),
    )

    source_f_transfer = source_f_norm_transfer * target_f_std.expand_as(
        source_f_norm
    ) + target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


def _mat_sqrt_batch(x):
    U, D, V = torch.svd(x)
    return torch.bmm(torch.bmm(U, torch.diag_embed(D.pow(0.5))), torch.transpose(V, 1, 2))


def _calc_batch_feat_flatten_mean_std(feat):
    # takes 4D feat (B, C, H, W), return mean and std of array within channels
    assert len(feat.size()) == 4 and feat.size()[1] == 3
    assert (isinstance(feat, torch.FloatTensor)) or (
        isinstance(feat, torch.cuda.FloatTensor)
    )
    bs = feat.shape[0]
    feat_flatten = feat.view(bs, 3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def batch_coral(source, target):
    # assume both source and target are 4D array (B, C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_batch_feat_flatten_mean_std(source)
    source_f_norm = (
        source_f - source_f_mean.expand_as(source_f)
    ) / source_f_std.expand_as(source_f)
    
    source_f_cov_eye = torch.bmm(source_f_norm, torch.transpose(source_f_norm, 1, 2)) + torch.eye(3).unsqueeze(0).cuda()

    target_f, target_f_mean, target_f_std = _calc_batch_feat_flatten_mean_std(target)
    target_f_norm = (
        target_f - target_f_mean.expand_as(target_f)
    ) / target_f_std.expand_as(target_f)
    target_f_cov_eye = torch.bmm(target_f_norm, torch.transpose(target_f_norm, 1, 2)) + torch.eye(3).unsqueeze(0).cuda()

    source_f_norm_transfer = torch.bmm(
        _mat_sqrt_batch(target_f_cov_eye),
        torch.bmm(torch.inverse(_mat_sqrt_batch(source_f_cov_eye)), source_f_norm),
    )

    source_f_transfer = source_f_norm_transfer * target_f_std.expand_as(
        source_f_norm
    ) + target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())
