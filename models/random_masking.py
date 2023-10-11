import torch
from PIL import Image
from torchvision import transforms

def random_masking(x, mask_ratio):
    '''Perform per-sample random masking by per-sample shuffling.

        Parameters:
            x (tensor): the tensor to mask
            mask_ratio (float): the percent of masked pixles

        Returns:
            the masked tensor and the mask with the same size of masked tensor
            '''
    # potential issues: how to choose mask in an unrandom manner
    B, C, H, W = x.shape  # bach, channel, height, width

    # masking in a sequence manner
    x_reshape = x.permute(0, 2, 3, 1).flatten(1, 2)
    L = H * W
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 1 is keep, 0 is remove
    mask_reshape = torch.zeros([B, L], device=x.device)
    mask_reshape[:, :len_keep] = 1
    # unshuffle to get the binary mask
    mask_reshape = torch.gather(mask_reshape, dim=1, index=ids_restore).unsqueeze(2).repeat(1, 1, C)
    x_masked_reshape =  mask_reshape * x_reshape
    # reshape to origin shape
    x_masked = x_masked_reshape.reshape(B, H, W, C).permute(0, 3, 1, 2)
    mask = mask_reshape.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return x_masked, mask

if __name__ == '__main__':
    image = Image.open("../datasets/single_image_monet_etretat/trainA/monet.jpg").convert('RGB')
    Image._show(image)
    trans1 = transforms.ToTensor()
    x = trans1(image)
    x = x.unsqueeze(0)
    x_mask, _ = random_masking(x, 0.8)
    x_mask = x_mask.squeeze(0)
    trans2 = transforms.ToPILImage()
    image_mask = trans2(x_mask)
    Image._show(image_mask)