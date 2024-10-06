import numpy as np
import random
import torch
from basicsr.utils.img_process_util import filter2D
from scipy import io
from basicsr.data.transforms import augment


def gamma_preprocess(gt_usm, data_label, batch_size, CTX_light_range, SELENE_light_range):
    mean_values = torch.mean(gt_usm, dim=(1, 2, 3))
    CTX_index =  [index for index, element in enumerate(data_label) if element == 'CTX']
    CTX_index1 = [index in CTX_index for index in range(batch_size)]
    CTX_index1 = torch.tensor(CTX_index1)
    selene_index = [not value for value in CTX_index1]
    selene_index = torch.tensor(selene_index)

    if torch.sum(CTX_index1) > 0:
        ctx = []
        for index, value in enumerate(CTX_index1):
            if mean_values[index] < (CTX_light_range[0]/255):
                ctx.append('Low')
            elif mean_values[index] > (CTX_light_range[1]/255):
                ctx.append('High')
            else:
                ctx.append('Normal')
    if torch.sum(selene_index) > 0:
        selene = []
        for index, value in enumerate(selene_index):
            if mean_values[index] < (SELENE_light_range[0]/255):
                selene.append('Low')
            elif mean_values[index] > (SELENE_light_range[1]/255):
                selene.append('High')
            else:
                selene.append('Normal')
    new_list = []
    number_of_true = []

    if torch.sum(CTX_index1) > 0 :
        number_of_true = np.sum(ctx == 'Normal')  # torch.sum(indicesCTX == 'Normal')
        new_list = ctx
    elif torch.sum(selene_index) > 0:
        number_of_true = np.sum(selene == 'Normal')
        new_list = selene
    return new_list, number_of_true


def gamma_correction(img, indices, gamma, batchsize):

    inv_gamma = gamma

    max_values, _ = torch.topk(img.view(batchsize, -1), k=1, dim=1, largest=True)
    min_values, _ = torch.topk(img.view(batchsize, -1), k=1, dim=1, largest=False)
    max_values = max_values.unsqueeze(2).unsqueeze(3)
    min_values = min_values.unsqueeze(2).unsqueeze(3)

    normal = (img - min_values) / (max_values - min_values)


    gamma = (normal ** inv_gamma) * (max_values - min_values) + min_values
    img[indices,:,:,:] = gamma[indices,:,:,:]

    return img


def gamma_correction_new(img, gamma, batchsize):

    inv_gamma = gamma

    max_values, _ = torch.topk(img.view(batchsize, -1), k=1, dim=1, largest=True)
    min_values, _ = torch.topk(img.view(batchsize, -1), k=1, dim=1, largest=False)

    max_values = max_values.unsqueeze(2).unsqueeze(3)
    min_values = min_values.unsqueeze(2).unsqueeze(3)

    normal = (img - min_values) / (max_values - min_values)

    gamma = (normal ** inv_gamma) * (max_values - min_values) + min_values

    return gamma

def gamma_correction_new1(img, gamma):

    inv_gamma = gamma


    max_values, _ = torch.topk(img, k=1, dim=1, largest=True)
    min_values, _ = torch.topk(img, k=1, dim=1, largest=False)

    max_values = max_values.unsqueeze(2).unsqueeze(3)
    min_values = min_values.unsqueeze(2).unsqueeze(3)


    normal = (img - min_values) / (max_values - min_values + 1e-6)


    gamma = (normal ** inv_gamma) * (max_values - min_values) + min_values

    return gamma

def add_missing(array,  missing_data_path, device, batchsize):

    data = io.loadmat(missing_data_path)
    missing_dataset = data['missing_template']


    batchsize = int(batchsize)
    random_num = random.sample(range(0, len(missing_dataset)), batchsize)
    missing = np.array(missing_dataset[random_num])


    random_x = random.sample(range(0, np.shape(missing_dataset)[1] - array.shape[2]), 1)
    random_y = random.sample(range(0, np.shape(missing_dataset)[2] - array.shape[3]), 1)
    missing = missing[:, random_x[0]:random_x[0]+array.shape[2], random_y[0]:random_y[0]+array.shape[3]]
    missing = missing.astype(np.int16)
    tensor_missing = torch.tensor(missing).unsqueeze(1)
    tensor_missing = tensor_missing.to(device)

    result_array = array * tensor_missing

    return result_array


def poisson(array):

    array[array < 0] = 0
    poisson_noise = torch.poisson(array[:,:,:,:])
    noise = poisson_noise - array
    return noise

def add_noise(array,  VS_template_path, device, batchsize):

    data = io.loadmat(VS_template_path)
    vs_dataset = data['VS_template']

    random_num = random.sample(range(0, len(vs_dataset)), batchsize)
    VS = np.array(vs_dataset[random_num])


    random_x = random.sample(range(0, np.shape(vs_dataset)[1] - array.shape[2]), 1)
    random_y = random.sample(range(0, np.shape(vs_dataset)[2] - array.shape[3]), 1)
    VS = VS[:, random_x[0]:random_x[0]+array.shape[2], random_y[0]:random_y[0]+array.shape[3]]/255
    VS = VS.astype(np.float16)
    tensor_dark_bias = torch.tensor(VS).unsqueeze(1)
    tensor_dark_bias = tensor_dark_bias.to(device)

    poisson_noise = poisson(array)

    return tensor_dark_bias, poisson_noise


def degradation_NEW(gt_usm, use_hflip, use_rot, kernel, blur_prob, missing_template_path, device, data_label,
                    CTX_light_range, SELENE_light_range, batch_size, gap_prob, ):
    

    gt_usm = gt_usm.clone().permute(0, 2, 3, 1).cpu().numpy()

    ## 1 Light intensity
    # Do augmentation for training: flip, rotation
    array = []
    for i in range(gt_usm.shape[0]):
        aug = augment(gt_usm[i,:,:,:], use_hflip[i].item(), use_rot[i].item())
        array.append(aug)
    gt_usm = torch.from_numpy(np.array(array).transpose(0, 3, 1, 2)).to(device)


    out = gt_usm.clone()
    new_list, number_of_true = gamma_preprocess(gt_usm, data_label, batch_size, CTX_light_range, SELENE_light_range)
    for index, value in enumerate(new_list):
        if value == 'Low':
            gamma = random.uniform(0.5, 1)
        elif value == 'High':
            gamma = random.randint(1, 3)
        else:
            gamma = 1

        sub_img = gamma_correction_new1(gt_usm[index], gamma)
        sub_img = torch.clamp(sub_img, 0, 1)
        gt_usm[index] = sub_img


    ## 2 Gap
    if np.random.uniform() < gap_prob:
        out = add_missing(out, missing_template_path, device, batch_size)

    ## 3 blur
    if np.random.uniform() < blur_prob:
        out = filter2D(out, kernel)

    return out, gt_usm