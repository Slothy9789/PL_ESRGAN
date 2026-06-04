import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from osgeo import gdal
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
import numpy as np
import tifffile as tif

def main():
    """Inference demo for Real-ESRGAN.
    """

    model = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4


    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default=r'', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
              'Only used for the realesr-general-x4v3 model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '-m','--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('--suffix', type=str, default='pl_esrgan', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--geo_info', action='store_true', help='save geo info of input image')
    parser.add_argument('--Maxvalue', type=float, default=255)
    parser.add_argument('--Minvalue', type=float, default=0)
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    parser.add_argument('--gap_zero_thr', type=int, default=0, help='Pixels <= thr are treated as zero')
    parser.add_argument('--gap_min_area', type=int, default=50, help='Min connected area (pixels)')
    parser.add_argument('--gap_min_width', type=int, default=2, help='Min connected width')
    parser.add_argument('--gap_min_height', type=int, default=30, help='Min connected height')
    parser.add_argument('--gap_dilate_iter', type=int, default=0, help='Dilation iterations for gap mask')
    parser.add_argument('--gap_filling', type=bool, default=True, help='Whether to save the gap mask')



    args = parser.parse_args()
    file_url = []
    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]

    # determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        # print(args.model_path)
        model_path = os.path.join('weights', args.model_name + '.pth')
        # print(model_path)
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)

    # use dni to control the denoise strength
    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    # print(args.output)
    os.makedirs(args.output, exist_ok=True)
    print(args.input)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        if path.endswith(".tif"):
            print('Predicting', idx, imgname)
            img = tif.imread(path)     #img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if args.Maxvalue != 255 and args.Minvalue != 0:
                print('NO 0-255 Range')
                denom = float(args.Maxvalue - args.Minvalue)
                if denom == 0:
                    img[:] = 0.0
                else:
                    img = (img - float(args.Minvalue)) * 255.0 / denom
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            if np.ndim(img) == 4 and np.shape(img)[0] == 1 and np.shape(img)[1] == 1:  #(1,1,64,64) 
                img =img[0,0,:,:]       
            elif np.ndim(img) == 3 and np.shape(img)[2] == 1 :  #(64,64, 1)
                img =img[:, :, 0] 
            elif np.ndim(img) == 3 and np.shape(img)[0] == 1 :  #(1, 64,64)
                img =img[0,:, :] 


            if len(img.shape) == 3 and img.shape[2] == 4:
                img_mode = 'RGBA'
            else:
                img_mode = None

            # ------------------ Generate connected zero-patch gap mask (black=gap, white=non-gap) ------------------
            # 1) Candidate: pixels <= threshold are treated as zero
            zero_thr = int(args.gap_zero_thr)
            zero_mask0 = (img <= zero_thr).astype(np.uint8)  # 0/1

            # 2) Connected components (8-connectivity)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(zero_mask0, connectivity=8)

            H, W = img.shape

            # ---- thresholds for a tall vertical stripe ----
            min_h_rel   = 0.05      # must cover at least 60% of image height
            min_area_rel = 0.0003    # at least 0.3% of image area
            min_aspect  = 0.5       # h/w >= 6  (thin & tall)
            border_tol_rel = 0.02   # ~2% of height; require touching top/bottom within this tolerance

            min_h = int(min_h_rel * H)
            min_area = int(min_area_rel * H * W) # also guard small images
            min_w = int(args.gap_min_width)
            border_tol = max(1, int(border_tol_rel * H))

            candidates = []

            for i in range(1, num):  # 0 is background
                x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, cv2.CC_STAT_AREA]
                if area < min_area:  # 
                    continue
                if h < min_h:       # 
                    continue
                if w < min_w:       # 
                    continue
                aspect = h / (w + 1e-6)   # 
                if aspect < min_aspect:
                    continue
                # must roughly touch both top and bottom (a seam-like stripe)
                touches_top    = (y <= border_tol)   # 
                touches_bottom = (y + h >= H - border_tol)  # 
                if not (touches_top or touches_bottom):
                    continue
                candidates.append(i)

            selected = np.zeros_like(zero_mask0, dtype=np.uint8)

            # --- keep candidates if any; otherwise leave empty (no-gap) ---
            if len(candidates) > 0:
                # if multiple, keep the tallest one
                heights = [stats[i, 3] for i in candidates]
                idx = candidates[int(np.argmax(heights))]
                selected[labels == idx] = 1
            # else: keep selected as all zeros → means “no gap found”

            # 4) Optional: small dilation for continuity
            dilate_iter = int(args.gap_dilate_iter)
            if dilate_iter > 0:
                kernel = np.ones((3, 3), np.uint8)
                selected = cv2.dilate(selected, kernel, iterations=dilate_iter)

            # 5) Final binary mask: gap=0, others=255
            gap_mask = np.where(selected == 1, 0, 255).astype(np.uint8)

            # 6) Upscale the mask (to match super-resolution scale)
            h, w = gap_mask.shape
            gap_mask = cv2.resize(gap_mask, (w * netscale, h * netscale), interpolation=cv2.INTER_NEAREST)
            # ------------------ End of gap mask generation ------------------



            try:
                output = upsampler.enhance(img, outscale=args.outscale)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
            else:
                if args.ext == 'auto':
                    extension = extension[1:]
                else:
                    extension = args.ext
                if img_mode == 'RGBA':  # RGBA images should be saved in png format
                    extension = 'png'
                if args.suffix == '':
                    save_path = os.path.join(args.output, f'{imgname}_SR.{extension}')
                else:
                    save_path = os.path.join(args.output, f'{imgname}_SR.{extension}')

                output_without_gap = output_without_gap.astype(np.uint8)
                output = output_without_gap.copy()
                output[gap_mask == 0] = 0
                gap_mask = gap_mask.astype(np.uint8)
                
                if args.geo_info == True:
                    dataset = gdal.Open(path)
                    driver = gdal.GetDriverByName("GTiff")
                    proj = dataset.GetProjection()
                    gt = dataset.GetGeoTransform()
                    gt_list = list(gt)
                    gt_list[1] = gt_list[1] / netscale
                    gt_list[5] = gt_list[5] / netscale
                    gt_tuple = tuple(gt_list)

                    # 
                    New_YG_dataset = driver.Create(save_path, np.shape(output)[1], np.shape(output)[0], 1,
                                                   gdal.GDT_Byte)  # , gdal.GDT_Int32
                    New_YG_dataset.SetGeoTransform(gt_tuple)
                    New_YG_dataset.SetProjection(proj)
                    band1 = New_YG_dataset.GetRasterBand(1)
                    band1.WriteArray(output, 0, 0) 
                    New_YG_dataset = None

                    if args.gap_filling:
                        mask_save_path = os.path.join(args.output, f'{imgname}_GapMask.tif')
                        output_without_gap_save_path = os.path.join(args.output, f'{imgname}_SR_GapFilling.tif')

                        New_ds = driver.Create(mask_save_path, gap_mask.shape[1], gap_mask.shape[0], 1, gdal.GDT_Byte)
                        New_ds.SetGeoTransform(gt_tuple)
                        New_ds.SetProjection(proj)
                        New_ds.GetRasterBand(1).WriteArray(gap_mask, 0, 0)
                        New_ds.FlushCache()
                        New_ds = None

                        New_ds1 = driver.Create(output_without_gap_save_path, output_without_gap.shape[1], output_without_gap.shape[0], 1, gdal.GDT_Byte)
                        New_ds1.SetGeoTransform(gt_tuple)
                        New_ds1.SetProjection(proj)
                        New_ds1.GetRasterBand(1).WriteArray(output_without_gap, 0, 0)
                        New_ds1.FlushCache()
                        New_ds1 = None
                else:
                    tif.imwrite(save_path, output)   
                    if args.gap_filling:      
                        mask_save_path = os.path.join(args.output, f'{imgname}_GapMask.tif')
                        output_without_gap_save_path = os.path.join(args.output, f'{imgname}_SR_GapFilling.tif')
                        # gap_mask = cv2.flip(gap_mask, 0)
                        tif.imwrite(mask_save_path, gap_mask)
                        tif.imwrite(output_without_gap_save_path, output_without_gap)


if __name__ == '__main__':
    main()
