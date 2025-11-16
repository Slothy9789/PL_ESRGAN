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

def main():
    """Inference demo for Real-ESRGAN.
    """

    model = RRDBNet(num_in_ch=1, num_out_ch=1, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4


    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input image or folder')
    # parser.add_argument(
    #     '-n',
    #     '--model_name',
    #     type=str,
    #     default='RealESRGAN_x4plus',
    #     help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
    #           'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    # parser.add_argument('-o', '--output', type=str, default='results_segment', help='Output folder')
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
    parser.add_argument('--save_gap_mask', type=bool, default=True, help='Whether to save the gap mask')


    args = parser.parse_args()
    file_url = []
    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    # if args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
    #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    #     netscale = 4
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    # elif args.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
    #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    #     netscale = 4
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    # elif args.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
    #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    #     netscale = 4
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    # elif args.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
    #     model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    #     netscale = 2
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    # elif args.model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
    #     model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
    #     netscale = 4
    #     file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    # elif args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
    #     model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
    #     netscale = 4
    #     file_url = [
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
    #         'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
    #     ]
    # -----------------------------------------------------------------------------------------------------------------------------------------



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

        print('Predicting', idx, imgname)
        import tifffile as tif       # -----------------------------------------------------------------------------------
        if path.endswith(".tif"):
            img = tif.imread(path)     #img = cv2.imread(path, cv2.IMREAD_UNCHANGED) ----------------------------------------------
            if args.Maxvalue != 255 and args.Minvalue != 0:
                
                # 
                denom = float(args.Maxvalue - args.Minvalue)
                if denom == 0:
                    # 
                    img[:] = 0.0
                else:
                    img = (img - float(args.Minvalue)) * 255.0 / denom
                # 
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            # print(np.shape(img))

            if np.ndim(img) == 4 and np.shape(img)[0] == 1 and np.shape(img)[1] == 1:  #(1,1,64,64)  # -----------------------------
                img =img[0,0,:,:]         # --------------------------------------------------------------------------------------
            elif np.ndim(img) == 3 and np.shape(img)[2] == 1 :  #(64,64, 1)  # -----------------------------
                img =img[:, :, 0]         # --------------------------------------------------------------------------------------
            elif np.ndim(img) == 3 and np.shape(img)[0] == 1 :  #(1, 64,64)  # -----------------------------
                img =img[0,:, :]         # --------------------------------------------------------------------------------------


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
            min_area = max(1000, int(min_area_rel * H * W))  # also guard small images
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
                # if args.face_enhance:
                #     _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                # else:
                      #output, _ = upsampler.enhance(img, outscale=args.outscale)
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
                    # save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
                    save_path = os.path.join(args.output, f'{imgname}_SR.{extension}')

                if args.geo_info == True:
                    dataset = gdal.Open(path)
                    driver = gdal.GetDriverByName("GTiff")
                    # 
                    proj = dataset.GetProjection()
                    gt = dataset.GetGeoTransform()
                    gt_list = list(gt)
                    gt_list[1] = gt_list[1] / netscale
                    gt_list[5] = gt_list[5] / netscale
                    gt_tuple = tuple(gt_list)

                    # 
                    New_YG_dataset = driver.Create(os.path.join(args.output, save_path),
                                                   np.shape(output)[1], np.shape(output)[0], 1,
                                                   gdal.GDT_Byte)  # , gdal.GDT_Int32
                    New_YG_dataset.SetGeoTransform(gt_tuple)
                    New_YG_dataset.SetProjection(proj)
                    band1 = New_YG_dataset.GetRasterBand(1)
                    band1.WriteArray(output, 0, 0)  # 
                    New_YG_dataset = None

                    if args.save_gap_mask:
                        mask_save_path = os.path.join(args.output, f'{imgname}_GapMask.tif')
                        New_ds = driver.Create(mask_save_path, gap_mask.shape[1], gap_mask.shape[0], 1, gdal.GDT_Byte)
                        New_ds.SetGeoTransform(gt_tuple)
                        New_ds.SetProjection(proj)
                        New_ds.GetRasterBand(1).WriteArray(gap_mask, 0, 0)
                        New_ds.FlushCache()
                        New_ds = None
                else:
                    output = output.astype(np.uint8)   # -------------------------------------------------------------------------
                    # output = cv2.flip(output, 0)
                    tif.imwrite(save_path, output)     # -------------------------------------------------------------------------
                    if args.save_gap_mask:             # -------------------------------------------------------------------------
                        mask_save_path = os.path.join(args.output, f'{imgname}_gapmask.tif')
                        gap_mask = gap_mask.astype(np.uint8)
                        # gap_mask = cv2.flip(gap_mask, 0)
                        tif.imwrite(mask_save_path, gap_mask)


if __name__ == '__main__':
    main()
