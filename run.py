'''
我自己写的代码，为了循环去运行inference_realesrgan.py代码
为了自动处理多景月球影像，进行超分辨率
'''
import os
import subprocess

# input_path = r'F:\2023_SR\MetaSR-PyTorch-master\input_segment_finished'
# save_folder = r'F:\2023_SR\ESRGAN\Moon'
# list_folder = os.listdir(input_path)
# full_path = []
# save_path = []
# for i in list_folder:
#     full = os.path.join(input_path, i)
#     full_path.append(full)
#     save = os.path.join(save_folder, i)
#     if not os.path.exists(save):
#         os.mkdir(save)
#     save_path.append(save)
# for i in range(len(save_path)):
#     result = subprocess.run(['python', R'F:\2023_SR\ESRGAN\Real-ESRGAN-master\inference_realesrgan.py',
#                              # '-nnet_g_10000',
#                              '-i'+full_path[i],
#                              '-s3.7',
#                              '-o'+save_path[i]])


folder = [r'B:\small_body\Callisto\1_clip_50d', r'B:\small_body\Dione\1_clip_50d',
          r'B:\small_body\Enceladus\1_clip_50d', r'B:\small_body\Europa\1_clip_50d',
          r'B:\small_body\Ganymede\1_clip_50d', r'B:\small_body\Iapetus\1_clip_50d',
          r'B:\small_body\mimas\1_clip_50d', r'B:\small_body\Tethys\1_clip_50d']
save = [r'B:\small_body\Callisto', r'B:\small_body\Dione', r'B:\small_body\Enceladus',
                     r'B:\small_body\Europa', r'B:\small_body\Ganymede', r'B:\small_body\Iapetus',
                     r'B:\small_body\mimas', r'B:\small_body\Tethys']
for i in range(0,len(folder)):
    result = subprocess.run([r'python', '.\inference_realesrgan.py',
                         '-n'+r'B:\ESRGAN_new_w\experiments\finetune_RealESRGANx4plus_400k_Small_bodies_select\models\net_g_20000.pth', #r'finetune_RealESRGANx4plus_400k_Small_bodies_select',
                         '-i'+ folder[i],#r'B:\small_body\Ganymede\1_clip',
                         '-s4.0',
                         '-o'+ save[i]+r'\13_predict_small_bodies'])



