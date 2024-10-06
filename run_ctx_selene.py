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



# result = subprocess.run(['python', '.\inference_realesrgan.py',
#                      '-n'+r'B:\ESRGAN_new_Fan\experiments\finetune_RealESRGANx4plus_SELENE\models\net_g_115000.pth',
#                      '-i'+ r'B:\Image_Data\TEST_ESRGAN_Predict\2_downscale\TCO_MAP_02_N03E231N00E234SC_Mercator_downscalex4_8bit.tif',
#                      '-s4.0',
#                      '-o'+ r'B:\Image_Data\TEST_ESRGAN_Predict\3_predict\SELENE'])
model = [r'B:\ESRGAN_new_Fan\experiments\train_RealESRGANx4plus_CTX\models\net_g_25000.pth',
         r'B:\ESRGAN_new_Fan\experiments\train_RealESRGANx4plus_SELENE\models\net_g_30000.pth',
         r'B:\ESRGAN_new_Fan\experiments\train_RealESRGANx4plus_CTX_SELENE\models\net_g_30000.pth'

]
# model = [r'B:\ESRGAN_new_Fan\experiments\finetune_RealESRGANx4plus_CTX\models\net_g_30000.pth',
#          r'B:\ESRGAN_new_Fan\experiments\finetune_RealESRGANx4plus_SELENE\models\net_g_115000.pth',
#          r'B:\ESRGAN_new_Fan\experiments\finetune_RealESRGANx4plus_CTX_SELENE\models\net_g_40000.pth']
folder = [r'B:\Image_Data\TEST_ESRGAN_Predict\2_downscale_CTX', r'B:\Image_Data\TEST_ESRGAN_Predict\2_downscale_SELENE']
save = [r'B:\Image_Data\TEST_ESRGAN_Predict\3_predict\CTX', r'B:\Image_Data\TEST_ESRGAN_Predict\3_predict\SELENE']
for a in range(0,len(model)):
    if a == 0:
        result = subprocess.run(['python', '.\inference_realesrgan.py',
                             '-n'+ model[a],
                             '-i'+ folder[0],#r'B:\small_body\Ganymede\1_clip',
                             '-s4.0',
                             '-o'+ save[0]+'\\_CTX.tif'])
    elif a == 1:
        result = subprocess.run(['python', '.\inference_realesrgan.py',
                             '-n'+ model[a],
                             '-i'+ folder[1],#r'B:\small_body\Ganymede\1_clip',
                             '-s4.0',
                             '-o'+ save[1]+'\\_SELENE.tif'])
    else:
        for i in range(0, len(folder)):
            result = subprocess.run(['python', '.\inference_realesrgan.py',
                                     '-n' + model[a],
                                     '-i' + folder[i],  # r'B:\small_body\Ganymede\1_clip',
                                     '-s4.0',
                                     '-o' + save[i]+'\\_CTX_SELENE.tif'])
