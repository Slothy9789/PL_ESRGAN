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



# ### gap filling
# result = subprocess.run(['python', '.\inference_realesrgan.py',
#                      '-n'+r'B:\ESRGAN_new_Fan\experiments\finetune_RealESRGANx4plus_SELENE_degradationModel\models\net_g_60000_512.pth',
#                      '-i'+ r'B:\Image_Data\true_bad_data\missing', # r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit.tif',
#                      '-s4.0',
#                      '-o'+ r'B:\Image_Data\true_bad_data\missing_predict' #r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit_out.tif'
#                      ])


# ### vs
# result = subprocess.run(['python', '.\inference_realesrgan.py',
#                      '-n'+r'B:\ESRGAN_new\experiments\finetune_RealESRGANx4plus_CTX_SELENE_for_testing_degradationModel\models\net_g_60000.pth',
#                      '-i'+ r'B:\Image_Data\true_bad_data\vs', # r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit.tif',
#                      '-s4.0',
#                      '-o'+ r'B:\Image_Data\true_bad_data\vs_predict' #r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit_out.tif'
#                      ])

# ### light
# result = subprocess.run(['python', '.\inference_realesrgan.py',
#                      '-n'+r'B:\ESRGAN_new\experiments\finetune_RealESRGANx4plus_CTX_SELENE_for_testing_degradationModel_w\models\net_g_120000.pth',
#                      '-i'+ r'B:\Image_Data\true_bad_data\light_clip\N00E00', # r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit.tif',
#                      '-s4.0',
#                      '-o'+ r'B:\Image_Data\true_bad_data\light_predict\N00E00_w' #r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit_out.tif'
#                      ])

# ### blur
# result = subprocess.run(['python', '.\inference_realesrgan.py',
#                      '-n'+r'B:\ESRGAN_new\experiments\finetune_RealESRGANx4plus_CTX_SELENE_for_testing_degradationModel_blur_kernel1\models\net_g_60000.pth',
#                      # '-n' + r'B:\ESRGAN_new\experiments\finetune_RealESRGANx4plus_CTX_SELENE_for_testing_degradationModel\models\net_g_60000.pth',
#                      # '-n'+r'B:\ESRGAN_new_Fan\experiments\finetune_RealESRGANx4plus_CTX_SELENE\models\net_g_40000.pth',
#                      '-i'+ r'B:\Image_Data\true_bad_data\light_clip\N00E00', # r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit.tif',
#                      '-s4.0',
#                      '-o'+ r'B:\Image_Data\true_bad_data\blur_clip_predict_' #r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit_out.tif'
#                      ])

### Mercury
result = subprocess.run(['python', '.\inference_realesrgan.py',
                     '-n' + r'B:\ESRGAN_new_Fan\experiments\finetune_RealESRGANx4plus_Mercury_degradationModel\models\net_g_20000.pth',
                     '-i'+ r'B:\Image_Data\FOR_SHOW', # r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit.tif',
                     '-s4.0',
                     '-o'+ r'B:\Image_Data\FOR_SHOW' #r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit_out.tif'
                     ])

# # ### ctx
#
# result = subprocess.run(['python', '.\inference_realesrgan.py',
#                      '-n' + r'B:\ESRGAN_new_Fan\experiments\finetune_RealESRGANx4plus_CTX\models\net_g_30000.pth',
#                      '-i'+ r'B:\Image_Data\true_bad_data\CTX\CLIP', # r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit.tif',
#                      '-s4.0',
#                      '-o'+ r'B:\Image_Data\true_bad_data\CTX' #r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit_out.tif'
#                      ])
# # ### selene
#
# result = subprocess.run(['python', '.\inference_realesrgan.py',
#                      '-n' + r'B:\ESRGAN_new_Fan\experiments\finetune_RealESRGANx4plus_SELENE_degradationModel_512\models\net_g_60000.pth',
#                      '-i'+ r'B:\Image_Data\true_bad_data\SELENE\CLIP', # r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit.tif',
#                      '-s4.0',
#                      '-o'+ r'B:\Image_Data\true_bad_data\SELENE' #r'B:\Image_Data\true_bad_data\missing\TCO_MAP_02_N18E156N15E159SC_Mercator_8bit_out.tif'
#                      ])