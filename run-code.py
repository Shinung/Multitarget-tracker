import os
import shutil

'''
    tip: configure the main setting
'''

detector_list = ['yolo', 'ssd', 'dw']
codec = 'h264'
dst_ext = 'mp4' if codec == 'h264' else 'avi'

def run_code(detector, threshold, video_num):

    # ---------------- main setting ---------------- #

    # base path setting
    base_video_path = os.path.join('D:\\Roy\\GitHub\\Multitarget-tracker\\data\\')
    base_code_path = os.path.join('D:\\Roy\\GitHub\\Multitarget-tracker\\')

    # execute file setting
    exefile = 'MultitargetTracker_{0}_{1:.1f}.exe'.format(detector, threshold)

    # sorce video folder name setting
    src_folder_name = 'videos'
    src_video_name = 'video_{0}.avi'.format(video_num)

    # ------------- end of main setting ------------- #

    # execute file error checking
    if not os.path.isfile(exefile):
        adding_path = os.path.join(os.getcwd(), 'build', 'Release')
        exefile = os.path.join(adding_path, exefile)
        if not os.path.isfile(exefile):
            print("ERROR: There's no {} file in current path".format(exefile))
            exit(-1)

    # detector keyword checking
    if detector not in detector_list:
        print("ERROR: There's no {} detector in my list".format(detector))
        exit(-1)
    elif detector=='yolo':
        example = 5
        extra_setting = ''

    elif detector == 'ssd':
        example = 6

        # extra setting for ssd
        base_data_path = os.path.join(base_code_path, 'data')
        deploy_name = 'ssd_ids_trained\\Deploy_Near.prototxt'
        weight_name = 'ssd_ids_trained\\Near.caffemodel'
        labelmap_name = 'ssd_ids_trained\\labelmap_Near.prototxt'

        deploy_path = os.path.join(base_data_path, deploy_name)
        weight_path = os.path.join(base_data_path, weight_name)
        labelmap_path = os.path.join(base_data_path, labelmap_name)

        extra_setting = "--deploy={0} --weights={1} --label-map={2}".format(deploy_path, weight_path, labelmap_path)

    elif detector == 'dw':
        example = 6

        # extra setting for ssd dw
        base_data_path = os.path.join(base_code_path, 'data')
        deploy_name = 'ssd_mobilenet_dw\\MobileNetSSD_deploy.prototxt'
        weight_name = 'ssd_mobilenet_dw\\SSD_MOBILENET_V1_DW_VOC_iter_120000.caffemodel'
        labelmap_name = 'ssd_mobilenet_dw\\labelmap_voc.prototxt'

        deploy_path = os.path.join(base_data_path, deploy_name)
        weight_path = os.path.join(base_data_path, weight_name)
        labelmap_path = os.path.join(base_data_path, labelmap_name)

        extra_setting = "--deploy={0} --weights={1} --label-map={2}".format(deploy_path, weight_path, labelmap_path)

    # base video path error checking
    if not os.path.isdir(base_video_path):
        print("ERROR: There's no folder : ".format(base_video_path))
        exit(-1)

    # sorce video path setting from sorce video name
    src_folder_path = os.path.join(base_video_path, src_folder_name)
    src_video_path = os.path.join(src_folder_path, src_video_name)

    # sorce video path error checking
    if not os.path.isdir(src_folder_path):
        print("ERROR: There's no folder : ".format(src_folder_path))
        exit(-1)
    # sorce video file non-exist case checking
    elif not os.path.isfile(src_video_path):
        print("ERROR: There's no video file : ".format(src_video_path))
        exit(-1)

    dst_folder_name = src_folder_name + '_tracker_{0}_{1:.1f}'.format(detector, threshold)
    dst_folder_path = os.path.join(base_video_path, dst_folder_name)
    dst_video_name = 'video_{0}_{1}_{2:.1f}.{3}'.format(video_num, detector, threshold, dst_ext)
    dst_video_path = os.path.join(dst_folder_path, dst_video_name)

    tmp_dst_foler_path = os.path.join(dst_folder_path, 'temp')
    tmp_dst_video_path = os.path.join(tmp_dst_foler_path, dst_video_name)


    # destination video folder checking.
    if not os.path.isdir(dst_folder_path):
        print("warning: There's no folder : ".format(dst_folder_path))
        os.makedirs(dst_folder_path)
        print("So I created this one..")

    # make temp folder for converting video
    if not os.path.isdir(tmp_dst_foler_path):
        os.makedirs(tmp_dst_foler_path)

    # make command and combine with extra setting
    command = "{0} {1} --example={2} --out={3} --end_delay=100 ".format(exefile, src_video_path, example, tmp_dst_video_path)
    full_command = command + extra_setting

    # execute the full command
    try:
        print(full_command)
        os.system(full_command)
    except:
        print("Command execute error !")

    # move the video file to out of the temp folder and delete temp folder
    try:
        if os.path.isfile(tmp_dst_video_path):
            if os.path.isfile(dst_video_path):
                os.remove(dst_video_path)
            shutil.move(tmp_dst_video_path, dst_video_path)
            os.rmdir(tmp_dst_foler_path)
    except:
        print("Unknown error !")





if __name__=='__main__':

    # run_code(detector, threshold, video_num)

    #for i in range(1,3):
    #    run_code('ssd', 0.5, i)

    run_code('ssd', 0.5, 3)

