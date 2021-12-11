import os, argparse
from glob import glob
from cv2 import imwrite
from util import *
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Do augumentation yolo data. (img, label)")
    parser.add_argument('--img_dir', default='./images', 
                        type=str, help='path to image dir')
    parser.add_argument('--label_dir', default='./labels',
                        type=str, help='path to label dir')
    parser.add_argument('--result_dir', default='./results',
                        type=str, help='path to result dir')
    parser.add_argument('--rotate_90', action='store_true')
    parser.add_argument('--rotate_180', action='store_true')
    parser.add_argument('--rotate_270', action='store_true')
    parser.add_argument('--gaussian_noise', action='store_true')
    parser.add_argument('--reverse_horizontal', action='store_true')
    parser.add_argument('--gamma_correction', action='store_true')

    args = parser.parse_args()
    
    result_dir = args.result_dir
    
    img_path_list = glob(args.img_dir + '/*')
    label_dir = args.label_dir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    is_all = not(
        args.rotate_90 and 
        args.rotate_180 and 
        args.rotate_270 and 
        args.gaussian_noise and 
        args.reverse_horizontal and 
        args.gamma_correction
    )
    if is_all or args.rotate_90:
        if not os.path.exists(result_dir + '/rotate_90'):
            os.makedirs(result_dir + '/rotate_90')
        if not os.path.exists(result_dir + '/rotate_90/labels'):
            os.makedirs(result_dir + '/rotate_90/labels')
        if not os.path.exists(result_dir + '/rotate_90/images'):
            os.makedirs(result_dir + '/rotate_90/images')
    if is_all or args.rotate_180:
        if not os.path.exists(result_dir + '/rotate_180'):
            os.makedirs(result_dir + '/rotate_180')
        if not os.path.exists(result_dir + '/rotate_180/labels'):
            os.makedirs(result_dir + '/rotate_180/labels')
        if not os.path.exists(result_dir + '/rotate_180/images'):
            os.makedirs(result_dir + '/rotate_180/images')
    if is_all or args.rotate_270:
        if not os.path.exists(result_dir + '/rotate_270'):
            os.makedirs(result_dir + '/rotate_270')
        if not os.path.exists(result_dir + '/rotate_270/labels'):
            os.makedirs(result_dir + '/rotate_270/labels')
        if not os.path.exists(result_dir + '/rotate_270/images'):
            os.makedirs(result_dir + '/rotate_270/images')
    if is_all or args.gaussian_noise:
        if not os.path.exists(result_dir + '/gaussian_noise'):
            os.makedirs(result_dir + '/gaussian_noise')
        if not os.path.exists(result_dir + '/gaussian_noise/labels'):
            os.makedirs(result_dir + '/gaussian_noise/labels')
        if not os.path.exists(result_dir + '/gaussian_noise/images'):
            os.makedirs(result_dir + '/gaussian_noise/images')
    if is_all or args.reverse_horizontal:
        if not os.path.exists(result_dir + '/reverse_horizontal'):
            os.makedirs(result_dir + '/reverse_horizontal')
        if not os.path.exists(result_dir + '/reverse_horizontal/labels'):
            os.makedirs(result_dir + '/reverse_horizontal/labels')
        if not os.path.exists(result_dir + '/reverse_horizontal/images'):
            os.makedirs(result_dir + '/reverse_horizontal/images')
    if is_all or args.gamma_correction:
        if not os.path.exists(result_dir + '/gamma_correction'):
            os.makedirs(result_dir + '/gamma_correction')
        if not os.path.exists(result_dir + '/gamma_correction/labels'):
            os.makedirs(result_dir + '/gamma_correction/labels')
        if not os.path.exists(result_dir + '/gamma_correction/images'):
            os.makedirs(result_dir + '/gamma_correction/images')
    for img_p in tqdm(img_path_list):
        img = import_image(img_p)
        name = os.path.splitext(os.path.basename(img_p))[0]
        label_path = label_dir + '/' + name + '.txt'
        if not os.path.exists(label_path):
            print("{} is not exist.".format(label_path))
            continue
        labels = import_label(label_path)
        if is_all or args.rotate_90:
            new_img, new_labels = rotate_90(img, labels)
            with open(result_dir + "/rotate_90/labels/{0}_rotate_90.txt".format(name), "w", encoding='UTF-8') as f:
                f.writelines(new_labels)
            imwrite(result_dir + "/rotate_90/images/{0}_rotate_90.jpg".format(name), new_img)
            
        if is_all or args.rotate_180:
            new_img, new_labels = rotate_180(img, labels)
            with open(result_dir + "/rotate_180/labels/{0}_rotate_180.txt".format(name), "w", encoding='UTF-8') as f:
                f.writelines(new_labels)
            imwrite(
                result_dir + "/rotate_180/images/{0}_rotate_180.jpg".format(name), new_img
            )
            
        if is_all or args.rotate_270:
            new_img, new_labels = rotate_270(img, labels)
            with open(result_dir + "/rotate_270/labels/{0}_rotate_270.txt".format(name), "w", encoding='UTF-8') as f:
                f.writelines(new_labels)
            imwrite(
                result_dir + "/rotate_270/images/{0}_rotate_270.jpg".format(name), new_img
            )
            
        if is_all or args.gaussian_noise:
            new_img, new_labels = gaussian_noise(img, labels, 0, 15)
            with open(result_dir + "/gaussian_noise/labels/{0}_gaussian_noise_15.txt".format(name), "w", encoding='UTF-8') as f:
                f.writelines(new_labels)
            imwrite(
                result_dir + "/gaussian_noise/images/{0}_gaussian_noise_15.jpg".format(name), new_img
            )
            
            new_img, new_labels = gaussian_noise(img, labels, 0, 30)
            with open(result_dir + "/gaussian_noise/labels/{0}_gaussian_noise_30.txt".format(name), "w", encoding='UTF-8') as f:
                f.writelines(new_labels)
            imwrite(
                result_dir + "/gaussian_noise/images/{0}_gaussian_noise_30.jpg".format(name), new_img
            )
            
        if is_all or args.reverse_horizontal:
            new_img, new_labels = reverse_horizontal(img, labels)
            with open(result_dir + "/reverse_horizontal/labels/{0}_reverse_horizontal.txt".format(name), "w", encoding='UTF-8') as f:
                f.writelines(new_labels)
            imwrite(
                result_dir + "/reverse_horizontal/images/{0}_reverse_horizontal.jpg".format(name), new_img
            )
            
        if is_all or args.gamma_correction:
            new_img, new_labels = gamma_correction(img, labels, 0.5)
            with open(result_dir + "/gamma_correction/labels/{0}_gamma_correction_0_5.txt".format(name), "w", encoding='UTF-8') as f:
                f.writelines(new_labels)
            imwrite(
                result_dir +
                "/gamma_correction/images/{0}_gamma_correction_0_5.jpg".format(
                    name), new_img
            )
            
        if is_all or args.gamma_correction:
            new_img, new_labels = gamma_correction(img, labels, 1.5)
            with open(result_dir + "/gamma_correction/labels/{0}_gamma_correction_1_5.txt".format(name), "w", encoding='UTF-8') as f:
                f.writelines(new_labels)
            imwrite(
                result_dir +
                "/gamma_correction/images/{0}_gamma_correction_1_5.jpg".format(
                    name), new_img
            )