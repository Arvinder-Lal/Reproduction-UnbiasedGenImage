import argparse
import os
import time
import util
import torch
 

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='res50', help='architecture for binary classification')

        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0)
        parser.add_argument('--blur_sig', default='0.5')
        parser.add_argument('--jpg_prob', type=float, default=0)
        parser.add_argument('--jpg_method', default='cv2')
        parser.add_argument('--jpg_qual', default='75')

        parser.add_argument('--dataroot', default='./dataset/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--classes', default='', help='image classes to train on')
        parser.add_argument('--class_bal', action='store_true')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--delr_freq', type=int, default=20, help='frequency of changing lr')
        parser.add_argument('--tiny_preprocess', type=str, default="right_diag", help='image preprocess type used')
        parser.add_argument('--features_dim', type=int, default=2048, help='nn.Linear(features_dim, num_of_classes)')
        parser.add_argument('--ladeda_preprocess', type=str, default="NPR", help='image preprocess type used')
        parser.add_argument('--preprocess', type=str, default="NPR", help='image preprocess type used')
        parser.add_argument('--model', type=str, default="LaDeDa", help='image preprocess type used')
        
        #hinzugefügt
        parser.add_argument('--csv_data_path', type=str, default="/pfs/work9/workspace/scratch/ma_alal-myws/UnbiasedGenImage/GenImage/genimage_metadata.csv", help='Path to csv file')
        parser.add_argument('--base_path', type=str, default="/pfs/work9/workspace/scratch/ma_alal-myws/UnbiasedGenImage", help='Path to GenImage folder')
        parser.add_argument('--dataset', type=str, default="jpeg96", help='One of the three modes: classic, jpeg96, size_constrained')
        parser.add_argument('--generator', type=str, default=None, choices=['Midjourney', 'stable_diffusion_v_1_5', 'stable_diffusion_v_1_4', 'wukong', 'ADM',
                                    'VQDM', 'glide', 'BigGAN'], help='One of the 8 generators')

        parser.add_argument('--class-map', default='../../class_map.txt', type=str, metavar='FILENAME',
                                  help='path to class to idx mapping file')
        parser.add_argument('--balance_val_classes', action='store_true', default=False,
                            help='whether or not to balance val data so that the class distribution is equal in ai images and nature images \
                                (number of instances per imagenet class is same in ai images and nature images)')
        parser.add_argument('--balance_train_classes', action='store_true', default=False,
                            help='whether or not to balance train data so that the class distribution is equal in ai images and nature images \
                                (number of instances per imagenet class is same in ai images and nature images)')
        parser.add_argument('--is_validating', action='store_true', default=False)

        
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return opt #parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
#        opt.name = opt.name + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # Conditional settings based on --model
        if opt.model in ['LaDeDa', 'ladeda']:
            opt.features_dim = 2048
            opt.preprocess = opt.ladeda_preprocess
        elif opt.model in ['Tiny', 'tiny']:
            opt.features_dim = 8
            opt.preprocess = opt.tiny_preprocess
        else:
            raise ValueError("Model type should be LaDeDa\ladeda or tiny\Tiny")



        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        opt.classes = opt.classes.split(',')
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        self.opt = opt
        return self.opt
