from .base_options import BaseOptions

 
class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--loss_freq', type=int, default=400, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_latest_freq', type=int, default=2000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')

        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')

        parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--is_aug', action='store_true')
        # parser.add_argument('--model_path')
        # parser.add_argument('--no_resize', action='store_true')
        # parser.add_argument('--no_crop', action='store_true')

        # hinzugefügt
        parser.add_argument('--size_constrained', action='store_true', help='Enable size constraint')
        parser.add_argument('--min_size', type=int, default=450, help='Minimum image size')
        parser.add_argument('--max_size', type=int, default=550, help='Maximum image size')
        parser.add_argument('--jpeg96', action='store_true', help='Enable JPEG96 compression filtering')
        parser.add_argument('--use_fft', action='store_true')
        parser.add_argument('--use_wavelet', action='store_true')

        # pre-transform for create_dataset:
        parser.add_argument('--jpeg_qf', type=int, default=None,
                            help='if set, all images are jpeg compressed with this quality factor')
        parser.add_argument('--sample_qf_ai', action='store_true', default=False,
                            help='If this is set and jpeg_qf is None, the ai qf is sampled from the distribution of the qf from all natural train images')
        parser.add_argument('--resize', type=int, default=None,
                            help='if set, all images are first resized to this')
        parser.add_argument('--cropsize', type=int, default=None,
                            help='if set, all images are cropped to this size arfter resizing')
        parser.add_argument('--cropmethod', type=str, choices="['center', 'random']", default='center')
        parser.add_argument('--compress_natural', action='store_true', default=False,
                            help=' Whether to also compress the natural images with the given jpeg qf')

        self.isTrain = True
        return parser
