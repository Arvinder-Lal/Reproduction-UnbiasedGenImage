from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        #parser.add_argument('--dataroot')
        parser.add_argument('--model_path')
        parser.add_argument('--no_resize', action='store_true')
        parser.add_argument('--no_crop', action='store_true')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
        parser.add_argument('--niter', type=int, default=0, help='# of iter at starting learning rate')
        parser.add_argument('--is_aug', action='store_true')

        #hinzugefügt
        parser.add_argument('--size_constrained', action='store_true', help='Enable size constraint')
        parser.add_argument('--min_size', type=int, default=450, help='Minimum image size')
        parser.add_argument('--max_size', type=int, default=550, help='Maximum image size')
        parser.add_argument('--jpeg96', action='store_true', help='Enable JPEG96 compression filtering')

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

        #hinzugefügt
        parser.add_argument('--result_path', type=str, default=None,help='Output csv file for validation results (summary)')
        parser.add_argument('--generator_trained_on', type=str, default=None,
                           choices=['Midjourney', 'stable_diffusion_v_1_5', 'stable_diffusion_v_1_4', 'wukong', 'ADM',
                                    'VQDM', 'glide', 'BigGAN'],
                           help='This is used for datset == SIZE so that all genimage subsets are used, except the one trained on')
        # hinzugefügt (If dataset == "SIZE"):
        parser.add_argument('--min_width', type=int, default=None,
                           help='Only nature images in intervall [min_width:max_width, min_height:max_height] are included')
        parser.add_argument('--max_width', type=int, default=None,
                           help='Only nature images in intervall [min_width:max_width, min_height:max_height] are included')
        parser.add_argument('--min_height', type=int, default=None,
                           help='Only nature images in intervall [min_width:max_width, min_height:max_height] are included')
        parser.add_argument('--max_height', type=int, default=None,
                           help='Only nature images in intervall [min_width:max_width, min_height:max_height] are included')
        parser.add_argument('--min_qf', type=int, default=None,
                           help='Only nature images with qf > min_qf are included')

        self.isTrain = False
        return parser
