#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import csv
import glob
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models, set_fast_norm
from timm.data import create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser,\
    decay_batch_step, check_batch_size_retry

from sklearn.metrics import roc_auc_score

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

from get_data import get_data
from get_transform import get_transform
from create_dataset import create_dataset

#hinzugefügt
import numpy as np

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
group.add_argument('dataset', type=str,
                    help='the dataset defines which dataset is used in the dataloader -> One of classic, jpeg96, controlled or define new')
group.add_argument('csv_data_path', type=str,
                    help='path to csv file')
group.add_argument('--base_path', type=str,
                    help='The base path is the folder, where the data is stored. this path gets prepended to the path in the csv')
group.add_argument('--generator', type=str, default=None, choices=['Midjourney', 'stable_diffusion_v_1_5', 'stable_diffusion_v_1_4', 'wukong', 'ADM', 'VQDM', 'glide', 'BigGAN'],
                    help='this is the generator on which the detector is evaluated, so it defines the genimage subset')
group.add_argument('--generator_trained_on', type=str, default=None, choices=['Midjourney', 'stable_diffusion_v_1_5', 'stable_diffusion_v_1_4', 'wukong', 'ADM', 'VQDM', 'glide', 'BigGAN'],
                    help='This is used for datset == SIZE so that all genimage subsets are used, except the one trained on')
group.add_argument('--class-map', default='../../class_map.txt', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file')
group.add_argument('--balance_val_classes', action='store_true', default=False,
                    help='whether or not to balance val data so that the class distribution is equal in ai images and nature images \
                        (number of instances per imagenet class is same in ai images and nature images)')

#hinzugefügt: if dataset == WildRF
group.add_argument('--platform', type=str, default=None, choices=['facebook', 'twitter', 'reddit'],
                    help='this is the platform on which the images where collected to form the dataset on which the detector is evaluated, so it defines the genimage subset')
                        
# If dataset == "SIZE":
group.add_argument('--min_width', type=int, default=None,
                    help='Only nature images in intervall [min_width:max_width, min_height:max_height] are included')
group.add_argument('--max_width', type=int, default=None,
                    help='Only nature images in intervall [min_width:max_width, min_height:max_height] are included')
group.add_argument('--min_height', type=int, default=None,
                    help='Only nature images in intervall [min_width:max_width, min_height:max_height] are included')
group.add_argument('--max_height', type=int, default=None,
                    help='Only nature images in intervall [min_width:max_width, min_height:max_height] are included')
group.add_argument('--min_qf', type=int, default=None,
                    help='Only nature images with qf > min_qf are included')
# pre-transform for create_dataset:
group.add_argument('--jpeg_qf', type=int, default=None,
                    help='if set, all images are jpeg compressed with this quality factor')
group.add_argument('--sample_qf_ai', action='store_true', default=False,
                    help='If this is set and jpeg_qf is None, the ai qf is sampled from the distribution of the qf from all natural train images')
group.add_argument('--train_df', type=str, default=None,
                    help='if --sample_qf_ai, we need the train dataset to get the qf distribution. This is the csv path to the train df')              
group.add_argument('--resize', type=int, default=None,
                    help='if set, all images are first resized to this')
group.add_argument('--cropsize', type=int, default=None,
                    help='if set, all images are cropped to this size arfter resizing')
group.add_argument('--cropmethod', type=str, choices="['center', 'random']", default='center')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
group.add_argument('--compress_natural', action='store_true', default=False,
                    help=' Whether to also compress the natural images with the given jpeg qf')

parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')

def load_class_map(map_or_filename, root=''):
    if isinstance(map_or_filename, dict):
        assert dict, 'class_map dict must be non-empty'
        return map_or_filename
    class_map_path = map_or_filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, class_map_path)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % map_or_filename
    class_map_ext = os.path.splitext(map_or_filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    elif class_map_ext == '.pkl':
        with open(class_map_path,'rb') as f:
            class_to_idx = pickle.load(f)
    else:
        assert False, f'Unsupported class map file extension ({class_map_ext}).'
    return class_to_idx

def binary_confusion_matrix(pred, target, args):
    """returns the confusion matrix of the binary classification task"""
    class_map = load_class_map(args.class_map)
    ai, nature = class_map["ai"], class_map["nature"]
    pred = pred.argmax(axis=1)
    pred = pred.round()
    target = target.round()

    result = torch.zeros(2, 2)
    result[0, 0] = torch.sum((pred == target) * (pred == ai))
    result[1, 1] = torch.sum((pred == target) * (pred == nature))
    result[0, 1] = torch.sum((pred != target) * (pred == nature))
    result[1, 0] = torch.sum((pred != target) * (pred == ai))
    
    return result

def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)
    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    data_val = get_data(args, is_training=False, balance_val_classes=args.balance_val_classes, balance_train_classes=False)
    n = len(data_val)
    if args.dataset.lower() == "size":
        print(f"\n there are {n} images in the intervall \n")
        if n == 0:
            print("===> EXIT")
            results = OrderedDict(
                                    model=args.model,
                                    top1=None, top1_err=None,
                                    top5=None, top5_err=None,
                                    param_count=round(param_count / 1e6, 2),
                                    img_size=data_config['input_size'][-1],
                                    crop_pct= 1.0 if test_time_pool else data_config['crop_pct'],
                                    interpolation=data_config['interpolation'],
                                    tpr=None,
                                    fpr=None,
                                    fnr=None,
                                    tnr=None,
                                    AUC=None,
                                    n=n,)
            return results

    train_df = None
    if args.sample_qf_ai:
        assert args.train_df
        train_df = pd.read_csv(args.train_df)

    pre_transform = get_transform(args, train_df)
    dataset = create_dataset(data_val, pre_transform, args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        end = time.time()
        total_confusion_matrix = torch.zeros(2, 2)
        AUC = None

        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)

            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            bcm = binary_confusion_matrix(output.detach(), target, args)
            total_confusion_matrix += bcm

            all_preds.append(output.detach()[:, 1])
            all_labels.append(target)

            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))
        total_confusion_matrix = total_confusion_matrix / total_confusion_matrix.sum()
        all_preds, all_labels = torch.cat(all_preds).cpu().numpy().reshape(-1), torch.cat(all_labels).cpu().numpy()

        try:
            AUC = roc_auc_score(all_labels, all_preds)
        except ValueError as e: 
            if "Only one class present in y_true. ROC AUC score is not defined in that case." in str(e):
                AUC = None
            else:
                raise e
        print(f"\n{total_confusion_matrix}\n")
    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        crop_pct=crop_pct,
        interpolation=data_config['interpolation'],
        tpr = total_confusion_matrix[0, 0].item(),
        fpr = total_confusion_matrix[1, 0].item(),
        fnr = total_confusion_matrix[0, 1].item(),
        tnr = total_confusion_matrix[1, 1].item(),
        auc = AUC,
        n=n,)

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = 'Unknown'
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            torch.cuda.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f'Reducing batch size to {batch_size} for retry.')
    results['error'] = error_str
    _logger.error(f'{args.model} failed to validate ({error_str}).')
    return results


def main():
    setup_default_logging()
    args = parser.parse_args()

    # Check if args are correct:
    if args.dataset.lower() == "SIZE".lower():
        assert args.min_width and args.max_width and args.min_height and args.max_height and args.generator_trained_on
    #hinzugefügt
    if args.dataset.lower() not in ["SIZE".lower(), "FFHQ_JPEG".lower(), 'wildrf']:
    #if args.dataset.lower() not in ["SIZE".lower(), "FFHQ_JPEG".lower()]:
        assert args.generator
    
    

    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*_in21k', '*_in22k', '*_dino'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if 'error' in r:
                    continue
                if args.checkpoint:
                    r['checkpoint'] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            results = validate(args)
    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')
    if args.results_file:
        import pandas as pd
        if not os.path.exists(os.path.dirname(args.results_file)):
            os.makedirs(os.path.dirname(args.results_file))
        df = pd.DataFrame([results]).T
        
        df.to_csv(args.results_file, header=False)

def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()