import time
import os
from ADGAN.options.test_options import TestOptions
from ADGAN.data.data_loader import CreateDataLoader
from ADGAN.models.models import create_model
from ADGAN.util.visualizer import Visualizer
from ADGAN.util import html
from ADGAN.util import util
import time

def run():
#python ../test.py --dataroot data --dirSem data --pairLst data/fashion-resize-pairs-test.csv --checkpoints_dir checkpoints --results_dir results 
    #--name fashion_AdaGen_sty512_nres8_lre3_SS_fc_vgg_cxloss_ss_merge3 --model adgan --phase test --dataset_mode keypoint 
    #--norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0,1 --BP_input_nc 18 --no_flip --which_model_netG ADGen --which_epoch 800
#python test.py --dataroot your_path/deepfashion/fashion_resize --dirSem your_path/deepfashion --pairLst your_path/deepfashion/fashion-resize-pairs-test.csv --checkpoints_dir ./checkpoints --results_dir ./results --name fashion_AdaGen_sty512_nres8_lre3_SS_fc_vgg_cxloss_ss_merge3 --model adgan --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0,1 --BP_input_nc 18 --no_flip --which_model_netG ADGen --which_epoch 800

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.dataroot = './deepfashion/fashion_resize'
    opt.dirSem = './deepfashion'
    opt.pairLst = './deepfashion/fashion-resize-pairs-test.csv'
    opt.checkpoints_dir = './checkpoints'
    opt.results_dir = './results'
    opt.name = 'fashion_AdaGen_sty512_nres8_lre3_SS_fc_vgg_cxloss_ss_merge3'
    opt.model = 'adgan'
    opt.phase = 'test'
    opt.dataset_mode = 'keypoint'
    opt.norm = 'instance'
    opt.batchSize = 1
    opt.resize_or_crop = 'no'
    opt.gpu_ids = 0,
    opt.BP_input_nc = 18
    opt.SP_input_nc = 8
    opt.which_model_netG = 'ADGen'
    opt.which_epoch = 800
    opt.how_many = 10

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    printArgs(opt)

    model = model.eval()

    print('dataset:', len(dataset), ', how_many:', opt.how_many, 'model.training:', model.training)
    # test
    for i, data in enumerate(dataset):
        print(' process {}/{} img ..'.format(i,len(dataset)))

        model.set_input(data)
        startTime = time.time()
        model.test()
        endTime = time.time()
        print(endTime-startTime)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        img_path = [img_path]
        print(img_path)
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()

def printArgs(opt):
    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')