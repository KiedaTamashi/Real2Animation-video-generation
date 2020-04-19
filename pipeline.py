from components import *
from data_prepare.video2vmd import video2keypoints
from data_prepare.smooth_pose import smooth_json_pose
from data_prepare.generate_pose_map_anime import compute_pose
from util import video2frames,json2npy,kps_Normalize_single,prepareForPoseTransfer,genVideoFromPoseTransfer
from test import genSingleImg
import util,os,numpy as np, pandas as pd
# from pytorch_lightning import Trainer
# from model import skeletonVAE
# from settings import args
# import logging,os

def inference(v_r_x,I_a,operating_dir=r"D:\work\pycharmproject\Real2Animation-video-generation\demo"):
    """
    if we have the whole model, given a real human video and anime character, how we get the result anime video?
    :param v_r_x:  path of real human video
    :param I_a:  path of Image
    :return:
    """
    kps_dir = os.path.join(operating_dir,"json_out")
    kps_npy_dir = os.path.join(operating_dir,"testK_ori")
    frame_dir = os.path.join(operating_dir,"test_ori")
    frameN_dir = os.path.join(operating_dir, "test")
    reference_dir = os.path.join(operating_dir, "reference") # put Anime_Image.jpg in it.
    kps_a = os.path.join(operating_dir, "tmpK","animeImage.jpg.npy") #TODO
    kps_norm_dir = os.path.join(operating_dir, "normK")
    kps_final_dir = os.path.join(operating_dir, "testK")
    for fdir in [kps_dir,kps_npy_dir,frame_dir,frameN_dir,kps_norm_dir,kps_final_dir]:
        if not os.path.exists(fdir):
            os.mkdir(fdir)
    video2keypoints(v_r_x,kps_dir,display=1)
    smooth_json_pose(kps_dir,window_length=11,polyorder=3)
    json2npy(os.path.basename(v_r_x)[:-4],kps_dir,kps_npy_dir)
    video2frames(v_r_x,frame_dir)
    genSingleImg(reference_dir)

    kps_Normalize_single(frame_dir,frameN_dir,kps_npy_dir,kps_a,kps_norm_dir,reference_dir,
                         vis=True,real_bone_num=19)
    compute_pose(kps_norm_dir,kps_final_dir)
    prepareForPoseTransfer(frameN_dir,kps_final_dir,I_a)

def postProcess():
    genVideoFromPoseTransfer(r"D:\work\pycharmproject\Real2Animation-video-generation\demo2\results\anime_PATN\test_680\images")

if __name__ == '__main__':
    # # video = "D:\work\OpenMMD1.0\examples\pose_test.mp4"

    # video = r"D:\work\pycharmproject\Real2Animation-video-generation\demo2\pose_test.mp4"
    # I_a = r"D:\work\pycharmproject\Real2Animation-video-generation\demo2\reference\animeImage.jpg"
    # opt_dir = r"D:\work\pycharmproject\Real2Animation-video-generation\demo2"
    # inference(video,I_a,operating_dir=opt_dir)
    postProcess()

# def skeleton_learning():
#     skeleton_model = skeletonVAE(args)  # block = Basicblock
#     trainer = Trainer(max_epochs=args.epochs, gpus=0,logger=True,val_check_interval=5) #TODO if the args have None, then logger cannot work
#     trainer.fit(skeleton_model)
#     # view tensorboard logs
#     logging.info(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}')
#     logging.info('and going to http://localhost:6006 on your browser')
'''
def GAN_pipeline():
    def inf_train_gen():
        while True:
            for images,targets in train_gen():
                yield images
    train_gen, dev_gen, test_gen = lib.mnist.load(args.batch_size, args.batch_size)
    data = inf_train_gen()

    # instantialize model
    netG = Generator()
    netD = Discriminator()
    print(netG)
    print(netD)


    # set optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

    one,mone = utils.generate_one_mone()

    # set GPU for items
    if use_cuda:
        netD = netD.cuda(args.gpu)
        netG = netG.cuda(args.gpu)
        one = one.cuda(args.gpu)
        mone = mone.cuda(args.gpu)


    # Activate pipeline
    for iteration in range(args.iters):
        start_time = time.time()
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for iter_d in range(args.critic_iters):
            _data = next(data)
            real_data = torch.Tensor(_data)
            if use_cuda:
                real_data = real_data.cuda(args.gpu)
            real_data_v = autograd.Variable(real_data)

            netD.zero_grad()

            # train with real
            D_real = netD(real_data_v)
            D_real = D_real.mean()
            # print D_real
            D_real.backward(mone)

            # train with fake
            noise = torch.randn(args.batch_size, 128)
            if use_cuda:
                noise = noise.cuda(args.gpu)
            noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
            fake = autograd.Variable(netG(noisev).data)
            inputv = fake
            D_fake = netD(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)
            #TODO change the loss f
            # train with gradient penalty WGAN part
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()

        noise = torch.randn(args.batch_size, 128)
        if use_cuda:
            noise = noise.cuda(args.gpu)
        noisev = autograd.Variable(noise)
        fake = netG(noisev)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()
        # TODO change the loss f
        # Write logs and save samples
        lib.plot.plot('tmp/mnist/time', time.time() - start_time)
        lib.plot.plot('tmp/mnist/train disc cost', D_cost.cpu().data.numpy())
        lib.plot.plot('tmp/mnist/train gen cost', G_cost.cpu().data.numpy())
        lib.plot.plot('tmp/mnist/wasserstein distance', Wasserstein_D.cpu().data.numpy())

        # Validation part
        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                imgs = torch.Tensor(images)
                if use_cuda:
                    imgs = imgs.cuda(args.gpu)
                imgs_v = autograd.Variable(imgs, volatile=True)

                D = netD(imgs_v)
                _dev_disc_cost = -D.mean().cpu().data.numpy()
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('tmp/mnist/dev disc cost', np.mean(dev_disc_costs))
            # test image
            generate_image(iteration, netG)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99):
            lib.plot.flush()

        lib.plot.tick()
'''