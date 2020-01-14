from components import *
import utils

# Dataset iterator
# TODO wait for rewrite, we may need a complex datasets, make it a class
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