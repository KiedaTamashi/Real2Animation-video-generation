import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
import os
import scipy.io as sio
import numpy as np
import sys
sys.path.append('../')
import data_generation
import networks
import param
import cv2
import truncated_vgg

def recover2img(img):
    img = (img / 2.0 + 0.5) * 255.0
    return img


def evaluate(model_name,gpu_id):
    params = param.get_general_params()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat(params['data_dir']+'/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)

    model.compile(optimizer=Adam(), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])
    iterations = range(1000, 185001, 1000)

    n_batches = 25
    losses = []
    for i in iterations:
        print(i)
        model.load_weights('../models/' + model_name+'/'+str(i) + '.h5')
        np.random.seed(11)
        feed = data_generation.create_feed(params, params['data_dir'], 'train')
        loss = 0
        for batch in range(n_batches):
            x, y = next(feed)
            loss += model.evaluate(x, y)
        loss /= (n_batches*1.0)
        losses.append(loss)
        sio.savemat('losses_by_iter.mat', {'losses': losses, 'iterations': iterations})

def predict(model_name,gpu_id,save_file_name):
    params = param.get_general_params()
    network_dir = params['model_save_dir'] + '/' + model_name
    save_dir = params['model_save_dir'] + '/' + model_name + '/result'
    params['batch_size'] = 1
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat(params['data_dir']+'/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)

    # model.compile(optimizer=Adam(), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])

    model.load_weights(network_dir+save_file_name)  # TODO not sure the final ckpt name
    np.random.seed(112)
    feed = data_generation.create_feed(params, params['data_dir'], 'train',do_augment=False)
    cnt = 8
    while True:
        try:
            x, y = next(feed)
            inp = recover2img(x[0])
            cv2.imwrite(os.path.join(save_dir, str(cnt) + "inp.jpg"), inp[0])
            # cv2.imwrite(os.path.join(save_dir, str(cnt) + "map.jpg",x[2][0][:,:,0]))

            out = model.predict(x)
            out = recover2img(out[0])
            cv2.imwrite(os.path.join(save_dir,str(cnt)+".jpg"),out)
            gt = recover2img(y[0])
            cv2.imwrite(os.path.join(save_dir,str(cnt)+"gt.jpg"),gt)
            cnt += 1
            break
        except:
            break


if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Need model name and gpu id as command line arguments.")
    # else:
    #     evaluate(sys.argv[1], sys.argv[2])
    predict('', 0,'gan5000.h5')