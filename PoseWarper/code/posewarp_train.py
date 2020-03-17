import tensorflow as tf
import os
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import truncated_vgg
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
import logging
from logging import handlers

def train(model_name, gpu_id):
    params = param.get_general_params()

    network_dir = params['model_save_dir'] + '/' + model_name

    # add logger
    logger = logging.getLogger("logger.log")
    log_file_path = os.path.join(network_dir,"logger.log")
    fh = logging.handlers.TimedRotatingFileHandler(
        filename=log_file_path,
        backupCount=0,
        encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    if not os.path.isdir(network_dir):
        os.mkdir(network_dir)

    train_feed = data_generation.create_feed(params, params['data_dir'], 'train')

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat(params['data_dir']+'/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)
    if not params['load_weights'] == None:
        model.load_weights(params['load_weights'])

    model.compile(optimizer=Adam(lr=1e-4), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])

    #model.summary()
    n_iters = params['n_training_iter']

    if params['load_weights'] == None:
        start = 0
    else:
        start = int(params['load_weights'].split("/")[-1][:-3])

    for step in range(start, n_iters):
        x, y = next(train_feed)

        train_loss = model.train_on_batch(x, y)

        util.printProgress(step, 0, train_loss)
        logger.debug(str(step)+":"+str(train_loss))

        if step > 0 and step % params['model_save_interval'] == 0:
            model.save(network_dir + '/' + str(step) + '.h5')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need model name and gpu id as command line arguments.")
    else:
        train(sys.argv[1], sys.argv[2])
    # train('anime', 0)
