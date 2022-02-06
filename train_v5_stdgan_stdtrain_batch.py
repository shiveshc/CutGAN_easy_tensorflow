import tensorflow as tf
from cnn_archs import *
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import shutil
import time
import argparse
import sys

def load_image(dir):
    x_img_list = os.listdir(dir)
    x_imgs = []
    for i in range(len(x_img_list)):
        curr_img = cv2.imread(dir + '/' + x_img_list[i], -1)
        curr_img = (curr_img.astype(np.float32) - 127.5)/127.5
        # curr_img = cv2.resize(curr_img, dsize= (256, 256))
        if len(curr_img.shape) < 3:
            curr_img = curr_img[:, :, np.newaxis]
            curr_img = np.repeat(curr_img, 3, axis= 2)
        x_imgs.append(curr_img)

    x_imgs = np.array(x_imgs)
    return x_imgs

def split_train_test(X, split_ratio):
    idx = [i for i in range(X.shape[0])]
    random.shuffle(idx)
    test_size = round(X.shape[0]*split_ratio)
    train_X = X[idx[:X.shape[0] - test_size], :, :, :]
    test_X = X[idx[X.shape[0] - test_size:], :, :, :]

    return train_X, test_X

def get_layer_features(inp, gen):
    _, layers = gen.conv_net(inp, inp.shape[3])

    return layers

def get_patch_features(x, x_hat, gen, num_samples):
    x_layer_feat = get_layer_features(x, gen)
    x_hat_layer_feat = get_layer_features(x_hat, gen)

    all_x_patch = {}
    all_x_hat_patch = {}
    l = 0
    for x_l, x_hat_l in zip(x_layer_feat, x_hat_layer_feat):
        random_sample = random.sample(range(x_l.shape[1] * x_l.shape[2]), num_samples)

        x_curr_feat = tf.reshape(x_l, [-1, x_l.shape[1]*x_l.shape[2], x_l.shape[3]])
        x_curr_patch = tf.gather(x_curr_feat, random_sample, axis= 1)
        all_x_patch[l] = x_curr_patch

        x_hat_curr_feat = tf.reshape(x_hat_l, [-1, x_hat_l.shape[1] * x_hat_l.shape[2], x_hat_l.shape[3]])
        x_hat_curr_patch = tf.gather(x_hat_curr_feat, random_sample, axis=1)
        all_x_hat_patch[l] = x_hat_curr_patch

        l = l + 1

    return all_x_patch, all_x_hat_patch


def get_ce_loss(z_x_hat, z_x, num_patches, batch_size):
    bs = tf.shape(z_x_hat)[0]
    logits = tf.matmul(z_x_hat, tf.transpose(z_x, (0, 2, 1)))
    labels = tf.tile(tf.expand_dims(tf.eye(num_patches), axis=0), [bs, 1, 1])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= labels, logits= logits), axis=0)

    return loss

def parse_argument(arg_list):
    if not arg_list:
        arg_list = ['-h']
        print('error - input required, see description below')

    parser = argparse.ArgumentParser(prog= 'train.py', description='CUTGan tensorflow implementation')
    parser.add_argument('run', type= int, help= 'run number to distinguish different runs')
    parser.add_argument('-tsize', help='data size to use for training')
    args = parser.parse_args(arg_list)
    return args.tsize, args.run

if __name__ == '__main__':
    tsize, run = parse_argument(sys.argv[1:])

    # x_dir = 'D:/Shivesh/CUTGan/data/horse2zebra/trainA'
    # t_dir = 'D:/Shivesh/CUTGan/data/horse2zebra/trainB'
    # x_dir = '/storage/coda1/p-hl94/0/schaudhary9/testflight_data/CUTGan/data/afhq/train/cat'
    # t_dir = '/storage/coda1/p-hl94/0/schaudhary9/testflight_data/CUTGan/data/afhq/train/dog'
    x_dir = '/storage/scratch1/0/schaudhary9/CUTGan/data/horse2zebra/trainA'
    t_dir = '/storage/scratch1/0/schaudhary9/CUTGan/data/horse2zebra/trainB'
    x_imgs = load_image(x_dir)
    t_imgs = load_image(t_dir)

    train_X, test_X = split_train_test(x_imgs, 0.3)
    tsize = train_X.shape[0]
    train_T = t_imgs

    training_iters = 200
    batch_size = 4
    lr = 0.0002

    x = tf.placeholder("float", [None, 256, 256, 3])
    t = tf.placeholder("float", [None, 256, 256, 3])

    with tf.variable_scope('gen', reuse= tf.AUTO_REUSE):
        x_hat, _ = generator_v2.conv_net(x, x.shape[3])
        x_patch, x_hat_patch = get_patch_features(x, x_hat, generator_v2, 256)

    with tf.variable_scope('disc', reuse= tf.AUTO_REUSE):
        x_hat_p = patch_discriminator.conv_net(x_hat)
        t_p = patch_discriminator.conv_net(t)

    cont_loss = []
    total_patches = 0
    for l in x_hat_patch.keys():
        num_patches = x_hat_patch[l].shape[1].value
        total_patches = total_patches + num_patches
        with tf.variable_scope('mlp' + str(l), reuse=tf.AUTO_REUSE):
            z_x_hat = mlp_v2.net(x_hat_patch[l])
            z_x = mlp_v2.net(x_patch[l])

        cont_loss.append(get_ce_loss(z_x_hat, z_x, num_patches, batch_size))

    f_cont_loss = tf.reduce_sum(tf.stack(cont_loss, axis= 0))/total_patches

    fake_as_true = tf.ones_like(x_hat_p)
    gen_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= fake_as_true, logits= x_hat_p))
    gen_loss = gen_adv_loss + f_cont_loss

    fake_as_fake = tf.zeros_like(x_hat_p)
    true_as_true = tf.ones_like(t_p)
    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.concat([fake_as_fake, true_as_true], axis= 0), logits= tf.concat([x_hat_p, t_p], axis= 0)))

    disc_variables = [v for v in tf.trainable_variables() if 'disc' in v.name]
    disc_opt = tf.train.AdamOptimizer(learning_rate= lr).minimize(disc_loss, var_list= disc_variables)

    gen_variables = [v for v in tf.trainable_variables() if 'gen' in v.name]
    gen_opt = tf.train.AdamOptimizer(learning_rate= lr).minimize(gen_loss, var_list= gen_variables)

    mlp_variables = [v for v in tf.trainable_variables() if 'mlp' in v.name]
    mlp_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(f_cont_loss, var_list= mlp_variables)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # make folder where all results will be saved
    # results_dir = 'D:/Shivesh/CUTGan/Results/CUTgan_train_v4_stdgan_newtrain_' + str(run) + '_' + str(tsize)
    results_dir = '/storage/scratch1/0/schaudhary9/CUTGan/Results/CUTgan_train_v5_stdgan_stdtrain_batch' + str(run) + '_' + str(tsize)
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)

    os.mkdir(results_dir)

    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(results_dir, sess.graph)
        file = open(results_dir + '/training_loss.txt', 'a')

        for i in range(training_iters):
            if tsize > 500:
                idx = random.sample(range(tsize), 400)
                curr_batch_X = train_X[idx, :, :, :]
                curr_batch_T = train_T[idx, :, :, :]
            else:
                curr_batch_X = train_X
                curr_batch_T = train_T

            tic = time.clock()

            # train discriminator, generator, mlp
            for batch in range(len(curr_batch_X)//batch_size):
                idx_x = random.sample(range(len(curr_batch_X)), batch_size)
                idx_t = random.sample(range(len(curr_batch_T)), batch_size)
                batch_x = curr_batch_X[idx_x[0:batch_size], :, :, :]
                batch_t = curr_batch_T[idx_t[0:batch_size], :, :, :]
                opt, c2 = sess.run([gen_opt, gen_loss], feed_dict={x: batch_x, t: batch_t})
                opt, c1 = sess.run([disc_opt, disc_loss], feed_dict={x: batch_x, t: batch_t})
                opt, c3 = sess.run([mlp_opt, f_cont_loss], feed_dict={x: batch_x})

            # # train generator
            # for batch in range(len(curr_batch_X) // batch_size):
            #     idx_x = random.sample(range(len(curr_batch_X)), batch_size)
            #     idx_t = random.sample(range(len(curr_batch_T)), batch_size)
            #     batch_x = curr_batch_X[idx_x[0:batch_size], :, :, :]
            #     batch_t = curr_batch_T[idx_t[0:batch_size], :, :, :]
            #     opt = sess.run(gen_opt, feed_dict={x: batch_x, t: batch_t})

            # # train mlp
            # for batch in range(len(curr_batch_X) // batch_size):
            #     idx_x = random.sample(range(len(curr_batch_X)), batch_size)
            #     idx_t = random.sample(range(len(curr_batch_T)), batch_size)
            #     batch_x = curr_batch_X[idx_x[0:batch_size], :, :, :]
            #     batch_t = curr_batch_T[idx_t[0:batch_size], :, :, :]
            #     opt = sess.run(mlp_opt, feed_dict={x: batch_x})
            toc = time.clock()

            # Calculate accuracy of 10 test images, repeat 10 times and report mean
            batch_test_cont_loss = []
            batch_test_gen_adv_loss = []
            batch_test_disc_loss = []
            for k in range(20):
                idx_x = random.sample(range(len(test_X)), 1)
                idx_t = random.sample(range(len(train_T)), 1)
                curr_cont_loss, curr_gen_adv_loss, curr_disc_loss = sess.run([f_cont_loss, gen_adv_loss, disc_loss], feed_dict={x: test_X[idx_x[:1], :, :, :], t: train_T[idx[:1], :, :, :]})
                batch_test_cont_loss.append(curr_cont_loss)
                batch_test_gen_adv_loss.append(curr_gen_adv_loss)
                batch_test_disc_loss.append(curr_disc_loss)

            batch_train_cont_loss = []
            batch_train_gen_adv_loss = []
            batch_train_disc_loss = []
            for k in range(20):
                idx_x = random.sample(range(len(train_X)), 1)
                idx_t = random.sample(range(len(train_T)), 1)
                curr_cont_loss, curr_gen_adv_loss, curr_disc_loss = sess.run([f_cont_loss, gen_adv_loss, disc_loss], feed_dict={x: train_X[idx_x[:1], :, :, :], t: train_T[idx[:1], :, :, :]})
                batch_train_cont_loss.append(curr_cont_loss)
                batch_train_gen_adv_loss.append(curr_gen_adv_loss)
                batch_train_disc_loss.append(curr_disc_loss)

            mean_train_cont_loss = sum(batch_train_cont_loss) / len(batch_train_cont_loss)
            mean_train_gen_adv_loss = sum(batch_train_gen_adv_loss) / len(batch_train_gen_adv_loss)
            mean_train_disc_loss = sum(batch_train_disc_loss) / len(batch_train_disc_loss)
            mean_test_cont_loss = sum(batch_test_cont_loss) / len(batch_test_cont_loss)
            mean_test_gen_adv_loss = sum(batch_test_gen_adv_loss) / len(batch_test_gen_adv_loss)
            mean_test_disc_loss = sum(batch_test_disc_loss) / len(batch_test_disc_loss)
            print("Iter " + str(i) + ", ::::Train Loss, " + "cont_loss:" + "{:.6f}".format(mean_train_cont_loss) + " adv_loss:" + "{:.6f}".format(mean_train_gen_adv_loss) + " disc_loss:" + "{:.6f}".format(mean_train_disc_loss) +
                  " ::::Test Loss, " + "cont_loss:" + "{:.6f}".format(mean_test_cont_loss) + " adv_loss:" + "{:.6f}".format(mean_test_gen_adv_loss) + " disc_loss:" + "{:.6f}".format(mean_test_disc_loss))
            file.write(str(i) + ',' + str(mean_train_cont_loss) + ',' + str(mean_train_gen_adv_loss) + ',' + str(mean_train_disc_loss) + ',' +
                       str(mean_test_cont_loss) + ',' + str(mean_test_gen_adv_loss) + ',' + str(mean_test_disc_loss) + ',' +
                       str(toc - tic) + ',' + str(run) + ',' + str(tsize) + '\n')

        file.close()

        # save final model
        saver.save(sess, results_dir + '/model')

        # # save some random prediction examples
        for i in range(10):
            temp_idx = random.randint(0, test_X.shape[0] - 1)
            temp_X = test_X[temp_idx, :, :, :]
            temp_X = temp_X[np.newaxis, :, :, :]

            temp_pred = sess.run(x_hat, feed_dict={x: temp_X})
            cv2.imwrite(results_dir + '/X_' + str(temp_idx + 1) + '.png', temp_X[0, :, :, :] * 127.5 + 127.5)  # this is the middle zplane corresponding to gt zplane
            cv2.imwrite(results_dir + '/pred_' + str(temp_idx + 1) + '.png', temp_pred[0, :, :, :] * 127.5 + 127.5)

        # calculate accuracy on test data
        file = open(results_dir + '/test_data_loss.txt', 'a')
        idx_x = random.sample(range(len(test_X)), 100)
        idx_t = random.sample(range(len(train_T)), 100)
        for i in range(100):
            temp_X = test_X[idx_x[i], :, :, :]
            temp_T = train_T[idx_t[i], :, :, :]
            temp_X = temp_X[np.newaxis, :, :, :]
            temp_T = temp_T[np.newaxis, :, :, :]
            curr_cont_loss, curr_gen_adv_loss, curr_disc_loss = sess.run([f_cont_loss, gen_adv_loss, disc_loss], feed_dict={x: temp_X, t: temp_T})
            tic = time.clock()
            temp_pred = sess.run(x_hat, feed_dict={x: temp_X})
            toc = time.clock()
            file.write(str(i) + ',' + str(idx[i] + 1) + ',' + str(curr_cont_loss) + ',' + str(curr_gen_adv_loss) + ',' + str(curr_disc_loss) + ',' + str(toc - tic) + ',' + ',' + str(run) + ',' + str(tsize) + '\n')

        file.close()

        summary_writer.close()