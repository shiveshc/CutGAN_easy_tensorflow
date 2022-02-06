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
        curr_img = curr_img.astype(np.float32)/255.0
        curr_img = cv2.resize(curr_img, dsize= (256, 256))
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
    _, l3, l6, l9, l12 = gen.conv_net(inp, inp.shape[3])

    return [l3, l6, l9, l12]

def get_patch_features(inp, gen):
    layer_feat = get_layer_features(inp, gen)
    inp_w = inp._shape_as_list()[2]
    inp_h = inp._shape_as_list()[1]

    patch_size = 64
    stride = 32
    all_patch = {}
    all_patch_id = {}

    x0 = 0
    x1 = x0 + patch_size
    while x1 <= inp_w:
        y0 = 0
        y1 = y0 + patch_size
        while y1 <= inp_h:
            for i, lfeat in enumerate(layer_feat):
                l_w = lfeat._shape_as_list()[2]
                l_h = lfeat._shape_as_list()[1]
                x0_lfeat = round(x0/inp_w*l_w)
                x1_lfeat = round(x1/inp_w*l_w)
                y0_lfeat = round(y0/inp_h*l_h)
                y1_lfeat = round(y1/inp_h*l_h)
                curr_l_patch = lfeat[:, y0_lfeat:y1_lfeat, x0_lfeat:x1_lfeat, :]
                if i in all_patch.keys():
                    all_patch[i].append(curr_l_patch)
                    last_id = all_patch_id[i][-1]
                    all_patch_id[i].append(last_id + 1)
                else:
                    all_patch[i] = [curr_l_patch]
                    all_patch_id[i] = [0]

            y0 = y0 + stride
            y1 = y0 + patch_size
        x0 = x0 + stride
        x1 = x0 + patch_size

    return all_patch, all_patch_id

def get_ce_loss(z, z_pos, z_neg):
    v_vpos = tf.matmul(z, tf.transpose(z_pos))
    v_vneg = tf.matmul(z, tf.transpose(z_neg))
    logits = tf.concat([v_vpos, v_vneg], axis= 1)
    labels = tf.concat([tf.ones_like(v_vpos), tf.zeros_like(v_vneg)], axis= 1)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= labels, logits= logits))

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

    # x_dir = 'D:/Shivesh/CUTGan/data/afhq/train/cat'
    # t_dir = 'D:/Shivesh/CUTGan/data/afhq/train/dog'
    x_dir = '/storage/coda1/p-hl94/0/schaudhary9/testflight_data/CUTGan/data/afhq/train/cat'
    t_dir = '/storage/coda1/p-hl94/0/schaudhary9/testflight_data/CUTGan/data/afhq/train/dog'
    x_imgs = load_image(x_dir)
    t_imgs = load_image(t_dir)

    train_X, test_X = split_train_test(x_imgs, 0.3)
    tsize = train_X.shape[0]
    train_T = t_imgs

    training_iters = 100
    batch_size = 1
    lr = 0.001

    x = tf.placeholder("float", [None, 256, 256, 3])
    t = tf.placeholder("float", [None, 256, 256, 3])

    with tf.variable_scope('gen', reuse= tf.AUTO_REUSE):
        x_hat, _, _, _, _ = generator.conv_net(x, x.shape[3])
        x_patch, x_patch_id = get_patch_features(x, generator)
        x_hat_patch, x_hat_patch_id = get_patch_features(x_hat, generator)

    with tf.variable_scope('disc', reuse= tf.AUTO_REUSE):
        x_hat_p = discriminator.conv_net(x_hat)
        t_p = discriminator.conv_net(t)

    cont_loss = 0
    total_patches = 0
    num_sample_ptaches = 16
    for l in x_hat_patch.keys():
        num_patches = len(x_hat_patch[l])
        sample_patches = random.sample(range(num_patches), num_sample_ptaches)
        total_patches = total_patches + num_sample_ptaches
        for i in sample_patches:
            v = x_hat_patch[l][i]

            v_pos = x_patch[l][i]
            v_neg_collec = [x_patch[l][k] for k in sample_patches if k!= i]
            v_neg = v_neg_collec[0]
            for k in range(1, len(v_neg_collec)):
                v_neg = tf.concat([v_neg, v_neg_collec[k]], axis= 0)

            with tf.variable_scope('mlp' + str(l), reuse= tf.AUTO_REUSE):
                z = mlp.net(v)
                z_pos = mlp.net(v_pos)
                z_neg = mlp.net(v_neg)

            cont_loss = cont_loss + get_ce_loss(z, z_pos, z_neg)

    cont_loss = cont_loss/total_patches

    fake_as_true = tf.ones_like(x_hat_p)
    gen_adv_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= fake_as_true, logits= x_hat_p))
    gen_loss = gen_adv_loss + cont_loss

    fake_as_fake = tf.zeros_like(x_hat_p)
    true_as_true = tf.ones_like(t_p)
    disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.concat([fake_as_fake, true_as_true], axis= 0), logits= tf.concat([x_hat_p, t_p], axis= 0)))

    disc_variables = [v for v in tf.trainable_variables() if 'disc' in v.name]
    disc_opt = tf.train.AdamOptimizer(learning_rate= lr).minimize(disc_loss, var_list= disc_variables)

    gen_variables = [v for v in tf.trainable_variables() if 'gen' in v.name]
    gen_opt = tf.train.AdamOptimizer(learning_rate= lr).minimize(gen_loss, var_list= gen_variables)

    mlp_variables = [v for v in tf.trainable_variables() if 'mlp' in v.name]
    mlp_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(cont_loss, var_list= mlp_variables)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # make folder where all results will be saved
    # results_dir = 'D:/Shivesh/CUTGan/Results/CUTgan_' + str(run) + '_' + str(tsize)
    results_dir = '/storage/coda1/p-hl94/0/schaudhary9/testflight_data/CUTGan/Results/CUTgan_train_v2_' + str(run) + '_' + str(tsize)
    if os.path.isdir(results_dir):
        shutil.rmtree(results_dir)

    os.mkdir(results_dir)

    with tf.Session() as sess:
        sess.run(init)
        summary_writer = tf.summary.FileWriter(results_dir, sess.graph)
        file = open(results_dir + '/training_loss.txt', 'a')

        for i in range(training_iters):
            if tsize > 1000:
                idx = random.sample(range(tsize), 1000)
                curr_batch_X = train_X[idx, :, :, :]
                curr_batch_T = train_T[idx, :, :, :]
            else:
                curr_batch_X = train_X
                curr_batch_T = train_T

            tic = time.clock()

            for k in range(5):

                # train discriminator
                for batch in range(50 // batch_size):
                    idx_x = random.sample(range(len(curr_batch_X)), batch_size)
                    idx_t = random.sample(range(len(curr_batch_T)), batch_size)
                    batch_x = curr_batch_X[idx_x[0:batch_size], :, :, :]
                    batch_t = curr_batch_T[idx_t[0:batch_size], :, :, :]
                    opt = sess.run(disc_opt, feed_dict={x: batch_x, t: batch_t})

                # train generator
                for batch in range(50 // batch_size):
                    idx_x = random.sample(range(len(curr_batch_X)), batch_size)
                    idx_t = random.sample(range(len(curr_batch_T)), batch_size)
                    batch_x = curr_batch_X[idx_x[0:batch_size], :, :, :]
                    batch_t = curr_batch_T[idx_t[0:batch_size], :, :, :]
                    opt = sess.run(gen_opt, feed_dict={x: batch_x, t: batch_t})

                # train mlp
                for batch in range(50 // batch_size):
                    idx_x = random.sample(range(len(curr_batch_X)), batch_size)
                    idx_t = random.sample(range(len(curr_batch_T)), batch_size)
                    batch_x = curr_batch_X[idx_x[0:batch_size], :, :, :]
                    batch_t = curr_batch_T[idx_t[0:batch_size], :, :, :]
                    opt = sess.run(mlp_opt, feed_dict={x: batch_x})
            toc = time.clock()

            # Calculate accuracy of 10 test images, repeat 10 times and report mean
            batch_test_cont_loss = []
            batch_test_gen_adv_loss = []
            batch_test_disc_loss = []
            for k in range(5):
                idx_x = random.sample(range(len(test_X)), 1)
                idx_t = random.sample(range(len(train_T)), 1)
                curr_cont_loss, curr_gen_adv_loss, curr_disc_loss = sess.run([cont_loss, gen_adv_loss, disc_loss], feed_dict={x: test_X[idx_x[:10], :, :, :], t: train_T[idx[:10], :, :, :]})
                batch_test_cont_loss.append(curr_cont_loss)
                batch_test_gen_adv_loss.append(curr_gen_adv_loss)
                batch_test_disc_loss.append(curr_disc_loss)

            batch_train_cont_loss = []
            batch_train_gen_adv_loss = []
            batch_train_disc_loss = []
            for k in range(5):
                idx_x = random.sample(range(len(train_X)), 1)
                idx_t = random.sample(range(len(train_T)), 1)
                curr_cont_loss, curr_gen_adv_loss, curr_disc_loss = sess.run([cont_loss, gen_adv_loss, disc_loss], feed_dict={x: train_X[idx_x[:10], :, :, :], t: train_T[idx[:10], :, :, :]})
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
            temp_idx = random.randint(0, test_X.shape[0])
            temp_X = test_X[temp_idx, :, :, :]
            temp_X = temp_X[np.newaxis, :, :, :]

            temp_pred = sess.run(x_hat, feed_dict={x: temp_X})
            cv2.imwrite(results_dir + '/X_' + str(temp_idx + 1) + '.png', temp_X[0, :, :, :].astype(np.uint8))  # this is the middle zplane corresponding to gt zplane
            cv2.imwrite(results_dir + '/pred_' + str(temp_idx + 1) + '.png', temp_pred[0, :, :, :].astype(np.uint8))

        # calculate accuracy on test data
        file = open(results_dir + '/test_data_loss.txt', 'a')
        idx_x = random.sample(range(len(train_X)), 100)
        idx_t = random.sample(range(len(train_T)), 100)
        for i in range(100):
            temp_X = test_X[idx_x[i], :, :, :]
            temp_T = train_T[idx_t[i], :, :, :]
            temp_X = temp_X[np.newaxis, :, :, :]
            temp_T = temp_T[np.newaxis, :, :, :]
            curr_cont_loss, curr_gen_adv_loss, curr_disc_loss = sess.run([cont_loss, gen_adv_loss, disc_loss], feed_dict={x: temp_X, t: temp_T})
            tic = time.clock()
            temp_pred = sess.run(x_hat, feed_dict={x: temp_X})
            toc = time.clock()
            file.write(str(i) + ',' + str(idx[i] + 1) + ',' + str(curr_cont_loss) + ',' + str(curr_gen_adv_loss) + ',' + str(curr_disc_loss) + ',' + str(toc - tic) + ',' + ',' + str(run) + ',' + str(tsize) + '\n')

        file.close()

        summary_writer.close()