import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
def discriminator(images):
    # First convolutional and pool layers
    # This finds 32 different 5 x 5 pixel features

    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + d_b1  # 28*28*32
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  #14*14

    # Second convolutional and pool layers
    # This finds 64 different 5 x 5 pixel features
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + d_b2 #14*14*64
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #7x7x64

    # First fully connected layer
    d_w3 = tf.get_variable('d_w3', [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))

    d3 = tf.reshape(d2, [-1, 7 * 7 * 64]) ## transforms the batchx7x7x64 into a batchx(7*7*64)
    d3 = tf.matmul(d3, d_w3)
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)

    # Second fully connected layer
    d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
    d4 = tf.matmul(d3, d_w4) + d_b4
    #d4 is the latent space of the image

    # d4 contains unscaled values
    return d4

def generator(z, batch_size, z_dim):

    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    #3196 = 56*56
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    #the first linear layer turns the random noise z_dim into an image of 56*56
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME') #28x28*50
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [56, 56])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [56, 56])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    # Dimensions of g4: batch_size x 28 x 28 x 1
    return g4


    #tf.reset_default_graph()
with tf.variable_scope("scope") as scope:
    z_dimensions = 100
    batch_size = 50

    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder') 
# z_placeholder is for feeding input noise to the generator

    x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder') 
# x_placeholder is for feeding input images to the discriminator

    Gz = generator(z_placeholder, batch_size, z_dimensions) 
# Gz holds the generated images
    Dx = discriminator(x_placeholder)

# Dx will hold discriminator prediction probabilities
# for the real MNIST images
    scope.reuse_variables() #start reusing variables after initializing
    Dg = discriminator(Gz)

# Dg will hold discriminator prediction probabilities for generated images
#d_loss_real = 
mnist = input_data.read_data_sets("MNIST_data/")
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels =tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]
print([var.name for var in d_vars])
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
        # pretrain the discriminator
    for i in range(300):
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])#generate the random seeds

        real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])

        _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake], #calling trainers and losses
                                               {x_placeholder: real_image_batch, z_placeholder: z_batch})

        if(i % 100 == 0):
            print("dLossReal:", dLossReal, "dLossFake:", dLossFake)


    for i in range(100000):
        real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])

        # Train discriminator on both real and fake images
        _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                               {x_placeholder: real_image_batch, z_placeholder: z_batch})

        # Train generator
        z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
        sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

        if i % 100 == 0:
        # Every 100 iterations, show a generated image
            print("Iteration:", i, "at", datetime.datetime.now())
            z_batch = np.random.normal(0, 1, size=[1, z_dimensions])
            generated_images = generator(z_placeholder, 1, z_dimensions)
            images = sess.run(generated_images, {z_placeholder: z_batch})
            plt.imshow(images[0].reshape([28, 28]), cmap='Greys')
            plt.show()

            # Show discriminator's estimate
            im = images[0].reshape([1, 28, 28, 1])
            result = discriminator(x_placeholder)
            estimate = sess.run(result, {x_placeholder: im})
            print("Estimate:", estimate)


