import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import skimage.draw as draw
import skimage.io as io
import skimage.color as color
import skimage.transform as transform
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import imageio
import csv
width_multiplier = 0.75
img_dim = 224
weight_multiplier = 1
def batch_norm(x):
  minn = 0
  ax = tf.transpose(x, [0,3,1,2])
  shapee = tf.shape(ax)
  ax = tf.reshape(ax,[shapee[0], shapee[1], -1])
  m = tf.reduce_max(ax, 2, keep_dims = True)
  j = tf.reduce_min(ax,2,keep_dims=True)
  ax = (ax-j)/(m - j)
  ax = tf.reshape(ax,shapee)
  ax = tf.transpose(ax, [0,2,3,1])
  return ax#(x-minn)/(maxx-minn)


weight_init = tf.contrib.layers.xavier_initializer()#tf.random_uniform_initializer(-weight_multiplier, weight_multiplier)
def depthwise_separable_conv(inputs,num_pwc_filters,width_multiplier,sc,downsample=False, batch_norm=True):
  
  num_pwc_filters = round(num_pwc_filters * width_multiplier)
  _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
  depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  scope=sc+'/depthwise_conv',
                                                  weights_initializer= weight_init)

  #bn = slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
  depthwise_conv = tf.nn.relu(depthwise_conv)
  if(batch_norm):
    depthwise_conv = tf.contrib.layers.batch_norm(depthwise_conv)
  pointwise_conv = slim.convolution2d(depthwise_conv,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        scope=sc+'/pointwise_conv',
                                        weights_initializer = weight_init)

  pointwise_conv = tf.nn.relu(pointwise_conv)
  if(batch_norm):
    pointwise_conv = tf.contrib.layers.batch_norm(pointwise_conv)
  #bn = slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
  return pointwise_conv

def conv_layer(inputs, num_filters, width_multiplier, scope, downsample = False, filter_shape = [3,3]):
  
  '''stride = 1
  if(downsample):
    stride = 2
  return tf.nn.relu(tf.contrib.layers.batch_norm(tf.contrib.layers.conv2d(inputs, round(num_filters * width_multiplier), filter_shape, stride=stride, padding='SAME', scope=scope, weights_initializer= weight_init)))
  '''
  return depthwise_separable_conv(inputs, num_filters, width_multiplier, scope, downsample)


def get_bilinear_filter(filter_shape, upscale_factor = 2):
  #filter_shape is [width, height, num_in_channels, num_out_channels]
  kernel_size = filter_shape[1]
  if(kernel_size%2 == 1): #if kernal size is an odd number
    centre_location = upscale_factor - 1
  else:
    centre_location = upscale_factor - 0.5

  bilinear = np.zeros([filter_shape[0], filter_shape[1]])
  for x in range(filter_shape[0]):
    for y in range(filter_shape[1]):
      value = (1- abs((x-centre_location) / upscale_factor)) * (1- abs((y-centre_location) / upscale_factor))
      bilinear[x,y] = value

  weights = np.zeros(filter_shape)
  for i in range(filter_shape[3]):
    weights[:,:,i,i] = bilinear
  init = tf.constant_initializer(value = weights, dtype = tf.float32)
  bilinear_weights = tf.get_variable(name = "deconv_filter", initializer = init, shape = filter_shape)
  return bilinear_weights

def upsample_layer(x, n_channels, scope, upscale_factor = 2):
  kernel_size = 2*upscale_factor - upscale_factor%2
  stride = upscale_factor
  strides = [1,stride, stride, 1]
  with tf.variable_scope(scope):
    in_shape = tf.shape(x)

    h = in_shape[1]*2#((in_shape[1] - 1) * stride) + 1
    w = in_shape[2]*2#((in_shape[2] - 1) * stride) + 1
    new_shape = [in_shape[0], h, w, n_channels]
    new_shape = tf.stack(new_shape)

    filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
    weights = get_bilinear_filter(filter_shape, upscale_factor)

    deconv = tf.nn.conv2d_transpose(x, weights, new_shape, strides = strides, padding = 'SAME')

  return deconv
def filterwise_softmax(x):
  a = tf.transpose(x, [0,3,1,2])
  shapee = tf.shape(a)
  a = tf.reshape(a,[shapee[0], shapee[1], -1])
  a = tf.nn.softmax(a, dim = 2)
  a = tf.reshape(a, shapee)
  a = tf.transpose(a, [0,2,3,1])
  return a
#weights_inits= tf.random_uniform()

x = tf.placeholder(tf.float32, [None, img_dim, img_dim, 3])
y = tf.placeholder(tf.int8, [None, img_dim/2, img_dim/2, 13])
y = tf.cast(y, tf.float32)

net = conv_layer(x, 64, width_multiplier, 'conv1')
net = conv_layer(net, 64, width_multiplier, 'downsample_1', downsample = True)
net = conv_layer(net, 128, width_multiplier, 'conv2')
net_a = conv_layer(net, 128, width_multiplier, 'conv3')

net = conv_layer(net_a, 128, width_multiplier, 'downsample_2', downsample = True)
net = conv_layer(net, 256, width_multiplier, 'conv4')
net_b = conv_layer(net, 256, width_multiplier, 'conv5')

net = conv_layer(net_b, 256, width_multiplier, 'downsample_3', downsample = True)
net = conv_layer(net, 512, width_multiplier, 'conv6')
net_c = conv_layer(net, 512, width_multiplier, 'conv7')
net = conv_layer(net_c, 512, width_multiplier, 'downsample_4', downsample = True) #14x14
net = conv_layer(net, 512, width_multiplier, 'conv8')

net = conv_layer(net, 512, width_multiplier, 'conv9')
net = upsample_layer(net, round(512*width_multiplier), 'upsample_1')

net = conv_layer(net, 512, width_multiplier, 'conv10')
net = tf.concat([net, net_c], axis = 3)
net = conv_layer(net, 512, width_multiplier, 'conv11')

net = upsample_layer(net, round(512*width_multiplier), 'upsample_2')
net = conv_layer(net, 256, width_multiplier, 'conv12')
net = tf.concat([net, net_b], axis = 3)
net = conv_layer(net, 256, width_multiplier, 'conv13')

net = upsample_layer(net, round(256*width_multiplier), 'upsample_3')
net = conv_layer(net, 128, width_multiplier, 'conv14')
net = tf.concat([net, net_a], axis = 3)
#net = conv_layer(net, 128, width_multiplier, 'conv15')
#net = upsample_layer(net, round(128*width_multiplier), 'upsample_4')
net = conv_layer(net, 64, width_multiplier, 'conv16')
net_logits = tf.contrib.layers.conv2d(net, 13, [1,1], stride=1, padding='SAME', scope='conv17', activation_fn = None)
net= tf.sigmoid(net_logits)
#net = (tf.tanh(net_logits) + 1)/2
net = tf.clip_by_value(net,0, 1)



saver = tf.train.Saver(tf.trainable_variables())

def performance_test(sess):
  img = np.random.rand(10,224,224,3)
  for i in range(100):
    segmap = sess.run(net, {x:img})
    print(i)

def video_test(sess, vid):
  reader = imageio.get_reader(vid)
  index = 0
  img = None
  for image in reader.iter_data():
    if(index%30 == 0):
      im = transform.resize(image, [img_dim, img_dim])
      out = sess.run(net, {x: [im]})[0]
      out = np.sum(out, axis = 2)
      if(img is None):
        img = plt.imshow(out)
      else:
        img.set_data(out)
        #plt.scatter([55],[30])
      plt.pause(0.1)
      plt.draw()
    index+=1

def video_test_static(sess, vid,n, skip):
  reader = imageio.get_reader(vid)
  index = 0
  img = None
  frames = []
  for image in reader.iter_data():
    if(index%skip == 0):
      if(index/skip > n):
        break
      else:
        im = transform.resize(image, [img_dim, img_dim])
        frames.append(im)
    index += 1
  print('frames ready for feed')
  out = sess.run(net, {x:np.array(frames)}) #sess.run(net, {x:np.array(frames)})
  out = np.sum(out, axis = 3)
  for i in range(len(frames)):
    im = out[i]#np.sum(out[i], axis = 2)

    if(img is None):
      img = plt.imshow(im)
    else:
      img.set_data(im)

    plt.pause(0.1)
    plt.draw()

def video_test_save(sess, vid, n, skip,fname, fps = 10, batch_size =300):
  reader = imageio.get_reader(vid)
  index = 0
  img = None
  frames = []
  for image in reader.iter_data():
    if(index%skip == 0):
      if(index/skip > n):
        break
      else:
        im = transform.resize(image, [img_dim, img_dim])
        frames.append(im)
        sys.stdout.write('\r%i' % index)
    index += 1
  print('\n')
  batches = int(n/batch_size)
  writer = imageio.get_writer(fname, fps =fps)
  print('frames ready for feed')
  for k in range(batches):
    print('batch ' + str(k) + ' / ' + str(batches))
    out = sess.run(net, {x:np.array(frames)[k*batch_size: (k+1)*batch_size]/255})
    out = np.sum(out, axis = 3)
    out = (out/np.max(out) * 255).astype(np.uint8)

    for i in range(batch_size):
      writer.append_data(out[i])

  writer.close()

def img_test(sess, img):
  img = transform.resize(img, [img_dim, img_dim])
  img = np.reshape(img, [1,img_dim, img_dim, 3])
  segmaps = sess.run(net, {x:img})[0]
  #for i in range(13):
  plt.imshow(np.sum(segmaps, axis = 2))
  plt.show()
  for i in range(13):
    plt.imshow(segmaps[:,:,i])
    plt.show()

def img_plot_test(sess, img):
  img = transform.resize(img, [img_dim, img_dim])
  img = np.reshape(img, [1,img_dim, img_dim, 3])
  segmaps = sess.run(net, {x:img})[0]

  segmaps = np.reshape(segmaps, [-1,13])
  amax = np.argmax(segmaps, 0)

  xs = amax/(img_dim/2)
  ys = amax%(img_dim/2)

  plt.imshow(img[0])
  plt.scatter(ys*2, xs*2)
  plt.show()



with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  #performance_test(sess)
  
  saver.restore(sess, '/Users/michaelfernandez/Desktop/coco/0.75/params-10')
  img = io.imread('testdata/me2.png')
  img = img[:,:,:3]
  #img = transform.rotate(img,90)
  img_test(sess, img)
  