from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2
import tensorflow as tf
import numpy




def get_image2(imgSrc: str):
    img = load_img(imgSrc, True)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150
    #x = x.reshape((1,) + x.shape)
    x = x.astype(float)
    x *= 1/255
    return x

image = get_image2('bon.png')

img_shape = image.shape
img_h, img_w, _ = img_shape

def sobel2(image):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    # Shape = height x width.
    #image = tf.placeholder(tf.float32, shape=[None, None])

    # Shape = 1 x height x width x 1.
    image_resized = tf.expand_dims(image, 0)

    Gx = tf.nn.conv2d(image_resized, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    Gy = tf.nn.conv2d(image_resized, sobel_y_filter,strides=[1, 1, 1, 1], padding='SAME')

    grad = tf.sqrt(tf.add(tf.pow(Gx,2),tf.pow(Gy,2)))
    grad = tf.truediv(grad,3.)

    grad = tf.reshape(grad, img_shape)

    return grad


input_img  = tf.placeholder(tf.float32, shape=img_shape, name="input_img")

gain   = tf.Variable(tf.constant(1, dtype=tf.float32, shape=img_shape), name="gain")
offset = tf.Variable(tf.constant(0, dtype=tf.float32, shape=img_shape), name="offset")

enhanced_img = tf.multiply(input_img, gain) + offset


#----------------------------------------------------------
# COST
#----------------------------------------------------------

input_img_deriv    = sobel2(input_img)
enhanced_img_deriv = sobel2(enhanced_img)

white_img      = tf.constant(1, dtype=tf.float32, shape=[img_h * img_w * 1])
flat_enh_img   = tf.reshape(enhanced_img, [-1])
flat_img_deriv = tf.reshape(input_img_deriv, [-1])

image_pixels_count = img_h * img_w

white_cost = tf.reduce_sum(tf.pow(flat_enh_img - white_img, 2))
sobel_cost = tf.reduce_sum(tf.pow(enhanced_img_deriv - input_img_deriv, 2))
cost = white_cost + sobel_cost


#----------------------------------------------------------
# TRAIN
#----------------------------------------------------------

# Parameters
learning_rate = 0.001
training_epochs = 100
display_step = 1

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

feed = {input_img: image }

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
# Start training
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict = feed)

        if (epoch+1) % display_step == 0:
            gen_img = sess.run(enhanced_img, feed_dict = feed)
            gen_img *= 255
            cv2.imwrite("output_2_{0}.png".format(epoch), gen_img)
            cv2.waitKey()
            #cv2.imshow("generated", gen_img)
            #cv2.waitKey()