from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2
import tensorflow as tf
import numpy



def get_images(imgSrc: str):
    image = cv2.imread(imgSrc)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray_image, cv2.CV_8U)

    laplacian = (255-laplacian)

    cv2.imwrite("laplacian.png", laplacian)
    #cv2.imshow("aa", laplacian)
    #cv2.waitKey()

    return image, laplacian


def get_image2(imgSrc: str):
    img = load_img(imgSrc)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)


img_gray, img_deriv = get_images("bon.png");

#
# 
#

def sobel_gradient(im):
    '''
    Args:
    im: Image to be differentiated [H,W,3]
    Returns:
    grad: Sobel gradient magnitude of input image [H,W,1]
    '''
    assert im.get_shape()[-1].value == 3
    print(im)
 
    Gx_kernel = tf.tile(
        tf.constant([[1,2,1],[0,0,0],[-1,-2,-1]],shape=[3,3,1],dtype=tf.float32)
        ,[1,1,3]
    )
    Gy_kernel = tf.tile(
        tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],shape=[3,3,1],dtype=tf.float32)
        ,[1,1,3]
    )

    #tf.transpose(Gx_kernel,[1,0,2,3])
 
    Gx = tf.nn.conv2d(im, Gx_kernel, [1,1,1], padding='SAME')
    Gy = tf.nn.conv2d(im, Gx_kernel, [1,1,1], padding='SAME')
 
    grad = tf.sqrt(tf.add(tf.pow(Gx,2),tf.pow(Gy,2)))
    grad = tf.truediv(grad,3.)
 
    return grad

def sobel2(im):
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])

    filtered_x = tf.nn.conv2d(image_resized, sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(image_resized, sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')


img_h, img_w, _ = img_gray.shape
img_shape = [img_h, img_w, 3]

#input_img_gray  = tf.placeholder(tf.float32, shape=img_shape, name="input_img_gray")
#input_img_deriv = tf.placeholder(tf.float32, shape=img_shape, name="input_img_deriv")

input_img = tf.placeholder(tf.float32, shape=img_shape)

gain   = tf.Variable(tf.constant(1, dtype=tf.float32, shape=img_shape), name="gain")
offset = tf.Variable(tf.constant(0, dtype=tf.float32, shape=img_shape), name="offset")

enhanced_img = tf.multiply(input_img_gray, gain) + offset

x = sobel_gradient(enhanced_img)

#----------------------------------------------------------
# COST
#----------------------------------------------------------

white_img = tf.constant(255, dtype=tf.float32, shape=[img_h * img_w])
flat_img_enh   = tf.reshape(enhanced_img, [-1])
flat_img_deriv = tf.reshape(input_img_deriv, [-1])



cost = tf.reduce_sum(tf.pow(flat_img_enh - white_img, 2)) + 90 * tf.reduce_sum(tf.pow(flat_img_enh - flat_img_deriv, 2))

#----------------------------------------------------------
# TRAIN
#----------------------------------------------------------

# Parameters
learning_rate = 0.01
training_epochs = 50
display_step = 10

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

feed = {input_img_gray: img_gray, input_img_deriv: img_deriv}
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict = feed)

    if (epoch+1) % display_step == 0:
        gen_img = sess.run(enhanced_img, feed_dict = feed)
        cv2.imwrite("output_2_{0}.png".format(epoch), gen_img)
        #cv2.imshow("generated", gen_img)
        #cv2.waitKey()