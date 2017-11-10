import cv2
import tensorflow as tf
import numpy

def normalize(image):
    image /= (image.max()/255.0)
    return image

def get_images(imgSrc: str):
    image = cv2.imread(imgSrc)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray_image, cv2.CV_8U)

    laplacian = (255-laplacian)

    cv2.imwrite("laplacian.png", laplacian)
    #cv2.imshow("aa", laplacian)
    #cv2.waitKey()

    return gray_image, laplacian


img_gray, img_deriv = get_images("bon.png");

img_h, img_w = img_gray.shape
img_shape = img_gray.shape #[img_h, img_w]

input_img_gray  = tf.placeholder(tf.float32, shape=img_shape, name="input_img_gray")
input_img_deriv = tf.placeholder(tf.float32, shape=img_shape, name="input_img_deriv")

gain   = tf.Variable(tf.constant(1, dtype=tf.float32, shape=img_shape), name="gain")
offset = tf.Variable(tf.constant(0, dtype=tf.float32, shape=img_shape), name="offset")

enhanced_img = tf.multiply(input_img_gray, gain) + offset

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
        cv2.imwrite("output_{0}.png".format(epoch), gen_img)
        #cv2.imshow("generated", gen_img)
        #cv2.waitKey()