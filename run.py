
import numpy as np
import cv2
import tensorflow as tf
import mscnn

def open_image(image_path, image_size_limit=None):
  image = cv2.imread(image_path)
  image = image.astype(np.float32, copy=False)

  if image_size_limit:
    longer_dimension = np.max(image.shape)
    scaling = np.min((longer_dimension, image_size_limit)) / longer_dimension
    if scaling < 1:
      image = cv2.resize(
        image,
        (int(image.shape[1] * scaling), int(image.shape[0] * scaling)),
        interpolation=cv2.INTER_AREA
      )

  image = image.reshape(1, image.shape[0], image.shape[1], 3)
  return image

images = tf.placeholder('float')
predict_op = mscnn.inference_bn(images)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  saver = tf.train.Saver()
  checkpoint_dir = tf.train.get_checkpoint_state('./model')
  saver.restore(sess, checkpoint_dir.model_checkpoint_path)

  # image = open_image('./Data_original/Data_im/test_im/IMG_3_A.jpg')
  # image = open_image('./Data_original/Data_im/test_im/IMG_170_B.jpg')
  image = open_image('/Users/erichuang/Downloads/test.jpg', image_size_limit=1000)

  density_map = sess.run([predict_op], feed_dict={images: image})
  crowd_count = np.sum(density_map).round().astype(int)
  print(crowd_count)

# density_map_gt = np.array(np.load('./Data_original/Data_gt/test_gt/test_data_gt_B_170.npy'))
# density_map_gt = np.array(density_map_gt, dtype=np.float32)
# density_map_gt = density_map_gt.reshape(1, len(density_map_gt), -1)
# crowd_count_gt = np.sum(density_map_gt).round().astype(int)
# print(crowd_count_gt)
