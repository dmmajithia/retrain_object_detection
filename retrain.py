# import tensorflow as tf
import tensorflow.compat.v1 as tf
from object_detection.utils import dataset_util
from pickler import ImageData as datapickle
import io


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('pickle_name', '', 'Pickle filename')
FLAGS = flags.FLAGS


def create_train_tf_record(example):
  # TODO(user): Populate the following variables from your example.
  height = 300 # Image height
  width = 300 # Image width
  filename = b"{example.imagepath}" # Filename of the image. Empty if image is not from file
  encoded_image_data = example.encoded_jpg() # Encoded image bytes
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [example.box[0][0]] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [example.box[1][0]] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [example.box[0][1]] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [example.box[1][1]] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [b'train'] # List of string class name of bounding box (1 per box)
  classes = [6] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example


class train_example():
	def __init__(self,imagepath,box):
		self.imagepath = imagepath
		self.box= box
		
	def encoded_jpg(self):
		with tf.gfile.GFile(self.imagepath, 'rb') as fid:
			return fid.read()

def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples variable
  examples = []
  dataframe = datapickle.load(FLAGS.pickle_name)
  for sub in dataframe.subs:
  	for imagename in sub.names:
  		example = train_example(imagepath=sub.path + imagename, box=sub.boxes[imagename])
  		examples.append(example)

  for example in examples:
    tf_example = create_train_tf_record(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()