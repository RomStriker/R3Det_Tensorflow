import os
import tensorflow as tf

trained_checkpoint_prefix = '/home/sommarjobbare/R3Det_Tensorflow/output/trained_weights/RetinaNet_DOTA_R3Det_4x_20200819/DOTA_8000model.ckpt'
export_dir = os.path.join('export_dir', '0')

graph = tf.Graph()

with tf.compat.v1.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.TRAINING, tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()

