import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
import dataset
import model

LOGDIR = './save'

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(tf.subtract(model.angle, model.angle_output))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.global_variables_initializer())

# summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# merge all summaries
merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)

# write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

epochs = 50
batch_size = 100
avg_loss = 0.0
num_loss = 0.0
lowest = 100.0

# train the model for 50 epochs
for epoch in range(epochs):
  for i in range(int(dataset.number_frames/batch_size)):
    frame, angle = dataset.LoadTrainBatch(batch_size)
    train_step.run(feed_dict={model.frame: frame, model.angle: angle, model.keep_prob: 0.8})
    if i % 10 == 0:
      frame, angle = dataset.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.frame: frame, model.angle: angle, model.keep_prob: 1.0})
      if loss_value < lowest:
        lowest = loss_value
      avg_loss += loss_value
      num_loss += 1.0
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.frame: frame, model.angle: angle, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * dataset.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.eframeists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

# output the average loss and lowest loss
avg_loss /= num_loss
print("The average loss is: %f" % avg_loss)
print("The lowest loss is: %f\n" % lowest)
print("Accurcy is: %f" % 100 - avg_loss)

print("Run the command line:\n" \
      "--> tensorboard --logdir=./logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
