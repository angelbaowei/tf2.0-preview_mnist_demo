# tf2.0-preview_demo
mnist on tf2.0-preview(Tensorflow 2.0)

train and restore_eval use ckpt

saved to .pb must use @tf.function because it is autograph. Combine eager with autograph.
