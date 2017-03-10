#Standardization
x_std[:, 0] = (x[:, 0] - x[:, 0].mean()) / x[:, 0].std()

#regularization
reg_strength = 0.001
reg = reg_strength * tf.reduce_sum(tf.square(W))