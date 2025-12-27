#import tensorflow as tf

#def learning_model(x,num_points):
#    model = tf.keras.models.Sequential()
#    model.add(tf.keras.layers.Dense(2, input_shape=(1,), activation='relu'))
#    model.add(tf.keras.layers.Dense(8, input_shape=(2,), activation='relu'))
#    model.add(tf.keras.layers.Dense(8, input_shape=(8,), activation='relu'))
#    model.add(tf.keras.layers.Dense(8, input_shape=(8,), activation='relu'))
#    model.add(tf.keras.layers.Dense(8, input_shape=(16,), activation='relu'))
#    model.add(tf.keras.layers.Dense(16, input_shape=(32,), activation='relu'))
#    model.add(tf.keras.layers.Dense(32, input_shape=(32,), activation='relu'))
#    model.add(tf.keras.layers.Dense(32, input_shape=(32,), activation='relu'))
#    model.add(tf.keras.layers.Dense(1*num_points, input_shape=(32,), activation='relu'))
#    print('\nModel created.')
#    print(model.summary())
#    return model(x)
