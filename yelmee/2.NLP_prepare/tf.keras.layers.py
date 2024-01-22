import tensorflow as tf

INPUT_SIZE = (20,1)
CONV_INPUT_SIZE = (1, 28, 28)
IS_TRAINING = True

# < Dense Layer >
# Dense란 신경망 구조의 가장 기본적인 형태를 의미함
# y=f(Wx+b) 의 수식을 만족하는 기본적인 신경망 형태의 층을 만드는 함수
# x와 b는 각각 입력 벡터, 편향 벡터이며 W는 가중치 행렬임
# Dense층 객체를 생성할 때 입력값을 함께 넣어줌
# units: 출력 값의 크기, integer 혹은 long 형태
inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
output = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)(inputs)

# < Dense Layer with 1 hidden layer >
inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
hidden = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)(inputs)
output = tf.keras.layers.Dense(units = 2, activation = tf.nn.sigmoid)(inputs)
