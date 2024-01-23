import tensorflow as tf

INPUT_SIZE = (20,1)
CONV_INPUT_SIZE = (1, 28, 28)
IS_TRAINING = True


'''
< Dense layer >
Dense란 신경망 구조의 가장 기본적인 형태를 의미함
y=f(Wx+b) 의 수식을 만족하는 기본적인 신경망 형태의 층을 만드는 함수
x와 b는 각각 입력 벡터, 편향 벡터이며 W는 가중치 행렬임
Dense층 객체를 생성할 때 입력값을 함께 넣어줌
units: 출력 값의 크기, integer 혹은 long 형태
'''
# inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
# output = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)(inputs)
# print(output) # KerasTensor(type_spec=TensorSpec(shape=(None, 20, 10), dtype=tf.float32, name=None), name='dense/Sigmoid:0', description="created by layer 'dense'")


'''
< Dense layer with 1 hidden layer >
dense가 1로 바뀜. hidden layer의 수를 세는 듯
'''
# inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
# hidden = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)(inputs)
# output = tf.keras.layers.Dense(units = 2, activation = tf.nn.sigmoid)(inputs)
# print(output) # KerasTensor(type_spec=TensorSpec(shape=(None, 20, 2), dtype=tf.float32, name=None), name='dense_1/Sigmoid:0', description="created by layer 'dense_1'")


'''
< Dropout layer >
과적합을 막기위한 대표적인 정규화 방법
학습 시 특정 확률로 노드들의 값을 0으로 만듦(예측/ 테스트 시에는 적용되지 않음)
이 모듈을 이용하면 특정 keras.layers의 입력값에 드롭아웃을 적용할 수 있음
사용법은 dense 층을 만드는 방법과 유사하게 Dropout 객체를 생성하셔 사용하면 됨
rate: 드롭아웃을 적용할 확률을 지정함. 0~1 사이의 값을 받음

tf.keras.layers.dropout의 확률을 0.2로 지정할 시, 노드의 20%를 0으로 만듦
tf.nn.dropout의 확률을 0.2로 지정할 시, 노드의 80%를 0으로 만듦
'''
# inputs = tf.keras.layers.Input(shape = INPUT_SIZE)
# dropout = tf.keras.layers.Dropout(rate = 0.2)(inputs)
# hidden = tf.keras.layers.Dense(units = 10, activation = tf.nn.sigmoid)(dropout)
# output = tf.keras.layers.Dense(units = 2, activation = tf.nn.sigmoid)(hidden)
# print(output) # KerasTensor(type_spec=TensorSpec(shape=(None, 20, 2), dtype=tf.float32, name=None), name='dense_1/Sigmoid:0', description="created by layer 'dense_1'")


'''
< Concolution layer >
Conv1D의 경우 사각형이 하나의 필터가 됨
필터가 가로 방향으로 옮겨가면서(slide) 입력값에 대한 합성곱을 수행함
출력값은 1차원 벡터가 됨

자연어 처리 분야에서는 각 단어(혹은 문자) 벡터의 차원 전체에 대해 필터를 적용시키기 위해 주로 Conv1D를 사용함
필터의 높이는 입력값의 차원 수와 동일하게 연산됨
kernel_size: 필터의 가로 사이즈
filters: 필터의 개수
padding: "same"을 지정할 경우 입력값과 출력값의 가로 크기가 같아짐

(5, 10)의 입력값에 대해 kernel_size = 2, filters = 10으로 설정하면
출력값의 형태는 (4, 10)이 됨

(1, 28, 28)의 입력값에 대해 kernel_size = 3, filters = 10으로 설정하면
출력값의 형태는 (1, 28, 10)이 됨
'''
# inputs = tf.keras.layers.Input(shape = CONV_INPUT_SIZE)
# print(inputs)
# conv = tf.keras.layers.Conv1D(
#     filters=10,
#     kernel_size=3,
#     padding='same',
#     activation=tf.nn.relu)(inputs) 
# print(conv) #KerasTensor(type_spec=TensorSpec(shape=(None, 1, 28, 10), dtype=tf.float32, name=None), name='conv1d/Relu:0', description="created by layer 'conv1d'")


'''
< Max pooling layer >
피처 맵(feature map)의 크기를 줄이거나 주요한 특징을 뽑아내기 위해 합성곱 이후에 적용되는 기법
맥스 풀링은 피처 맵에 대해 최댓값만을 뽑아내는 방식이고,
평균 풀링은 피처 맵에 대해 전체 값들을 평균한 값을 뽑는 방식
pool_size: 풀링을 적용할 필터의 크기
data_format: 데이터의 표현 방법을 선택함
             "channel_last"의 경우 데이터는 (batch, length, channels) 형태여야 하고,
             "channel_first"의 경우 데이터는 (batch, channels, length) 형태여야함

맥스 풀링 결괏값을 완전 연결 계층으로 연결하기 위해서는 행렬이었던 것을 백터로 만들어야 함(맥스 풀링 결과가 행렬인가보네??)
tf.keras.layers.Flatten을 사용하면 별다른 인자값 설정 없이도 사용할 수 있음             
'''
inputs = tf.keras.layers.Input(INPUT_SIZE)
dropout = tf.keras.layers.Dropout(rate = 0.2)(inputs)
conv = tf.keras.layers.Conv1D(
    filters=10,
    kernel_size=3,
    padding='same',
    activation=tf.nn.relu)(dropout) #(20, 10)
max_pool = tf.keras.layers.MaxPool1D(pool_size = 3, padding = 'same')(conv) #(7, 10)
flatten = tf.keras.layers.Flatten()(max_pool) #(70)
hidden = tf.keras.layers.Dense(units = 50, activation = tf.nn.relu)(flatten) #(50)
output = tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax)(hidden) #(10)