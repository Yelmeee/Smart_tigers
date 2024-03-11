import tensorflow as tf
from tensorflow.keras import layers

'''
< Sequential API >
tf.keras.Sequential은 케라스를 활용해 모델을 구축할 수 있는 가장 간단한 형태의 API
간단한 순차적인 레이어의 스택을 구현할 수 있음

예시는 간단한 형태의 환전 연결 계층(fully-connected layers)
sequential 인스턴스를 생성한 후 해당 인스턴스에 여러 레이어를 순차적으로 더하면 모델 완성됨
입력값을 더한 순서에 맞게 레이어들을 통과 시킨 후 최종 출력값을 뽑아옴

모델의 층들이 순차적으로 구성돼 있지 않는 경우에는 Sequential 모듈 사용이 어려움
'''
# model = tf.keras.Sequential()
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

'''
< Funtional API >
다중 입/출력값 모델, 공유 층을 활용하는 모델, 데이터 흐름이 순차적이지 않은 모델은 Sequential 모듈을 사용하기 어려움
이럴 경우 케라스의 Funtional API를 사용하거나 Subclassing 방식을 사용하는 것이 적절함

입력값을 받는 Input 모듈을 선언함
이 모듈을 선언할 때는 모델의 입력으로 받는 값의 형태(shape)를 정의하면 됨.
Input 모듈을 정의 한 후 입력값을 적용할 레이어를 호출할 때 인자로 전달하는 방식으로 구현
최종 출력값을 사용해 모델을 학습하면 마지막 출력값이 Sequential로 구현햇을 때의 모델과 동일한 형태가 됨
'''
# inputs = tf.keras.Input(shape=(32,))
# x = layers.Dense(64, activation='relu')(inputs)
# x = layers.Dense(64, activation='relu')(inputs)
# predictions = layers.Dense(10, activation='softmax')(x)

'''
< Subclassing(Custom Model) >
tf.keras.Model을 상속받고 모델 내부 연산들을 직접 구현하면 됨
객체를 생성할 때 호출되는 __init__ 메서드와 
생성된 인스턴스를 호출할 때(즉, 모델 연산이 사용될 때) 호출되는 call 메서드만 구현하면 됨
'''
# class MyModel(tf.keras.Model):

#     def __init__(self, hidden_dimension, hidden_dimension2, output_dimension):
#         super(MyModel, self).__init__(name='my model')
#         self.dense_layer1 = layers.Dense(hidden_dimension, activation='relu')
#         self.dense_layer2 = layers.Dense(hidden_dimension2, activation='relu')
#         self.dense_layer3 = layers.Dense(output_dimension, activation='softmax')

#     def call(self, inputs):
#         x = self.dense_layer1(inputs)
#         x = self.dense_layer2(x)

#         return self.dense_layer3(x)
    
'''
< keras 모델의 내장 API를 활용한 모델 학습법 >

1. 학습 과정에서 사용될 손실함수, 옵티마이저, 평가에 사용될 지표 등을 정의
2. fit 메서드를 호출하면 데이터들이 모델을 통과하며 학습이 진행됨
'''
# model.compile(optimizer=tf.keras.optimizers.Adam(),
#               loss=tf.keras.losses.CategoricalCrossentropy(),
#               metrics=[tf.keras.metrics.accuracy()])

# model.fit(x_train, y_train, batch_size=64, epochs=3)




