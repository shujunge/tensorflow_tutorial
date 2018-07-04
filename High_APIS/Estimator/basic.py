##预创建的 Estimator

"""
1.编写一个或多个数据集导入函数。 例如，您可以创建一个函数来导入训练集，并创建另一个函数来导入测试集。
每个数据集导入函数都必须返回两个对象：
一个字典，其中键是特征名称，值是包含相应特征数据的张量（或 SparseTensor）
一个包含一个或多个标签的张量
2.定义特征列。 每个 tf.feature_column 都标识了特征名称、类型和任何输入预处理。
例如，以下代码段创建了三个存储整数或浮点数据的特征列。前两个特征列仅标识了特征的名称和类型。
第三个特性列还指定了一个 lambda，该程序将调用此 lambda 来调节原始数据.

3.实例化相关的预创建的 Estimator。 例如，下面是对预创建的 Estimator（名为 LinearClassifier）进行示例实例化．
estimator = tf.estimator.Estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )
    
4.调用训练、评估或推理方法。例如，所有 Estimator 都提供训练模型的 train 方法。
# my_training_set is the function created in Step 1
estimator.train(input_fn=my_training_set, steps=2000)
"""


##################################################################################

import tensorflow as tf
# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_inception_v3.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)

##################################################################################