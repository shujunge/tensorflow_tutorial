tensorflow-tutorial
====

# installing
* pip install tensorflow
* 编译源码
```
>>>git clone https://github.com/tensorflow/tensorflow.git 
>>>cd ./tensorflow
>>>git checkout v1.8.0
>>>./configure
>>>bazel build --config=cuda --config=opt //tensorflow/tools/pip_package:build_pip_package
##注意编译时要连网络
>>>bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/
# 会在/tmp 目录下生成.whl文件
>>>pip install /tmp/tensorflow-1.8.0-cp36-cp36m-linux_x86_64.whl

```
##  [high Level APIS](./High_APIS)

### * Eager Execution
### * importing data
### * Estimators


## [low level APIS](Low_APIS)


## [Acclerators](./Accelerztors)



## [ML concepts](./ML_Concepts)

## [Embeddings](./Embeddings)
## [Debugging](./Debugging)

## [TensorBoard](./TensorBoard)

## [MISC](./Misc)


