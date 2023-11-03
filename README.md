#人工智慧課程期中報告-初學者的 TensorFlow 2.0 教程

## 第一步－將 TensorFlow 導入到程式碼
```python
import tensorflow as tf
```

## 第二步－加載Minist數據庫

將樣本數據從整数轉换為浮点数。

```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

```
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11490434/11490434 [==============================] - 0s 0us/step

## 第三步－建構機器學習模型

透過堆疊層來建構 tf.keras.Sequential 模型。

```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

對於每個樣本，模型都會傳回一個包含 logits 或 log-odds 分數的向量，每個類別一個。

```python
predictions = model(x_train[:1]).numpy()
predictions
```
array([[ 0.5755979 ,  0.03832029, -0.60393435,  0.32401356, -0.72236824,
         0.05254771, -0.1035907 , -0.46912807,  0.32492474,  0.09436399]],
      dtype=float32)

使用tf.nn.softmax 函數將這些 logits 轉換為每個類別的機率：

```python
tf.nn.softmax(predictions).numpy()
```
array([[0.1727045 , 0.10091761, 0.05309325, 0.13428949, 0.0471633 ,
        0.10236367, 0.08756606, 0.06075541, 0.1344119 , 0.1067349 ]],
      dtype=float32)

使用 losses.SparseCategoricalCrossentropy 為訓練定義損失函數，它會接受 logits 向量和 True 索引，並為每個樣本傳回一個標量損失。

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

此損失等於 true 類別的負對數機率：如果模型確定類別正確，則損失為零。

這個未經訓練的模型給出的機率接近隨機（每個類別為 1/10），因此初始損失應該接近 -tf.math.log(1/10) ~= 2.3。

```python
loss_fn(y_train[:1], predictions).numpy()
```
2.2792234

在開始訓練之前，使用 Keras Model.compile 配置和編譯模型。 
將 optimizer 類別設為 adam，將 loss 設定為您先前定義的 loss_fn 函數，並透過將 metrics 參數設為 accuracy 來指定要為模型評估的指標。

```python
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
     
```

## 第四步－訓練並評估模型

使用 Model.fit 方法調整您的模型參數並最小化損失：

```python
model.fit(x_train, y_train, epochs=5)
```
Epoch 1/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.2950 - accuracy: 0.9151
Epoch 2/5
1875/1875 [==============================] - 15s 8ms/step - loss: 0.1419 - accuracy: 0.9577
Epoch 3/5
1875/1875 [==============================] - 10s 5ms/step - loss: 0.1050 - accuracy: 0.9682
Epoch 4/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0887 - accuracy: 0.9730
Epoch 5/5
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0741 - accuracy: 0.9767
<keras.src.callbacks.History at 0x79dc7cc905e0>

Model.evaluate 方法通常在 "Validation-set" 或 "Test-set" 上檢查模型效能。

```python
model.evaluate(x_test,  y_test, verbose=2)
```
313/313 - 1s - loss: 0.0675 - accuracy: 0.9793 - 880ms/epoch - 3ms/step
[0.0675475224852562, 0.9793000221252441]

現在，這個照片分類器的準確度已經接近 98%。

如果您想讓模型返回機率，可以封裝經過訓練的模型，並將 softmax 附加到該模型：

```python
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```
```python

probability_model(x_test[:5])
```
<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[6.9868713e-07, 7.3982793e-09, 8.1544586e-06, 1.0590465e-04,
        1.0060162e-10, 3.3199171e-07, 1.5496221e-13, 9.9987686e-01,
        8.6935806e-08, 7.8739458e-06],
       [1.3347297e-09, 1.7784080e-04, 9.9978715e-01, 1.9186044e-05,
        4.4622572e-17, 1.5327085e-05, 2.9827696e-08, 2.6966916e-12,
        5.1624733e-07, 2.2245647e-15],
       [3.0035192e-08, 9.9974829e-01, 3.2191685e-05, 2.4198278e-06,
        2.6656175e-05, 1.5864445e-06, 1.0337752e-06, 1.4740136e-04,
        4.0295825e-05, 5.5774549e-08],
       [9.9955863e-01, 9.3173753e-08, 2.4591730e-04, 1.7508968e-08,
        2.0922444e-06, 3.9424248e-07, 1.4435763e-04, 4.1009047e-05,
        7.4048535e-07, 6.7769970e-06],
       [3.4853494e-07, 4.0361012e-08, 1.7212931e-06, 8.2820790e-08,
        9.9713969e-01, 2.5730953e-06, 4.2647821e-06, 2.7146960e-05,
        7.2234161e-07, 2.8234241e-03]], dtype=float32)>





## 參考文章:
https://tensorflow.google.cn/tutorials/quickstart/beginner?hl=zh_cn
