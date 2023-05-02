# 53th place solutions (Silver) - Single model and hand made feature approach.

Thanks to Kaggle for hosting this interesting competition!!!!
Very enjoyable competition!

# Team Member
@sugupoko, @hatakee, @kfuji

## SUMMARY
 - My solution is based on this notebook. Thank you @markwijkhuizen
     - Link : https://www.kaggle.com/code/markwijkhuizen/gislr-tf-data-processing-transformer-training
- The changes made are as follows:
    1. Added more features. Add features based on the following ideas.
        - Sign language is constructed from five elements: handshapes, hand orientation, movements, hand positions, and facial expressions. (We Asked Japanease Sign Language Professionals.)
    2. Applied coordinate normalization processing for each frame.
    3. Changed the parameters(epoch, MLP ratio, dropout).
- Our code is in Appendix.

## Scores transitions

| No. | Modifications                | CV     |Private|
|-----|------------------------------|--------|----|
| 1   | Baseline                     | 0.8327 ||
| 2   | Augmentation                 | 0.8338 | |
| 3   | Add feature (Vector)         | 0.8367 ||
| 4   | Add feature (velocity, distance) | 0.8408 ||
| 5   | Add feature (acceleration, angle) | 0.8444 | |
| 6   | Change MLP ratio             | 0.8496 | |
| 6   | Normalize position           | 0.8518 ||
| 7   | Epoch 300                    | 0.8575 ||
| 8   | Add feature (shape)          | 0.8643 | |
| 9   | Use All data (epoch100)      | ----   | 0.85     |

# Progress
- Early Stages
    - What was done
        - Understanding the competition
        - Understanding the baseline
        - Studying transformers
- Middle Stages
    - What was done
        - Changing network parameters
        - Implementing discussions
        - random split validation
    - Insights gained in the middle stage
        - Realized that adding features is key, from studying the basics and experimenting
- Final Stages
    - What was done
        - Adding features
        - Adjusting parameters
        - all data training


## Preprocessing
- Adjust all coordinates so that the center of numbers 11 and 12 is set to 0.
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2930242%2Ffa394dbd96e33fb875aef6ed1f25c757%2F1.png?generation=1682987922298156&alt=media)
- Agumentations (it runs outside of training loop).
    - NaN interpolation
    - 3D scaling
    - time direction scaling

Augmentation layer is in Appendix.

## Modeling
In our case, we added more features. Add features based on the following ideas.
- Sign language is constructed from five elements: handshapes, hand orientation, movements, hand positions, and facial expressions. (We Asked Japanease Sign Language Professionals.)

Here is our features.
- lip,body,hand : position, distance, velocity, accelaration, Angle, Angle velocity
- body,hand : Shape
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2930242%2Fb739316822c577ef9b4be204325f569e%2F2.png?generation=1682987881314581&alt=media)

if you want to know the details, please access to the base notebook.
- Network:
  - MLP ratio : 4
  - Embeddings : 384
  - Units : 512
  - Transfomer Block : 2
- Input 
  - input size : 32

## CV
- random split(8:2). 
    - Seed 4949 is the best!!
- The final submission was using all data

## other
- Chatgpt(GPT-4) was very helpful for writing code.

## not worked for me
- Augmentation
    - Local affine
    - Noise
- Bigger Paramer
    - MLP Ratio > 4
    - Epoch > 300
    - Length > 32


## Appendix:
### Augmentation layer code.

``` python
class NanInterpolation(tf.keras.layers.Layer):
    def __init__(self,  **kwargs):
        super(NanInterpolation, self).__init__(**kwargs)
        self.order = 3
        self.limit = 3
        
    def call(self, inputs, training=False):
        if training:
            # 入力データをNumpy配列に変換
            data = inputs.numpy()

            # 補間処理
            interpolated_data = []
            for i in range(data.shape[-1]):
                df = pd.DataFrame(data[..., i])
                # df = df.interpolate(method="spline", order=self.order, limit=self.limit, limit_direction='both')
                # df = df.interpolate(method="spline", order=self.order, limit_direction='both')
                df = df.interpolate(limit_direction='both')
                # df.fillna(method="ffill", inplace=True)   
                # df.fillna(method="bfill", inplace=True)
                interpolated_data.append(df.to_numpy())

            # 補間後のデータをテンソルに変換
            result = np.stack(interpolated_data, axis=-1)
            inputs = tf.convert_to_tensor(result, dtype=inputs.dtype)
            
        return inputs

    
class Scaling3D(tf.keras.layers.Layer):
    def __init__(self, scale_range=(0.9, 1.1), **kwargs):
        super(Scaling3D, self).__init__(**kwargs)
        self.scale_range = scale_range

    def call(self, inputs, training=False):
        if training:
            # ランダムなスケーリング係数を生成
            scale_factor = tf.random.uniform(
                (), minval=self.scale_range[0], maxval=self.scale_range[1]
            )

            # ポーズデータにスケーリング係数を適用
            inputs = inputs * scale_factor

        return inputs 

class TimeSeriesAugmentation(tf.keras.layers.Layer):
    def __init__(self, framerate_factor_range=(0.8, 1.2),  **kwargs):
        super(TimeSeriesAugmentation, self).__init__(**kwargs)
        self.framerate_factor_range = framerate_factor_range

    def call(self, inputs, training=False):
        if training:
            # フレームレート変更
            framerate_factor = tf.random.uniform(
                (), minval=self.framerate_factor_range[0], maxval=self.framerate_factor_range[1]
            )
            new_length = tf.cast(tf.cast(tf.shape(inputs)[0], tf.float32) * framerate_factor, tf.int32)
            inputs_expanded = tf.expand_dims(inputs, axis=0)
            resized_inputs = tf.image.resize(inputs_expanded, (new_length, tf.shape(inputs)[-2]))
            inputs = resized_inputs[0]

        return inputs

```
