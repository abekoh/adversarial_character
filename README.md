# adversarial_character
CNNを騙す文字画像を遺伝的アルゴリズムで作成．

## Requirement
- Python3 (3.5.3)
- deap (1.0.2)
- Keras (2.0.4)
- Tensorflow (1.1.0)
- Numpy
- Scipy
- Pillow
- h5py
- imageio

## Setup
```
pip install deap
pip install keras
vim ~/.keras/keras.json # change "theano" to "tensorflow"
pip install tensorflow # or tensorflow-gpu
pip install pillow
pip install h5py
pip install imageio
```

## How to use
### Train model
マルチフォント文字認識器を学習しそのモデルを保存．
6628種類のフォントで学習済みのモデルはこちら．
```
python train_model.py {学習画像パス，ラベルのcsv} [options]
```

**画像サイズは200x200の2値画像のみ利用可**

|オプション|効果|
|:-|:-|
|`-t (--test) {テスト画像パス，ラベルのcsv}`|テストを行う場合|
|`--hdf5`|モデルの保存パス(デフォルトでは'trained_weight.hdf5'|
|`-b (--batch_size)`|バッチサイズ|
|`-e (--epoch)`|学習回数|

#### CSVについて
1列目は画像パス，2列目はクラスID(Aから順に0,1,2...)
```
/home/hoge/font0000/A.png,0
/home/hoge/font0000/B.png,1
...
/home/hoge/font6627/Y.png,24
/home/hoge/font6627/Z.png,25
```

### Make adversarial character
AdversarialCharacter作成．

```
python make_adv_char.py {加工前の文字画像} {加工前の文字のアルファベット} {騙す先のアルファベット} [options]
```

|オプション|効果|
|:-|:-|
|`-d (--dst_path)`|保存先のパス(デフォルトでは'output')|
|`-t (--trained_weights)`|学習済みモデルのパス(デフォルトでは'trained_weight.hdf5')|
|`--cxpb`|交叉が起こる確率|
|`--mutpb`|突然変異が起こる確率|
|`--ngen`|進化の最大回数|
|`--npop`|子孫の数|
|`--breakacc`|打ち止めにする正解率|

