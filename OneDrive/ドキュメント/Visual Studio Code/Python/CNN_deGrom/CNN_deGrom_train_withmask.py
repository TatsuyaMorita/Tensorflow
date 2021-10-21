import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import os
import pickle
 
def list_imgs(directory, ext='jpg|jpeg|bmp|png|ppm'):
    return [os.path.join(root, f) for root, _, files in os.walk(directory) for f in files if re.match(r'([\w]+\.(?:' + ext + '))', f.lower())]

def plot_history(history, 
                save_graph_img_path, 
                fig_size_width, 
                fig_size_height, 
                lim_font_size):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # グラフ表示
    plt.figure(figsize=(fig_size_width, fig_size_height))
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = lim_font_size  # 全体のフォント
    #plt.subplot(121)

    # plot accuracy values
    plt.plot(epochs, acc, color = "blue", linestyle = "solid", label = 'train acc')
    plt.plot(epochs, val_acc, color = "green", linestyle = "solid", label= 'valid acc')
    #plt.title('Training and Validation acc')
    #plt.grid()
    #plt.legend()
 
    # plot loss values
    #plt.subplot(122)
    plt.plot(epochs, loss, color = "red", linestyle = "solid" ,label = 'train loss')
    plt.plot(epochs, val_loss, color = "orange", linestyle = "solid" , label= 'valid loss')
    #plt.title('Training and Validation loss')
    plt.legend()
    plt.grid()

    plt.savefig(save_graph_img_path)
    plt.close() # バッファ解放

def main():
    # ハイパーパラメータ
    batch_size = 5 # バッチサイズ
    num_classes = 2 # 分類クラス数
    epochs = 200      # エポック数(学習の繰り返し回数)
    dropout_rate = 0.2 # 過学習防止用：入力の20%を0にする（破棄）

    # 入力画像のパラメータ
    img_width = 32 # 入力画像の幅
    img_height = 32 # 入力画像の高さ
    img_ch = 3 # 3ch画像（RGB）で学習

    # データの保存先(自分の環境に応じて適宜変更)
    SAVE_DATA_DIR_PATH = "keras/04_img/ex2_data/"

    # ディレクトリがなければ作成
    os.makedirs(SAVE_DATA_DIR_PATH, exist_ok=True)

    # グラフ画像のサイズ
    FIG_SIZE_WIDTH = 12
    FIG_SIZE_HEIGHT = 10
    FIG_FONT_SIZE = 25

    data_x = []
    data_y = []
    num_classes = 2

    # クラス0の画像データ群をロード
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "img0"):
        img = img_to_array(load_img(filepath, target_size=(img_width,img_height, img_ch)))
        data_x.append(img)
        data_y.append(0) # 教師データ（正解）
    
    # クラス1の画像データ群をロード
    for filepath in list_imgs(SAVE_DATA_DIR_PATH + "img1"):
        img = img_to_array(load_img(filepath, target_size=(img_width,img_height, img_ch)))
        data_x.append(img)
        data_y.append(1) # 教師データ（正解）

   
    
    # NumPy配列に変換
    data_x = np.asarray(data_x)

    # 学習データはNumPy配列に変換し
    data_y = np.asarray(data_y)

    # 学習用データとテストデータに分割
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

    # 学習データはfloat32型に変換し、正規化(0～1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # 正解ラベルをone hotエンコーディング
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # データセットの個数を表示
    print(x_train.shape, 'x train samples')
    print(x_test.shape, 'x test samples')
    print(y_train.shape, 'y train samples')
    print(y_test.shape, 'y test samples')

   
    # モデルの構築
    model = Sequential()

    # 入力層:32×32*3
    # 【2次元畳み込み層】
    # Conv2D：2次元畳み込み層で、画像から特徴を抽出（活性化関数：relu）
    # 入力データにカーネルをかける（「3×3」の32種類のフィルタを各マスにかける）
    # 出力ユニット数：32（32枚分の出力データが得られる）
    model.add(Conv2D(32,(3,3), 
                padding='same', 
                input_shape=x_train.shape[1:],
                activation='relu'))

    # 【2次元畳み込み層】
    # 画像から特徴を抽出（活性化関数：relu）
    # relu(ランプ関数)は、フィルタ後の入力データが0以下の時は出力0（入力が0より大きい場合はそのまま出力）
    # 入力データにカーネルをかける（「3×3」の32種類のフィルタを使う）
    # 出力ユニット数：32（32枚分の出力データが得られる）
    # 問題が複雑ならフィルタの種類を増やす
    # padding="same"は:の入力と同じ長さを出力がもつように入力にパディング
    model.add(Conv2D(32,(3,3),
                padding='same',
                activation='relu'))
    
     # 【プーリング層】
    # 特徴量を圧縮する層。（ロバスト性向上、過学習防止、計算コスト抑制のため）
    # 畳み込み層で抽出された特徴の位置感度を若干低下させ、対象とする特徴量の画像内での位置が若干変化した場合でもプーリング層の出力が普遍になるようにする。
    # 画像の空間サイズの大きさを小さくし、調整するパラメーターの数を減らし、過学習を防止
    # pool_size=(2, 2):「2×2」の大きさの最大プーリング層。
    # 入力画像内の「2×2」の領域で最大の数値を出力。
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # ドロップアウト(過学習防止用, dropout_rate=0.2なら20%のユニットを無効化）
    model.add(Dropout(dropout_rate))

    # 【2次元畳み込み層】
    # 画像から特徴を抽出（活性化関数：relu）
    # relu(ランプ関数)は、フィルタ後の入力データが0以下の時は出力0（入力が0より大きい場合はそのまま出力）
    # 入力データにカーネルをかける（「3×3」の64種類のフィルタを使う）
    # 出力ユニット数：64（64枚分の出力データが得られる）
    # 問題が複雑ならフィルタの種類を増やす
    model.add(Conv2D(64,(3,3),
                padding='same',
                activation='relu'))
    
    # 【2次元畳み込み層】
    # 画像から特徴を抽出（活性化関数：relu）
    # relu(ランプ関数)は、フィルタ後の入力データが0以下の時は出力0（入力が0より大きい場合はそのまま出力）
    # 入力データにカーネルをかける（「3×3」の64種類のフィルタを使う）
    # 出力ユニット数：64（64枚分の出力データが得られる）
    # 問題が複雑ならフィルタの種類を増やす
    model.add(Conv2D(64,(3,3),
                padding='same',
                activation='relu'))

    # 【プーリング層】
    # 特徴量を圧縮する層。（ロバスト性向上、過学習防止、計算コスト抑制のため）
    # 畳み込み層で抽出された特徴の位置感度を若干低下させ、対象とする特徴量の画像内での位置が若干変化した場合でもプーリング層の出力が普遍になるようにする。
    # 画像の空間サイズの大きさを小さくし、調整するパラメーターの数を減らし、過学習を防止
    # pool_size=(2, 2):「2×2」の大きさの最大プーリング層。
    # 入力画像内の「2×2」の領域で最大の数値を出力。
    model.add(MaxPooling2D(pool_size=(2, 2)))

     # ドロップアウト(過学習防止用, dropout_rate=0.2なら20%のユニットを無効化）
    model.add(Dropout(dropout_rate))

    # 平坦化（次元削減）
    # 1次元ベクトルに変換
    model.add(Flatten())

    # 全結合層
    # 出力ユニット数：512
    model.add(Dense(512, activation='relu'))

    # ドロップアウト(過学習防止用, dropout_rate=0.2なら20%のユニットを無効化）
    model.add(Dropout(dropout_rate))
    
    # 全結合層
    # 10分類（0から9まで）なので、ユニット数10, 分類問題なので活性化関数はsoftmax関数
    # Softmax関数で総和が1となるように、各出力の予測確率を計算
    model.add(Dense(num_classes, activation='softmax')) # 活性化関数：softmax

    # モデル構造の表示
    #model.summary()

    # コンパイル（多クラス分類問題）
    # 最適化：RMSpropを使用
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    # 構築したモデルで学習（学習データ:trainのうち、10％を検証データ:validationとして使用）
    # verbose=1:標準出力にログを表示
    history = model.fit(x_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_split=0.1)

    # テスト用データセットで学習済分類器に入力し、パフォーマンスを計測
    score = model.evaluate(x_test, 
                            y_test,
                            verbose=0
                            )

    # パフォーマンス計測の結果を表示
    # 損失値（値が小さいほど良い）
    print('Test loss:', score[0])

    # 正答率（値が大きいほど良い）
    print('Test accuracy:', score[1])
    
    
    # 学習過程をプロット
    plot_history(history, 
                save_graph_img_path = SAVE_DATA_DIR_PATH + "graph.png", 
                fig_size_width = FIG_SIZE_WIDTH, 
                fig_size_height = FIG_SIZE_HEIGHT, 
                lim_font_size = FIG_FONT_SIZE)
    
    # モデル構造の保存
    open(SAVE_DATA_DIR_PATH  + "model.json","w").write(model.to_json())  

    # 学習済みの重みを保存
    model.save_weights(SAVE_DATA_DIR_PATH + "weight.hdf5")

    # 学習履歴を保存
    with open(SAVE_DATA_DIR_PATH + "history.json", 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    main()