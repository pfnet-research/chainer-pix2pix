# chainer-pix2pix
chainer implementation of pix2pix
https://phillipi.github.io/pix2pix/

# Example result on CMP facade dataset
<img src="https://github.com/mattya/chainer-pix2pix/blob/master/image/example.png?raw=true">
左からinput, output, ground_truth


# usage
1. chainerの最新版を入れる
2. facade datasetを http://cmp.felk.cvut.cz/~tylecr1/facade/ から持ってくる
3. `python train_facade.py -g [GPUの番号] -i [データセットのroot directory] --out [出力ディレクトリ] --snapshot_interval 10000`
4. 数時間待つ
 - `--out`で指定したディレクトリに、`--snapshot_interval`で指定した頻度で、現在のモデルと、結果を可視化した画像が記録される
 - モデルは結構サイズが大きいので、`--snapshot_interval`を下げすぎないように注意

# facade以外のデータセットでやるとき
- データセットを用意する。位置の合った画像(あるいは画像的な構造を持つarray)のペアが必要。数百枚程度でもそれっぽい結果が出せると言われている。
- `facade_dataset.py`を書き換える。get_exampleが呼ばれた時に、i番目の(入力画像, 教師出力画像)が返るようになっていれば良い(両方numpy array)。
- `updater.py`でlossの計算を行っているが(現在はL1 loss + adversarial loss)、変える必要があれば変える。
- `facade_visualizer.py`の可視化コードをデータセットに合わせて書き換える。
- `train_facade.py`は50行目くらいのin_ch, out_chをデータセットに合わせて書き換える。
