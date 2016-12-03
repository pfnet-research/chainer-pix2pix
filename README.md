# chainer-pix2pix
chainer implementation of pix2pix
https://phillipi.github.io/pix2pix/

# usage
1. chainerの最新版を入れる
2. facade datasetを http://cmp.felk.cvut.cz/~tylecr1/facade/ から持ってくる
3. `python train_facade.py -g [GPUの番号] -i [データセットのroot directory] --out [出力ディレクトリ] --snapshot_interval 10000`
4. 数時間待つ

# facade以外のデータセットでやるとき
- データセットは位置の合った画像(あるいは画像的な構造を持つarray)のペア
- `facade_dataset.py`を書き換える。get_exampleが呼ばれた時に、(入力画像, 教師出力画像)が返るようになっていれば良い(両方numpy array)。
- `facade_visualizer.py`の可視化コードをデータセットに合わせて書き換える。
- `train_facade.py`は50行目くらいのin_ch, out_chをデータセットに合わせて書き換える。
