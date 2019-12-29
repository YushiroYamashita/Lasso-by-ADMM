# Lasso-by-ADMM
このプログラムはPython3.7でテストされています。

プロジェクト構造
lassoadmm
 ┣README.txt
 ┣lassoadmm.py
 ┗test_lassoadmm.py
モジュール、テストコードともに1ファイルで済んだため、ルートディレクトリに直接置いています。

プログラムの実行方法
lassoadmm内で
$python test_lassoadmm.py
を実行すると、自作したLasso回帰モジュールを使ったサンプルプログラムが実行できます。
他のコードでモジュールを利用する場合、scikit-learnのlinear_model.Lassoと同じように使える予定です。

依存ライブラリのインストール方法
使用しているライブラリはnumpy、scikit-learnの2つです。それぞれ
$pip install numpy
$pip install scikit-learn
を実行すればインストールすることが出来ます。
念のためPyPIのURLを張っておきます。
numpy -> https://pypi.org/project/numpy/
scikit-learn -> https://pypi.org/project/scikit-learn/
