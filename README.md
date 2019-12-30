# lassoadmm
このプログラムはPython3.7でテストされています。  
  
プロジェクト構造  
lassoadmm  
 ┣README.md  
 ┣lassoadmm.py  
 ┗test_lassoadmm.py  
モジュール、テストコードともに1ファイルで済んだため、ルートディレクトリに直接置いています。  
  
プログラムの実行方法  
ルートディレクトリで  
$python test_lassoadmm.py  
を実行すると、自作したLasso回帰モジュールを使ったサンプルプログラムが実行できます。  
他のコードでモジュールを利用する場合、scikit-learnのlinear_model.Lassoと同じように使える予定です。 
import lassoadmmをして、  
model = lassoadmm.Admm(lambd=1.0, rho=1.0, max_iter=1000)
というように引数λ、ρ、max_iter（全て正数）を与えて読み込みます。  
λは係数を無視する際の閾値で、大きいほどモデルが疎になりやすいです。  
ρは正則化項の係数で、大きいほど正則化項の制約が強くなります。  
max_iterは係数の更新の繰り返し回数です。  
  
依存ライブラリのインストール方法  
使用しているライブラリはnumpy、scikit-learnの2つです。それぞれ  
$pip install numpy  
$pip install scikit-learn  
を実行すればインストールすることが出来ます。  
念のためPyPIのURLを張っておきます。  
numpy -> https://pypi.org/project/numpy/  
scikit-learn -> https://pypi.org/project/scikit-learn/  
