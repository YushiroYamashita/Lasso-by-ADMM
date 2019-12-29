import numpy as np
from sklearn.datasets import load_diabetes  #糖尿病患者のデータセット
from sklearn.preprocessing import StandardScaler    #説明変数の標準化に利用
from sklearn.model_selection import train_test_split    #トレーニングデータとテストデータに分けて性能確認する
import lassoadmm

diabetes = load_diabetes()  #データセットを読み込む
print("データセットの数(データ数, 説明変数の数)", end=":")
print (diabetes.data.shape)

X = diabetes.data   #Xに説明変数を格納
y = diabetes.target #yに目標変数を格納

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)   #ランダムでデータを分離

print("トレーニングデータの数(データ数, 説明変数の数)", end=":")
print(X_train.shape)

print("テストデータの数(データ数, 説明変数の数)", end=":")
print(X_test.shape)

scaler = StandardScaler()   #説明変数の標準化
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#性能確認
model = lassoadmm.Admm(lambd=1.0, rho=1.0, max_iter=1000)
model.fit(X_train_scaled, y_train)

np.set_printoptions(formatter={'float': '{: 0.10f}'.format})    #見やすいよう設定
print("")
print("ADMMによるLasso")
print("係数行列↓")
print(model.coef_)
print("")
print("トレーニングデータのR2スコア", end=":"),
print(np.corrcoef(model.predict(X_train_scaled),y_train)[0,1]**2)
print("テストデータのR2スコア", end=":"),
print(np.corrcoef(model.predict(X_test_scaled),y_test)[0,1]**2)