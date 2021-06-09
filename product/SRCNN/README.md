# プロダクト開発演習  
## テーマ：JPEG圧縮画像の超解像モデル開発  
### 摘要  
**画像の保存領域削減のため縮小かつJPEG圧縮された画像を補完して美しく参照するため、JPEG圧縮画像の超解像モデルを開発する。JPEG圧縮画像の超解像モデルを実装し超解像処理結果の確認と考察を行う。基本とするニューラルネットワークモデルはSRCNNとし、Pytorchフレームワーク上に参照論文記載の構成を実装した。まず初めに、BICUBIC拡大縮小画像を学習対象として、SRCNNの基本性能を確認した。次に、BICUBIC拡大縮小に加えて、OoenCVによるJPEG圧縮処理した画像を学習対象として、複数のモデルを提案し、その性能を評価した。データセットとして、[General100](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)データセットを用い学習並びにホールドアウト法によるPSNR(dB)にて評価を行った。最後に、テスト結果(testview.ipynb)検討から提案モデルの優劣を考察した。**  

**学習処理(train.ipynb)概要（本jupyter-notebook）**  
  1. 環境定義  
  2. データローディングとデータ拡張、画像劣化処理
  3. モデル定義とインスタンス  
  4. 学習ループ  

**学習結果表示処理(viewres.ipynb)**  
学習過程のPSNRとLossグラフ化  

**テストデータ評価(testview.ipynb)**  
テスト用画像を用いた各モデルの評価とPSNRまとめ

### 参照論文モデル(train.ipynb)  
**SRCNN**    
参照論文:"Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, in Proceedings of European Conference on Computer Vision (ECCV), 2014"  

### 学習データ及びデータ拡張  
[General100 Dataset](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)を学習データとして用いる  
100画像を学習用途80、バリデーション用途、テスト用途に各10画像に分け学習を行うが、データセットのダウンロード、展開、dataディレクトリへの分割については[付録参照](#学習データ)  
**データ拡張：PILのFlip, Mirror, Rotateを実装**  
**画像劣化手法：PILのBICUBLIC拡大縮小並びにOpenCVによるJPEG圧縮処理を実装**  

### 提案モデル(train.ipynb)  
**特徴抽出層のカーネルサイズ9x9に11x11の特徴抽出層を追加：SRCNN11**  
演算量は同等のまま特徴抽出方法を変更  
広範囲の特徴を加味した推論を可能とする  
**5層モデル：SYM**  
SRCNNに残差機構を追加し、低層の情報を上位層へ直接伝えると共に、勾配消失を回避  

### 学習結果(viewres.ipynb)
**PSNRの推移グラフ：学習結果(dB推移)をnpyで保存し、別ipynb(viewres.ipynb)で表示**  
自身のGPU環境ではjupyter-notebook未サポートのため 

### 考察並びに改善案(testview.ipynb)  
