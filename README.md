# work-keiba-learning

### Setup

* Anaconda インストール
* 仮想環境作成
* JupyterLab インストール

```
以下コマンドを Anaconda Prompt で実施

conda create -n keiba_ai_learning2
conda activate keiba_ai_learning2
conda update --all

JupyterLab インストール
> conda install jupyterlab

ワーキングディレクトリ変更
> jupyter notebook --generate-config

  生成されたファイル内にある以下設定を変更
  > c.NotebookApp.notebook_dir = 'C:\\umejima\\work\\visualpaper\\keiba\\keiba-ai\\'

pip install tensorflow
pip install keras-rl2
pip install gym

勉強に必要なもの
pip install matplotlib
```
