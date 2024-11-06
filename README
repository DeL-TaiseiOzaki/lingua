# Meta Lingua

**Mathurin Videau***, **Badr Youbi Idrissi***, Daniel Haziza, Luca Wehrstedt, Jade Copet, Olivier Teytaud, David Lopez-Paz. ***Equal and main contribution**

Meta Linguaは、研究用に設計されたミニマルで高速なLLMの学習・推論ライブラリです。Meta Linguaは、新しいアーキテクチャ、損失関数、データなどを試すために、簡単に変更可能なPyTorchコンポーネントを使用しています。このコードベースは、エンドツーエンドの学習、推論、評価を可能にし、速度と安定性をより良く理解するためのツールを提供することを目指しています。Meta Linguaは現在開発中ですが、このコードベースの使用方法を示すための複数の`apps`を提供しています。

<p align="center">  
 <img src="lingua_overview.svg" width="100%"/>
</p>

## クイックスタート

以下のコマンドは、Meta Lingua用の環境を作成するSLURMジョブを起動します。
環境の作成には、ダウンロード時間を除いて約5分かかります。

```bash
git clone https://github.com/facebookresearch/lingua
cd lingua

bash setup/create_env.sh
# SLURMクラスタにアクセスできる場合
sbatch setup/create_env.sh
```
環境が作成されたら、以下のコマンドで有効化できます：
```bash
conda activate lingua_<date>
```
提供されているスクリプトを使用して、huggingfaceからデータをダウンロードし準備します（`fineweb_edu`、`fineweb_edu_10bt`、または`dclm_baseline_1.0`から選択）。
以下のコマンドは`fineweb_edu`をダウンロードし、`./data`ディレクトリで学習用に準備します。その際、`terashuf`（サンプルのシャッフルに使用されるツール）に割り当てるメモリ量を指定します。
```bash
python setup/download_prepare_hf_data.py fineweb_edu <MEMORY> --data_dir ./data --seed 42
```
トークナイザー（ここではllama3）をダウンロードするには、以下のスクリプトを使用します：
```bash
python setup/download_tokenizer.py llama3 <SAVE_PATH> --api_key <HUGGINGFACE_TOKEN>
```
すべてが正常に動作するかを確認するために、デバッグジョブを起動します。**提供されている設定はテンプレートなので、動作させるためには適切に調整する必要があります（`dump_dir`、`data.root_dir`、`data.tokenizer.path`などを変更）**

```bash
# stoolはSLURM toolの略です！
python -m lingua.stool script=apps.main.train config=apps/main/configs/debug.yaml nodes=1 partition=<partition>
# ローカルで起動する場合はtorchrunを使用できます
torchrun --nproc-per-node 8 -m apps.main.train config=apps/main/configs/debug.yaml
# または1つのGPUで起動することもできます
python -m apps.main.train config=apps/main/configs/debug.yaml
```

## 学習結果

多くのダウンストリームタスクで非常に強力なパフォーマンスを達成し、[DCLM baseline 1.0](https://arxiv.org/abs/2406.11794)のパフォーマンスと一致しています。

### DCLM 60Bトークンでの1Bモデル
| モデル名        | arc_challenge | arc_easy | boolq |  copa | hellaswag |  obqa |  piqa |  siqa | winogrande |  nq  |  tqa  |
|----------------|:-------------:|:--------:|:-----:|:-----:|:---------:|:-----:|:-----:|:-----:|:----------:|:----:|:-----:|
| Transformer 1B |     36.48     |   62.83  | 62.57 | 79.00 |   63.62   | 37.40 | 75.14 | 45.19 |    61.64   | 8.75 | 26.31 |
| minGRU 1B      |     30.82     |   57.89  | 62.05 | 74.00 |   50.27   | 37.00 | 72.31 | 43.76 |    52.49   | 3.24 |  9.03 |
| minLSTM 1B     |     31.76     |   60.04  | 62.02 | 73.00 |   53.39   | 36.40 | 72.36 | 45.09 |    52.80   | 4.52 | 12.73 |
| Hawk 1B        |     34.94     |   63.68  | 62.42 | 76.00 |   63.10   | 38.20 | 73.23 | 46.01 |    55.33   | 8.42 | 23.58 |
| Mamba 1B       |     35.54     |   63.42  | 62.63 | 74.00 |   64.16   | 38.80 | 75.24 | 45.14 |    60.14   | 8.84 | 26.64 |

### 7Bモデル

| モデル名                        | arc_challenge | arc_easy | boolq | copa  | hellaswag | obqa  | piqa  | siqa  | winogrande | mmlu  | nq    | tqa   | bbh   |
|----------------------------------|---------------|----------|-------|-------|-----------|-------|-------|-------|------------|-------|-------|-------|-------|
| Mamba 7B 200Bトークン           | 47.21         | 76.03    | 65.63 | 84.00 | 77.80     | 44.00 | 80.25 | 49.69 | 70.24      | 32.81 | 20.53 | 51.93 | 20.35 |
| Llama 7B 200Bトークン           | 46.95         | 75.73    | 64.80 | 84.00 | 77.45     | 45.00 | 80.20 | 48.26 | 70.32      | 48.64 | 20.66 | 51.01 | 31.47 |
| Llama 7B squared relu 1Tトークン | 49.61         | 76.74    | 72.45 | 89.00 | 81.19     | 44.80 | 82.05 | 49.95 | 72.14      | 60.56 | 25.68 | 59.52 | 42.11 |

## プロジェクト概要

Meta Linguaは以下のような構造になっています：

```
📦meta-lingua
 ┣ 📂lingua # コアライブラリ
 ┃ ┣ 📜args.py
 ┃ ┣ 📜checkpoint.py
 ┃ ┣ 📜data.py
 ┃ ┣ 📜distributed.py
 ┃ ┣ 📜float8.py
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜optim.py
 ┃ ┣ 📜probe.py
 ┃ ┣ 📜profiling.py
 ┃ ┣ 📜stool.py
 ┃ ┣ 📜tokenizer.py
 ┃ ┗ 📜transformer.py
 ┣ 📂setup
 ┃ ┣ 📜create_env.sh
 ┃ ┗ 📜download_prepare_hf_data.py
 ┗ 📂apps # コンポーネントを組み合わせるアプリケーション
   ┣ 📂main # llamaを使用したメイン言語モデリングアプリ
   ┃ ┣ 📂configs
   ┃ ┣ 📜eval.py
   ┃ ┣ 📜generate.py
   ┃ ┣ 📜train.py
   ┃ ┗ 📜transformer.py
   ┣ 📂fastRNN 
   ┃ ┣ 📂component
   ┃ ┣ 📂hawk
   ┃ ┣ 📂minGRU
   ┃ ┣ 📂minLSTM
   ┣ 📂mamba
   ┣ 📂mtp # マルチトークン予測
   ┗ 📂plots
```

`lingua`フォルダには必須の再利用可能なコンポーネントが含まれており、`apps`フォルダにはそれらのコンポーネントを組み合わせるスクリプトが含まれています。例えば、メインの学習ループは`apps/main`にあります。これをテンプレートとして使用し、実験に合わせて自由に変更することを強くお勧めします。

Meta Linguaでは、何も神聖なものはありません。特に、できる限り簡単に変更できるように設計されています！そのため、自由にブランチを作成し、何でも変更してください。

以下は、最も重要なファイルと機能の簡単な説明です：

- **`transformer.py`**：モデルアーキテクチャを定義します。これは純粋なPyTorch `nn.Module`です！特別なことは何もありません。
- **`distributed.py`**：複数のGPUにモデルを分散させる処理を行います。これは`parallelize_module`関数を通じて行われ、通常の`nn.Module`をラップし、データ並列処理、完全シャード化データ並列処理、モデル並列処理、`torch.compile`、アクティベーションチェックポインティング、`float8`のほぼすべての組み合わせを適用します。
- **`data.py`**：LLMの事前学習用データローダー。

<p align="center">  
 <img src="dataloader.png" width="40%"/>
</p>

- **`profiling.py`**：xformersのプロファイラーの小さなラッパーで、自動的なMFUとHFUの計算を提供し、ダンプディレクトリのprofilingフォルダにプロファイルトレースを出力します。メモリプロファイリングトレースも含まれています。
- **`checkpoint.py`**：モデルチェックポイントを管理します。モデルをダンプディレクトリのcheckpointsフォルダに.distcp形式（PyTorchの新しい分散保存方式）で保存します。この形式により、異なる数のGPUや異なるシャーディングでモデルを再読み込むことができます。`torch.distributed.checkpoint.format_utils.dcp_to_torch_save`を使用して通常のPyTorchチェックポイントに変換することも、その逆も可能です。
- **`args.py`**：設定を扱うためのユーティリティ。

## 設定

ほとんどのコンポーネントは設定が必要で、これらの設定オブジェクトを表現するためにデータクラスを使用することを選択しました。`args.py`は`config.yaml`と設定辞書を各データクラスに変換する手助けをします。

例えば、`apps/main/train.py`の`TrainArgs`には、`LMTransformerArgs`、`OptimArgs`などが子として含まれています。

以下は`TrainArgs`に変換される設定ファイルの例です：

```yaml
# Meta Linguaが実験に関連するものを保存する場所です
dump_dir: /path/to/dumpdir
name: "debug"
steps: 1000

seed: 12

optim:
    lr: 3e-4
    warmup: 2000
    lr_min_ratio: 0.000001
    clip: 10.0

distributed:
    fsdp_type: full_shard
    compile: true
    selective_activation_checkpointing: false

model:
    dim: 1024
    n_layers: 8
    n_heads: 8

data:
    root_dir: data/shuffled
    sources:
      wikipedia: 80.0
      arxiv: 20.0
    batch_size: 32
    seq_len: 1024
    load_async: true
    tokenizer:
        name: sp
        path: tokenizers/llama2.model
```

## ジョブの起動

### コマンドライン引数

すべてのスクリプト（`train.py`、`eval.py`、`stool.py`）のコマンドラインインターフェースは[OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#from-command-line-arguments)を使用します。
これはドット記法での引数を受け付けます。
データクラスが以下のような場合：
```python
@dataclass
class DummyArgs:
    name: str = "blipbloup"
    mode: LMTransformerArgs = LMTransformerArgs()
    
@dataclass
class LMTransformerArgs:
    dim: int = 512
    n_layers: int = 12
```

`model.dim = 32`を渡して`LMTransformerArgs`の値を変更したり、
トップレベルの属性の場合は単に`name = tictac`のように指定できます。

**`train.py`**は設定ファイルへのパスを引数として受け取り、その設定を読み込みます。動作は以下の通りです：
1. デフォルト値で`TrainArgs`をインスタンス化します
2. 提供された設定ファイルの値でデフォルト値を上書きします
3. コマンドラインで提供された追加の引数で結果
