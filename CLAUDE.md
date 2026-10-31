# CLAUDE.md

このファイルは、Claude Code がこのリポジトリで作業するときのルールです。

## プロジェクトの目的

決算説明資料の図表（グラフ・表）に強い RAG を作り、次の3方式を**自作の評価セットで定量比較**する。完成したアプリそのものより、**比較実験の正しさ**が成果物の中心。

- **A. テキストのみ**：PyMuPDF でテキストを抽出 → BGE-M3 で埋め込み（基準）
- **B. 図表キャプション**：ページ画像を丸ごと画像対応 LLM に渡し、ページ内の図表ごとに説明文と Markdown 表を出させて埋め込む（B1: Claude API／B2: ローカルの Qwen2.5-VL）。図表の切り出しはしない（README の Design decisions を参照）
- **C. ページ画像検索**：ColQwen2.5／ColPali の複数ベクトル埋め込み ＋ Late Interaction（MaxSim）
- 回答生成：どの方式でも、検索で見つかったページ画像と質問を画像対応 LLM に渡して回答させる。**評価セットの問題を Claude が作っているので、回答生成には Claude を使わない**。回答生成は **Gemini の API**（2026-10-31 決定）。エージェント型のツール（Codex など）は、リポジトリ内の正解を読めてしまうので回答生成に使わない
- 採点：検索は Recall@1・3・5 と MRR。回答は、数値の質問を自動判定し、文章の質問は **DeepSeek V4.1 Flash の API** が採点する。問題を作る（Claude）・答える（Gemini）・採点する（DeepSeek）を別々のモデルにして、自分の答えを自分で採点する偏りを避けるため（詳細は `eval/README.md`）

開発者は ML エンジニア／データサイエンティストを目指して学習中。**開発者がすべてのコードを説明できる状態**を保つことが、速さより優先される。

## 環境

- Windows 11 ／ PowerShell 5.1 ／ AMD Radeon RX 9070 XT（gfx1201, VRAM 16GB）／ RAM 32GB
- Python 3.12（winget で入れたシステムの Python）／ uv で管理
- PyTorch 2.14.0 + ROCm 10.1.0（AMD の Windows 版。ROCm 版でも API 名は `torch.cuda`）
- Ollama（`qwen2.5vl:7b`）は `http://127.0.0.1:11434` で常駐

## コマンド

```powershell
uv run python scripts/check_gpu.py   # GPU の確認（"OK" が出ること）
uv run pytest                        # テスト
uv run streamlit run scripts/eval_editor.py   # 評価セットの入力アプリ
uv run python scripts/validate_eval.py        # 評価セットの形式チェック
uv add <pkg>                         # 依存関係の追加（pip install は使わない）
uv add --dev <pkg>                   # 開発用の依存関係
```

Claude Code の PowerShell では PATH が古いことがあるので、コマンドの前に次を付ける。
`$env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')`

## 依存関係のルール

- `pip install` や `uv pip install` で直接入れない。必ず `uv add` で `pyproject.toml` と `uv.lock` に記録する。
- **依存関係を追加・更新したら、必ず `torch.version.hip` が None でないことを確認する。** PyTorch が CUDA 版や CPU 版に置き換わっていないかのチェック。
- `pyproject.toml` の `[tool.uv]` にある AMD の配布元、`exclude-dependencies = ["triton"]`、`environments` の設定は、理由を説明して許可を得てから変更する（経緯は README の Setup notes を参照）。
- `uv.toml` を作らない（作ると `pyproject.toml` の `[tool.uv]` が無視される）。

## 保護されたファイル（変更は開発者の明示的な指示があるときだけ）

実験の結論はここで決まる。バグや、AI が AI を採点する構造が入り込むと、すべての結果が無意味になる。

- `eval/questions.csv` … 評価セット（質問・正解ページ・正解）。開発者の判断（2026-10-13）で、q010 以降は Claude が作る。そのときのルール：
  - Claude が作った問題は `author=claude`、`reviewed=none` で追加する。作成とは別の AI（文脈を持たないサブエージェント）がページ画像で検証し、通ったものだけ `reviewed=ai` にする。自分で作った問題を自分で `reviewed=ai` にしない。
  - 根拠ページは、`data/pages.jsonl` で同じ値を検索し、漏れなく登録する。
  - 開発者が作った問題（`author=human`）を直すときは、メモ欄に「AIが修正」と理由を書く。
  - 入力アプリと `src/multimodal_rag/evalset.py` は書式の変換と検証だけを行い、中身を変えない。
- `src/multimodal_rag/eval/metrics.py` … Recall@k、MRR など。開発者の判断（2026-10-26）で Claude が実装する。そのときのルール：
  - 指標の定義（何を正解とみなすか、順位の数え方）は開発者が決める。定義は docstring に書く。
  - 実装より先に `tests/test_metrics.py` を書く。期待値は手計算で求め、計算の過程をコメントに残す。
  - 実装後、文脈を持たない別の AI（サブエージェント）に、定義どおりに動くかを検証させる。
  - 一度確定したら、開発者の明示的な指示なしに書き換えない。
- `tests/test_metrics.py` … 上の指標のテスト。手計算で答えが分かる小さな例で書く。

保護されたファイルに問題を見つけたら、直さずに報告する。

## ディレクトリ構成（予定）

```
src/multimodal_rag/
  ingest/       PDF → ページ画像・テキスト（data/pages.jsonl）
  retrieval/    方式 A / B1 / B2 / C（共通のインターフェースで実装）
  generation/   回答生成（Gemini）と、文章の答えの採点（DeepSeek）
  eval/         指標と評価の実行（metrics.py は保護）
scripts/        実行用のスクリプト（各方式のインデックス作成・評価の実行など）
tests/          pytest
eval/           評価セット（コミットする）
results/        実験結果の表やグラフ（コミットする）
data/           決算資料の PDF とその加工物（.gitignore 済み。コミットしない）
outputs/        キャッシュや途中生成物（.gitignore 済み）
```

- 検索方式はすべて同じインターフェースを実装する（例：`index(pages)` と `search(query, k) -> list[PageHit]`）。評価のコードは方式に依存しない。
- 設定値（モデル名、k、チャンクサイズ、dpi など）はコードに直接書かず、設定ファイルか引数で渡す。

## 実験のルール

- 再現できるようにする：乱数シードを固定し、LLM は `temperature=0`。実行ごとに、設定・モデル名・評価セットのバージョン（ファイルのハッシュ）を結果と一緒に保存する。
- 評価セットのバージョンが違う結果どうしを比べない。
- ファインチューニングに使うデータは、評価セットと重複させない（データリークの防止）。
- LLM の出力（キャプション、回答）は `outputs/cache/` にキャッシュし、同じ入力で API を二度呼ばない。
- 結果の数値をもっともらしく作らない。実行していない実験の結果を書かない。

## Claude API とコスト

- API キーは `.env` から読む（Claude は `ANTHROPIC_API_KEY`、Gemini は `GEMINI_API_KEY`、DeepSeek は `DEEPSEEK_API_KEY`）。キーを表示・ログ出力・コミットしない。この節の費用のルールは Gemini と DeepSeek にも当てはめる。
- モデル ID は、予告なく中身が差し替わる呼び名（例：`deepseek-flash`）より、日付つきの固定版を優先する。使った ID は結果と一緒に保存する。
- モデル名はコードに直接書かず、設定から渡す。
- **API を50回以上呼ぶ処理は、実行前に呼び出し回数と費用の見積もりを示し、開発者の了承を得る。** まず数件で試してから全件を回す。

## GPU（VRAM 16GB）

- 大きなモデルは同時に1つだけ VRAM に載せる。使い終わったら `del model; torch.cuda.empty_cache()`。
- Ollama のモデルも VRAM を使う（qwen2.5vl:7b で約5.5GB）。PyTorch で大きなモデルを動かす前に、`ollama ps` で常駐しているモデルを確認する。
- 学習中に GPU のリセットやクラッシュが起きたら、AMD の既知の問題として `TORCH_BLAS_PREFER_HIPBLASLT=0` を試す（遅くなる場合がある）。

## データと著作権

- 決算資料の PDF やページ画像は `data/` に置き、コミットしない。
- README や results に載せる図は、自分で作ったグラフに限る。資料のページ画像そのものを載せない。
- データを Web から自動で取得するコードは、対象のサイトの利用規約を確認してから書く。

## 作業の進め方

- 1回の作業は小さく区切る。作業ごとにテストを通し、コミットメッセージ（英語・命令形）を提案する。コミットは開発者の確認後に行う。
- 新しいライブラリや手法を使うときは、AI の記憶ではなく公式ドキュメントで API を確認する（colpali-engine や ROCm は変化が速い）。
- コードを書いたら、開発者が理解できるよう、要点を日本語で短く説明する。特に、なぜその書き方にしたのか、どこが間違えやすいか。
- 開発者が「自分で書く」と言ったら、完成したコードではなく、ヒント・方針・テストケースを出す。
- コードのコメントと識別子は英語、開発者への説明は日本語。
