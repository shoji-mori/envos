# envos 開発者向けガイド (CONTRIBUTING.md)

## 1. 作業ルール(PLAN.md §0 より)

### 1タスク = 1ブランチ = 1PR
- 各タスクは独立したブランチとプルリクエストで実施する。
- PR タイトルにタスクID を含める(例: `P1-2: tools.shell の書き直し`)。
- タスク外の修正を同じ PR に混ぜない。作業中に新しい問題を見つけた場合は
  ISSUES.md に追記し、別タスクとして扱う。

### ブランチ戦略
- 統合ブランチ: **`develop`**(本計画文書・PLAN.md を含む)
- タスクブランチ: `develop` から分岐し、PR のマージ先も `develop`
  - 命名規則: `task/<タスクID>-<短い説明>`
    例: `task/P0-3-test-harness`、`task/P1-2-shell-rewrite`
- マイルストーン(M0/M1/M2)到達時に `develop` → `master` の統合 PR を出す。
- `master` への直接コミットはしない。

### PR 本文テンプレート(全タスク共通)
```
## タスク: <PLAN.md のタスクID> <タイトル>
## 対応する ISSUES: <番号列挙>
## 変更内容: <箇条書き>
## 検証: <実行したテスト・受け入れ基準の充足>
## 物理変更: なし / あり(変化量: ...、ゴールデン再生成: G*)
## 削除シンボル: なし / あり(各シンボルの git grep 結果が定義行のみであることを確認済み)
```

### 物理変更のある PR(⚠物理変更マーク)
- ゴールデンデータ(`tests/golden/`)の再生成と変化量の定量的な記録を PR 本文に必ず含める。
- `physics-change` ラベルを付ける。
- ゴールデンデータの変更と本体の変更が**同一コミット**に含まれていることを確認する。

### 削除の判断基準
削除する条件(両方満たす場合のみ):
1. リポジトリ内で呼び出し元がゼロである
2. 壊れている・到達不能・重複実装のいずれか

残す条件: 呼び出し元ゼロでも、動作しており公開 API として意図されたもの。
詳細は PLAN.md §0-9 および §2.6 を参照。

---

## 2. 開発環境の構築

### 前提
- Python 3.9 以上

### 手順

#### 1. リポジトリの取得
```bash
git clone <repo-url>
cd envos
git checkout develop
```

#### 2. 仮想環境の作成と依存パッケージのインストール
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

`[dev]` グループには `pytest` と `ruff` が含まれる。

#### 3. radmc3dPy のインストール
radmc3dPy は PyPI に存在しないため、公式リポジトリから手動インストールする。
詳細は README §4.1 を参照。

簡易手順:
```bash
git clone https://github.com/dullemond/radmc3d-2.0.git
cd radmc3d-2.0/python/radmc3dPy
pip install .
```

または一行でインストール:
```bash
pip install "git+https://github.com/dullemond/radmc3d-2.0.git#subdirectory=python/radmc3dPy"
```

#### 4. 動作確認
```bash
python -c "import envos"
```

---

## 3. テストの実行

### 通常のテスト(CI デフォルト)
```bash
pytest -m "not slow and not radmc"
```

### マーカーの説明
| マーカー | 意味 |
|---|---|
| (なし) | 常に実行。radmc3d バイナリ不要。実行時間が短い。|
| `slow` | 実行時間が長いテスト。CI 週次ジョブで実行。|
| `radmc` | `radmc3d` バイナリが必要なテスト。CI 週次ジョブで実行。|

### 遅いテストも含めて実行
```bash
pytest -m "not radmc"
```

### radmc3d が必要なテストを含めて実行(radmc3d バイナリが PATH にある場合)
```bash
pytest -m "radmc or slow"
```

### 全テスト
```bash
pytest
```

---

## 4. ゴールデンデータの再生成

ゴールデンデータ(`tests/golden/`)は、物理結果の正しさを担保する回帰テスト用の
基準値ファイルである。⚠物理変更を伴う PR では必ず再生成し、変化量を PR 本文に記録する。

### 再生成手順
**注意: P0-4 が完了してからこのスクリプトは有効になる。**

```bash
python tests/make_golden.py
```

再生成後、変更されたファイルを確認:
```bash
git diff tests/golden/
```

変化量をPR本文に定量的に記録すること。詳細は `tests/make_golden.py` の
docstring に記載されている。

---

## 5. コードスタイル

linter として `ruff` を使用する。設定は `ruff.toml` を参照。

```bash
ruff check envos/
```

CI の `lint` ジョブと同じチェックがローカルでも実行できる。

---

## 6. タスクの優先順位と依存関係

詳細は PLAN.md を参照。基本的な順序:
1. P0-1〜P0-6(安全網の構築) → M0
2. P1-*(バグ修正・デッドコード削除) → M1
3. P2-*(構造的再設計) → M2

P1-2(tools.shell 書き直し)は最優先。P1-5 → P1-4 の順序制約あり。
各タスクの「依存」欄を必ず確認してから着手すること。
