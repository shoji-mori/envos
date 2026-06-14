# envos リファクタリング作業計画書(PLAN.md)

- **版**: v3.0-draft(Phase 0〜2 完了・v2.0.0 到達。Phase 3 以降の継続計画を §6.5 に追記。改訂履歴と検証ログは§10-11)
- **対象リポジトリ**: shoji-mori/envos
- **前提文書**: `ISSUES.md`(コードレビュー結果70件+追補。本計画のタスクは原則 ISSUES.md の項目に対応。対応表は付録B)
- **想定読者**: 本リポジトリを初めて触る開発者。Python・科学計算の経験はあるが envos の内部構造は知らない人。

---

## 0. この文書の使い方と作業ルール

1. **タスクには依存関係がある。** 各タスクの「依存」欄を満たしてから着手すること。
2. **1タスク = 1ブランチ = 1PR** を原則とする。PRタイトルにタスクID(例: `P1-11`)を含める。
   **ブランチ戦略**: 統合ブランチ **`develop`**(本計画文書を含む)を起点とする。タスクブランチは `develop` から分岐し(命名: `task/P0-1-packaging` のように `task/<タスクID>-<短い説明>`)、PR のマージ先も `develop`。マイルストーン(M0/M1/M2)到達時に `develop` → `master` の統合PRを出す。`master` への直接コミットはしない。
3. **タスク外の修正を同じPRに混ぜない。** 作業中に新しい問題を見つけたら ISSUES.md に追記し、別タスクとして扱う。
4. **物理結果が変わる修正(「⚠物理変更」マーク)は、ゴールデンデータ(P0-4)の再生成と変化量の定量的な記録をPR本文に必ず含める。** PRに `physics-change` ラベルを付ける。
5. 各タスクの「受け入れ基準」を全て満たしたら完了。判断に迷ったら「§9 未決事項」(D1〜D9)の推奨案に従う。
6. テストは `pytest`。`radmc3d` バイナリが必要なテストは `@pytest.mark.radmc`、遅いテストは `@pytest.mark.slow` を付け、CIデフォルトジョブから除外する。
7. 本文中の行番号は **2026-06-12 時点の HEAD(コミット 9cd3bac 系列)** のもの。先行タスクの完了で行番号はずれるため、**位置の特定は必ず引用されているコード断片(アンカー)の grep で行う**こと(例: P1-3 なら `git grep -n "def set_radmcdir"`)。行番号は補助情報にすぎない。
8. **PR本文テンプレート**(全タスク共通):
   ```
   ## タスク: <PLAN.md のタスクID> <タイトル>
   ## 対応する ISSUES: <番号列挙>
   ## 変更内容: <箇条書き>
   ## 検証: <実行したテスト・受け入れ基準の充足>
   ## 物理変更: なし / あり(変化量: ...、ゴールデン再生成: G*)
   ## 削除シンボル: なし / あり(各シンボルの git grep 結果が定義行のみであることを確認済み)
   ```
9. **削除の判断基準**(本計画全体で統一):
   - 削除する: (a) リポジトリ内で呼び出し元ゼロ、**かつ** (b) 壊れている・到達不能・重複実装のいずれか。
   - 残す: 呼び出し元ゼロでも、動作しており公開APIとして意図されたもの(§2.6 の一覧)。修正コストが極小なら直す。
10. **ロールバック方針**: 各タスクは独立PRなので revert 単位もPR。⚠物理変更PRの revert 時はゴールデンデータも同時に戻すこと(`tests/golden/` の変更が同一コミットに含まれていることを保証する)。

---

## 1. ゴールと非ゴール

### ゴール
- **Phase 0**: 安全網の構築。現代の依存環境でテストが回り、リグレッションを検出できる状態。
- **Phase 1**: ISSUES.md の A・B ランク全件と C ランクの大部分の修正、デッドコード削除、examples の復旧。
- **Phase 2**: 構造的な再設計4件 — (A) パス管理、(B) obs.py 分割、(C) ロギング簡素化、(D) FITS I/O 再構築。

### 非ゴール
- 物理モデル(`tsc.py` の方程式、`models.py` の Ulrich/CM 解、`cubicsolver.py`)の変更。検証済み資産として挙動を固定する。
- 新機能の追加。
- `mori2023.py` の全面改修(研究再現スクリプト。最小限の修正に留める。P1-20)。
- 性能最適化(明らかなリーク修正を除く)。

---

## 2. 現状の事実(計画の根拠。全て grep / 通読で確認済み)

### 2.1 依存パッケージ
- **必須**: `numpy`, `scipy`, `pandas`, `matplotlib`, `astropy`, `radmc3dPy`, `scikit-image`(`plot_funcs.py:10`)
- **遅延import(オプション)**: `joblib`(`tools.savefile`, `obs.save`)、`psutil`(`tools.show_used_memory`)
- **注意**: `radmc3dPy` と `scikit-image` は `envos/__init__.py` の import 連鎖で必ず読み込まれる。これらが無いと `import envos` 自体が失敗する。README の Requirements に `scikit-image` の記載がない。
- **外部バイナリ**: `radmc3d`(`calc_thermal_structure` / `observe_line` / `observe_cont` の実行時のみ。モデル生成・解析・プロットには不要)

### 2.2 パッケージング
- `setup.py` / `pyproject.toml` は存在しない(`.gitignore` の `envos.egg-info/` から過去には存在したと推定)。現状は「リポジトリ直下で実行」方式。
- `storage/`(オパシティ・分子データ)は**リポジトリ直下**にあり、パッケージ(`envos/`)の外。`gpath.storage_dir` のデフォルトは `Path(__file__).parents[1]/"storage"` で、**editable install またはリポジトリ内実行でしか解決しない**(D9)。

### 2.3 互換性を守る公開API
付録Aに全リスト。特に: `mori2023.py` は `envos.gpath.run_dir`(303行)と `envos.gpath.home_dir`(38-39行)、`envos.log.update_logfile()`(322行)、`envos.tools.clean_radmcdir()`(449行)を直接使用。

### 2.4 グローバル状態(gpath)の使用箇所 — P2-A の影響範囲(26箇所)

| ファイル | 行 | 内容 |
|---|---|---|
| `envos/config.py` | 334, 337, 340, 343, 347 | `__post_init__` が gpath を書き換える(副作用の発生源) |
| `envos/tools.py` | 27, 61, 70 | `savefile` デフォルト保存先、`clean_radmcdir` |
| `envos/log.py` | 119 | `set_logfile` のデフォルトログパス |
| `envos/models.py` | 9(61, 69 で使用) | `from .gpath import run_dir`(import時に値を束縛。A-9) |
| `envos/radmc3d.py` | 55-63, 482 | `set_dirs` フォールバック、`kappa.save` |
| `envos/obs.py` | 66, 1047, 1351 | `ObsSimulator.__init__`、`BaseObsData.save`、`save_fits` |
| `envos/tsc.py` | 295, 359, 487, 509-519 | TSC解テーブルの保存/読込先、図の出力先 |
| `envos/plot_tools/plot_funcs.py` | 877-878 | `savefig` 出力先 |
| `envos/plot_tools/obs_output.py` | 249 | `plot_lineprofile` 出力先 |
| `envos/streamline.py` | 88-90 | `global run_dir`(未定義参照。B-23) |

### 2.5 デッドコード一覧(呼び出し元ゼロを grep で確認済み — P1-12 の対象)
- `envos/obs.py`: `observe_line_profile`(242-283、本体が文字列リテラル)、`set_radmc_input`(117-122)、`convolve_image`(617-624)、`find_proper_nthread`(385-387)、`__main__` ブロック(1778-1787、個人環境のパスがハードコード)
- `envos/grid.py`: `get_interface_coord`(133-163、B-27のバグはここ)、`compressed_x2`(166-186)、`Grid.calc_interface_coord` 内の `thax_ver==2/3` 分岐(103-113。`thax_ver=1` 固定のため到達不能)
- `envos/tsc.py`: `make_function_loglog`(410-423)
- `envos/log.py`: `DebugFormatter`(204-221)、`color`(167-181)— P2-C で削除
- `envos/header.py`: モジュール全体(どこからも import されていない。Python2 向けバージョンチェックの残骸。C-62 はこの削除でクローズ)
- `envos/model_generator.py`: `read_model` の到達不能 `return`(295)
- `envos/physical_params.py`: `calc_dependent_params`(6-52。`PhysicalParameters` と重複したモジュール関数で呼び出し元ゼロ、かつ B-26 のバグ持ち → §0-9 の基準により削除対象。P1-6)

### 2.6 リポジトリ内未使用だが「残す」公開メソッド(§0-9 基準の (b) を満たさないもの)
観測データ解析用の公開APIとして意図されたと推定されるため**削除しない**。ただしテスト整備の優先度は低く、修正タスクの対象箇所以外はテストを書かなくてよい:
`BaseObsData` の `mask / reverse_ax / reversed_ax_data / move_position / move_center / get_Imax_pos / ra / dec / radec / freq / set_coord_from_radec*`、`Image.convert_perbeam_to_perpixel`、`CircumstellarModel.calc_midplane_average`(B-28 で1行修正して残す)、`tools.find_roots / x_cross_zero / make_array_center / make_array_interface / compute_object_size / show_used_memory`、`obs.read_image_fits / read_pv_fits / read_cube_fits`(P2-D の対象)、`datacor.calc_datacor`(外部利用想定)。

---

## 3. 全体戦略

### 3.1 リグレッション防止
1. **ゴールデンデータ回帰テスト**を最初に作る(P0-4)。radmc3d 不要の小グリッド UCM モデルの数値をコミットし、全PRで照合。
2. 物理を変えるPRは、ゴールデンを**同じPRで再生成**し、変化率を本文に記録(`physics-change` ラベル)。
3. radmc3d が必要なパイプラインは CI のオプショナルジョブ(P0-5)。

### 3.2 フェーズの順序とマイルストーン
```
P0-1 → P0-2 → P0-3 → P0-4 → P0-5 → P0-6        [M0: 安全網完成]
→ P1-2(最優先)→ P1-5 → P1-4 → 残りのP1-*(並列可) [M1: 既知バグ解消・examples復旧]
→ P2-A → P2-C → P2-B → P2-D                     [M2: 再設計完了]
```
P2 実施順は **A → C → B → D**。理由: A は B/C/D の前提となるシグネチャを確定させる。C は A が廃止する `change_rundir` に依存するため A の直後。B は意味を変えない移動なので C の後に安全に実施。D は外部仕様の決定(D4)を伴うため最後。

### 3.3 工数の目安(envos 未経験の開発者を想定)
| フェーズ | 目安 |
|---|---|
| Phase 0 | 3〜5人日(radmc3d の CI ビルドが難航すると +2日) |
| Phase 1 | 5〜8人日 |
| Phase 2-A | 3〜4人日 |
| Phase 2-B | 1〜2人日 |
| Phase 2-C | 1〜2人日 |
| Phase 2-D | 4〜6人日 |

### 3.4 バージョニング方針
- 現行 `__version__ = "1.0.0"` を維持したまま M0/M1 を進める。
- **M1 完了時に `v1.1.0` をタグ**(後方互換のバグ修正リリース。デッドコード削除は §0-9 基準のもののみ)。
- **M2 完了時に `v2.0.0` をタグ**(`tools.savefile` 等のシグネチャ変更、gpath/update_logfile の Deprecation を含むため)。Deprecation シムは v2.x の間維持し、v3 で削除。
- `pyproject.toml` と `envos/__init__.py` のバージョンは P0-1 で単一情報源化(`[project] dynamic = ["version"]` + `attr: envos.__version__`)。

### 3.5 テストファイル台帳(タスクとの対応)
| テストファイル | 作成タスク | 主な利用タスク |
|---|---|---|
| `tests/conftest.py`(Agg設定、G1モデルの session フィクスチャ) | P0-3 | 全タスク |
| `tests/test_smoke.py` | P0-3 | 全タスク |
| `tests/test_golden.py` + `tests/make_golden.py` + `tests/golden/` | P0-4 | P1-1, P1-9, 全⚠タスク |
| `tests/test_tools.py`(shell/filecopy/dataclass_str) | P1-2 | P2-A |
| `tests/test_gpath.py` → P2-A で `tests/test_paths.py` に改組 | P1-3 | P2-A |
| `tests/test_config.py`(logfile/level/`from envos import *`) | P1-4 | P2-C |
| `tests/test_log.py`(ハンドラ非増殖・レベル) | P1-5 | P2-C |
| `tests/test_physical_params.py` | P1-6 | — |
| `tests/test_grid.py` | P1-7 | — |
| `tests/test_models.py`(save先・midplane average) | P1-8 | P2-A |
| `tests/test_model_generator.py`(f_dg・disk合成 G4) | P1-9 | — |
| `tests/test_radmc3d.py`(入力ファイル生成。radmc3d 実行は不要な範囲) | P1-10 | P2-A |
| `tests/test_obsdata.py`(合成 Cube/Image/PVmap の操作系) | P1-11 | P2-B, P2-D |
| `tests/test_streamline.py` | P1-13 | P2-A |
| `tests/test_column_density.py` | P1-14 | — |
| `tests/test_datacor.py` | P1-15 | — |
| `tests/test_plot.py`(Agg 完走系) | P1-16〜18 | — |
| `tests/test_examples.py`(slow/radmc) | P1-19 | M1 確認 |
| `tests/test_paths.py`(多重 run_dir) | P2-A | P2-B〜D |
| `tests/test_pickle_compat.py` + `tests/data/cube_legacy.pkl` | P2-B | — |
| `tests/test_fits_io.py` + `tests/data/`(合成FITS) | P2-D | — |

---

## 4. Phase 0: 安全網の構築

### P0-1: パッケージングとインストール手順の整備
- **依存**: なし
- **作業**:
  1. `pyproject.toml` を新規作成(setuptools backend)。`name="envos"`、バージョンは `dynamic = ["version"]` + `[tool.setuptools.dynamic] version = {attr = "envos.__version__"}` で `envos/__init__.py:25` を単一情報源に(§3.4)。`requires-python=">=3.9"`(D3)。
  2. dependencies: `numpy`, `scipy>=1.6`, `pandas`, `matplotlib`, `astropy`, `scikit-image`。
     `radmc3dPy` は PyPI に無いため dependencies に**入れず**、README に手順を明記(既存 README §4.1 を維持)。
  3. `[project.optional-dependencies]`: `dev = ["pytest", "ruff"]`、`extra = ["joblib", "psutil"]`。
  4. README の Requirements に `scikit-image` を追記。
  5. **インストール方式は editable(`pip install -e .`)を標準とする**(`storage/` の解決が package 親ディレクトリ依存のため。通常インストール対応は D9 / P2-A-1 で扱う)。
- **受け入れ基準**: クリーンな venv で `pip install -e .[dev]` 成功。radmc3dPy を入れた上で `python -c "import envos"` 成功。

### P0-2: 現代環境互換修正(A-10 + A-7 の繰り上げ)
- **依存**: P0-1
- **A-7 の繰り上げ(Python 3.11 対応)**: Python 3.11 以降、dataclass は unhashable なデフォルト値を `ValueError` で拒否するため、`refpos: RefPos = RefPos()`(`obs.py` 1153, 1268, 1316 行)は **`import envos` 自体を失敗させる**(実環境で確認済み。§11 検証パス6)。よって P1-11 項5 を本タスクに繰り上げ、3箇所を `dataclasses.field(default_factory=RefPos)` に変更する。この1修正のみで Python 3.11 + 最新依存での import が通ることは確認済み。
- **修正**(機械的API改名。数値不変):
  | ファイル:行 | 旧 | 新 |
  |---|---|---|
  | `envos/tsc.py:92` | `integrate.cumtrapz(y, x, initial=0)` | `integrate.cumulative_trapezoid(y, x, initial=0)` |
  | `envos/tsc.py:345` | `integrate.simps(y, x)` | `integrate.simpson(y, x=x)` |
  | `envos/obs.py:1192` | `integrate.simps(...)` | `integrate.simpson(..., x=self.vkms, axis=-1)` |
  | `envos/plot_tools/plot_funcs.py:405` | `integrate.simps(r, z_ax)` | `integrate.simpson(r, x=z_ax)` |
- **受け入れ基準**: SciPy >= 1.14 で `import envos` と P0-3 スモークが通る。

### P0-3: テストハーネスとスモークテスト
- **依存**: P0-2
- **作業**:
  1. `tests/`、`tests/conftest.py`(冒頭で `matplotlib.use("Agg")`)、pyproject に pytest 設定(`markers = ["slow", "radmc"]`)。
  2. `tests/test_smoke.py`:
     - `import envos` 成功。
     - `envos.Config(run_dir=tmp_path, rau_in=10, rau_out=100, dr_to_r=0.1, CR_au=100, Ms_Msun=0.3, T=10)` 生成。
     - 上記 config で `ModelGenerator` → `calc_kinematic_structure()` → `get_model()`。`model.rhogas` が有限・非負・非ゼロ。
     - `Grid(rau_lim=[10,100], dr_to_r=0.1)` 単体生成。
     - `PhysicalParameters` の3通りの入力(T+CR+Ms / T+t_yr+Omega / Mdot_smpy+Ms+jmid)で属性が埋まる。
       ※ `T+t_yr` の組合せは B-26 類似のバグが `calc_Ms` には無いことを確認済み(`physical_params.py:98-100` で `Ms` を正しく定義している)。
  3. **既知バグでまだ書けないテストは `xfail(strict=True)` で先に書く**(例: `from envos import *` は P1-4 まで xfail。修正PRで xfail を外す)。
- **受け入れ基準**: `pytest -m "not slow and not radmc"` が全パス(xfail 含む)。

### P0-4: ゴールデンデータ回帰テスト
- **依存**: P0-3
- **作業**:
  1. `tests/golden/` と再生成スクリプト `tests/make_golden.py` を作成。保存対象:
     - **G1(UCM、radmc3d不要)**: `rau_in=10, rau_out=1000, nr=30, ntheta=20, nphi=1, CR_au=100, Ms_Msun=0.3, T=10, cavangle_deg=45` の `rhogas, vr, vt, vp, rc_ax, tc_ax` を `.npz` で。
     - **G2(UCM+TSC)**: G1+`outenv="TSC"`。`@pytest.mark.slow`。TSC解テーブル(`tscsol.pkl`)は `tests/golden/` に同梱し、テストでは `storage_dir` をそこへ向けて解き直しを回避する。
     - **G3(ppar スカラー値、JSON)**: (a) `PhysicalParameters(T=10, CR_au=100, Ms_Msun=0.3)`、(b) `PhysicalParameters(Mdot_smpy=4.5e-6, Ms_Msun=0.2, CR_au=200)` の `Mdot, cs, t, CR, jmid, Omega`。
       ※ (b) は `nc.smpy` を経由するため **P1-1(year修正)の影響を捕捉できる唯一のケース**。(a) は year に依存せず不変のはず — P1-1 のPRでこの予言が成り立つことも確認する。
  2. `tests/test_golden.py`: 再計算して `np.testing.assert_allclose(rtol=1e-10)`。
  3. ゴールデンは **P0-2 適用済みの HEAD** で生成してコミット。
- **受け入れ基準**: 再実行で全一致。`make_golden.py` の docstring に再生成手順と記録ルールを明記。

### P0-5: CI(GitHub Actions)
- **依存**: P0-3
- **作業**: `.github/workflows/ci.yml`:
  - **lint**: `ruff check envos/`(ルールは `E9, F63, F7, F82`(構文・未定義名)から開始)。Phase 1 完了前は既知の F821(`streamline.py` の `run_dir` 等)が残るため、`pyproject.toml` の `[tool.ruff] per-file-ignores` に該当ファイルを列挙して除外し、**修正タスクのPRで除外を1つずつ削る**。
  - **test**: Python 3.9 / 3.11 / 3.12 マトリクス。`pip install -e .[dev]` + radmc3dPy(`pip install "git+https://github.com/dullemond/radmc3d-2.0.git#subdirectory=python/radmc3dPy"` — **要動作確認**。失敗時は clone して `pip install ./radmc3d-2.0/python/radmc3dPy`)+ `pytest -m "not slow and not radmc"`。
  - **test-full**(週次 or 手動): gfortran で radmc3d をビルドし `pytest -m "radmc or slow"`。
- **受け入れ基準**: PR で lint+test が自動実行されグリーン。

### P0-6: 運用文書の整備
- **依存**: P0-1〜P0-5
- **作業**: `CONTRIBUTING.md`(本書§0、テスト実行方法、ゴールデン再生成手順)。ISSUES.md に対応状況列(タスクID/チェックボックス)を追加。
- **受け入れ基準**: 新規参加者が CONTRIBUTING.md のみで開発環境を再現できる。

---

## 5. Phase 1: バグ修正とデッドコード削除

**P1-2 を最初に**(以降の radmc3d 系作業の信頼性に直結)。P1-5 → P1-4 の順序制約あり。他は並列可。末尾括弧は ISSUES.md 項目番号。

### リスクと規模の一覧
リスク = 修正が新たな不具合や挙動変化を生む可能性。規模 = 想定 diff 行数(テスト除く)。

| タスク | リスク | 規模 | リスクの根拠 |
|---|---|---|---|
| P1-1 | 中 | ~5行 | ⚠物理変更(0.33%)。ゴールデン再生成を伴う |
| P1-2 | **高** | ~80行 | shell() は radmc3d 実行の要。書き直しのため radmc 環境での手動確認必須(受け入れ基準に含む) |
| P1-3 | 低 | 2行 | — |
| P1-4 | 低 | ~20行 | P1-5 完了が前提 |
| P1-5 | 中 | ~40行 | ロギングは全モジュールに波及。テストで非増殖を担保 |
| P1-6 | 低 | ~40行 | 削除+ガード追加のみ |
| P1-7 | 低 | ~15行 | 例外化のみ(正常系の数値は不変) |
| P1-8 | 低 | ~15行 | — |
| P1-9 | 中 | ~25行 | ⚠disk 使用時の物理変更(D1)。G4 新設 |
| P1-10 | 中 | ~40行 | radmc3d 入力ファイル生成に触れる。ファイル内容のバイト一致テストで担保 |
| P1-11 | **高** | ~120行 | 観測データクラスの中核。項目が多いため**コミットを項目単位に分割**すること |
| P1-12 | 低 | -400行 | 削除のみ(§0-9 の基準と grep 確認に従う) |
| P1-13 | 低 | ~30行 | mirror 削除は D5 で承認済み |
| P1-14 | 低 | ~15行 | — |
| P1-15 | 低 | ~10行 | — |
| P1-16 | 中 | ~40行 | ピーク探索はプロットの数値出力(質量推定)に影響。合成データで定量検証 |
| P1-17 | 低 | ~15行 | 図のみの変化 |
| P1-18 | 低 | ~10行 | — |
| P1-19 | 低 | ~60行 | examples のみ |
| P1-20 | 低 | ~20行 | 研究スクリプトのガードのみ |

### P1-1: 物理定数の修正 ⚠物理変更(A-1)
- **対象**: `envos/nconst.py:28-29`
- **修正**: `year = yr = 3.15576e7`(ユリウス年 365.25 日)。`Myr = 3.15576e13`。出典コメントを付す。
- **やらないこと**: `Lsun = 3.848e33`(IAU公称値 3.828e33 と不一致)は**変更しない**(D7)。コメントで注記のみ。
- **ゴールデン**: G3(b) が変わる(`smpy` 定義変更。変化率 ≈ 0.33%)。G3(a)・G1 は不変のはず。両方を確認し、G3(b) を再生成。変化率をPR本文に記録。
- **受け入れ基準**: `nc.year / 86400 == pytest.approx(365.25)`。G1・G3(a) 不変、G3(b) 更新済み。

### P1-2: `tools.shell` の書き直しと `tools.py` 修正(A-2, C-42, C-43, D-68, 追補-75)
- **対象**: `envos/tools.py`、`envos/obs.py:363`
- **修正**:
  1. `shell()`(236-313行)を**単一の Popen ベース実装に書き直す**(現状の `simple=1` ハードコードによる二重実装を解消):
     - `subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=PIPE, stderr=STDOUT, universal_newlines=True)` で行単位に読み、`log=True` なら `logger.info(log_prefix + line)` に流す(radmc3d の長時間実行で進捗が見える状態を維持する)。
     - `error_keyword`(str または list)にマッチする行があれば記録。冒頭に `import re` を追加(現状未import)。
     - 終了後、`returncode != 0` または keyword ヒットで `CalledProcessError` を送出(`skip_error=True` なら警告のみ)。
     - **`simple` 引数は削除**(リポジトリ内の呼び出し元4箇所 — `obs.py:229,363,419`, `radmc3d.py:378` — はいずれも渡していないことを確認済み)。`dryrun` は維持。
     - 戻り値は `None` でよい(呼び出し元4箇所とも戻り値を使っていないことを確認済み)。
     - デバッグ残骸 `print("line:", _line)`(290行)を削除。
  2. `obs.py:363`(`observe_line` の単スレッド経路)に `error_keyword="ERROR"` を追加(現状エラーチェック無しの唯一の radmc3d 呼び出し)。
  3. `filecopy()`(316-353行): コピー先が存在し `error_already_exist=False` のとき、警告ログ直後に `return` を追加。`logger.warn` → `logger.warning`。
  4. `dataclass_str()`(217行): リスト分岐直後の `if isinstance(v, np.ndarray)` を `elif` に。
  5. `savefile()`(31行): `mkdir(parents=True, exist_ok=True)`。
- **テスト**: `shell("false")` が例外。`shell("echo ERROR x", error_keyword="ERROR")` が例外。`shell("echo ok")` が正常終了しログに "ok"。`filecopy` の既存先スキップ。
- **受け入れ基準**: テストパス。`grep -n "simple" envos/tools.py` が(変数として)ヒットしない。**さらに、radmc3d のある環境(ローカルまたは CI の test-full ジョブ)で `calc_thermal_structure` のスモークを1回実行し、書き直した `shell()` 経由の radmc3d 実行・ログ転送・エラー検出が機能することを確認する**(本タスクはリスク「高」のため実機確認を必須とする)。

### P1-3: `gpath.set_radmcdir` の修正(B-11)
- **対象**: `envos/gpath.py:34-37`
- **修正**: `global run_dir; run_dir = ...` → `global radmc_dir; radmc_dir = Path(path)`。
- **テスト**: `set_radmcdir("/tmp/x")` 後に `gpath.radmc_dir == Path("/tmp/x")` かつ `gpath.run_dir` 不変。
- **備考**: P2-A で gpath は縮退するが、それまでの正しさのため修正(5分)。

### P1-4: `config.py` と `envos/__init__.py` の修正(B-12, B-13, D-63, 追補-72)
- **依存**: P1-5(`set_level` の修正を先に)
- **対象**: `envos/config.py`, `envos/__init__.py`
- **修正**:
  1. `config.py:348` `set_logfile("on")` → `update_logfile()`(name デフォルト "envos"、書き込み先は直前の行で設定済みの `gpath.logfile`)。`set_logfile` ではなく `update_logfile` を使う理由: Config を複数回生成してもファイルハンドラが重複しない(unset→set。前提として P1-5 の `unset_logfile` 修正が必要)。`config.py:7` の import も `update_logfile` に変更。
  2. `molabun: float = ""`(277行)→ `= None`。`RadmcController.set_lineobs_inpfiles`(`radmc3d.py:202`)冒頭に `if self.molabun is None: raise ValueError("molabun is not set")`。
  3. `nphot: int = 1e6`(269行)→ `= 1_000_000`。
  4. docstring 140行: `disk` のオプション `"exptail"` → `"powerlaw"`(`model_generator.py:248` の実装に一致させる)。`rot_ccw`(142-143行)の説明に「**現状未実装**(設定しても効果なし。D8)」を追記。
  5. **`level_stdout` / `level_logfile`(227-228行)の配線**(追補-72: 現状どこからも読まれていない): `__post_init__` に
     `if self.level_stdout is not None: log.set_level("envos", self.level_stdout, target="stream")`(`level_logfile` は `target="file"`。file handler 追加後に呼ぶ順序に注意)を追加。
  6. `envos/__init__.py`: `__all__` から `read_mg` を削除。`from . import column_density` を追加(`__all__` の `"column_density"` を実体化)。
- **テスト**: `Config(run_dir=tmp, logfile=tmp/"log.txt")` がクラッシュせずログファイルが作られる。`Config(level_stdout="DEBUG")` 後に stream handler が DEBUG。`from envos import *` が成功(P0-3 の xfail を外す)。
- **受け入れ基準**: テストパス。

### P1-5: `log.py` の最小修正(B-29, B-30, D-65)
- **対象**: `envos/log.py`(全面再設計は P2-C。ここでは壊れている関数のみ)
- **修正**:
  1. `unset_logfile(name)`(123-129行): `logger.removeHandler` → `loggers[name].removeHandler`、`for hdlr in list(loggers[name].handlers):` とコピーへの反復に。remove の前に `hdlr.close()` を呼ぶ(ファイルディスクリプタのリーク防止。`update_logfile` が繰り返し呼ばれるため)。
  2. `set_level`(133-156行): `ver` 引数と `ver==1` 分岐(グローバル `logger` 誤用)を削除。
  3. `change_rundir`(71-83行): 現実装は無効(str.replace の戻り値破棄)。書き直す: 各 FileHandler について `close()` → remove → `new_rundir / Path(旧baseFilename).name` で `add_file_hdlr(logger, 新パス, level=旧level, write_mode="a")`。
     ※ 唯一の呼び出し元は `gpath.update_all_dirs_dependent_on_rundir`(gpath.py:67)。
  4. `StandardFormatter.__init__`(186行): フォーマット末尾のデバッグ残骸 `" is this used? "` を削除。
- **テスト**: `update_logfile()` を2回呼んでもハンドラが増殖しない。`set_level("envos","DEBUG")` で stream handler が DEBUG。`Config(run_dir=A)` 後にログファイル設定→ `Config(run_dir=B)` でログ出力先が B に移る。
- **受け入れ基準**: テストパス。`mori2023.py:322` 相当の `envos.log.update_logfile()` が動く。

### P1-6: `physical_params.py` の整理(B-26)
- **対象**: `envos/physical_params.py`
- **修正**:
  1. **`calc_dependent_params`(6-52行)を削除**(検証パス4で呼び出し元ゼロを確認。`PhysicalParameters` と完全に重複したモジュール関数であり、`t_yr` 経路の `Ms` 未定義バグ(B-26)を抱えたまま放置されてきた。§0-9 の基準 (a)+(b) を満たす。B-26 は削除でクローズ)。
  2. `PhysicalParameters.calc_Mdot / calc_Ms / calc_jmid`(81-124行)に、どの分岐にも入らなかった場合の `ValueError("insufficient parameters: specify one of (T, Mdot_smpy) / (Ms_Msun, t_yr) / (CR_au, jmid, Omega, rexp_au+Omega)")` ガードを追加(現状は `UnboundLocalError` / `AttributeError` で不親切)。
- **テスト**: `PhysicalParameters(T=10, t_yr=1e5, CR_au=100)` が全属性非None。`PhysicalParameters()` / `PhysicalParameters(T=10)`(不足)が `ValueError`。
- **受け入れ基準**: G3 不変。`git grep calc_dependent_params` が0件。

### P1-7: `grid.py` の修正(C-50, C-51。B-27 は P1-12 でデッドコードごと削除)
- **対象**: `envos/grid.py`
- **修正**:
  1. `Grid.__init__`(44-45行): `return None` → `raise ValueError("Grid requires either (ri_ax, ti_ax, pi_ax) or rau_lim")`。
  2. `Grid.calc_interface_coord`: `nr=None` かつ `dr_to_r=None` なら `ValueError("Specify either nr or dr_to_r")`。`ntheta` も同様。
  3. ringhost(47-51行): ログ文言を実態に合わせ「Adding 3 ghost cells」に。`1.001 * 4 * nc.Rsun` は挙動維持し、「TODO: Rstar_Rsun と連動していない」コメントを付す。
- **テスト**: 引数不足の `Grid()` が `ValueError`。`Config(nr未指定, dr_to_r未指定)` → `ModelGenerator` が `ValueError`。ringhost=True で `len(ri_ax)` が +3 されている。
- **受け入れ基準**: G1 不変。

### P1-8: `models.py` の修正(A-9, B-28, C-53の一部)
- **対象**: `envos/models.py`
- **修正**:
  1. 9行 `from .gpath import run_dir` → `from . import gpath`。61・69行の `run_dir` → `gpath.run_dir`(呼び出し時点の値を参照。恒久対応は P2-A)。
  2. `calc_midplane_average`(203-207行): `self.take_midplane_average(_vn)` → `self.get_midplane_profile(_vn)`(72-79行の既存メソッド)。※リポジトリ内未使用だが §2.6 の方針(動く公開APIは残す)により1行修正で残す。回帰保護のテストを追加。
  3. `PowerlawDisk.get_Sigma`(359-366行): `tail` が `"exp"`/`"cut"` 以外なら `ValueError`。
  4. `CassenMoosmanInnerEnvelope.calc_kinematic_structure` 末尾: `if np.isnan(self.rho).any(): logger.warning("NaN in inner-envelope density (no cubic solution at some cells)")` を追加(挙動は維持、可視化のみ)。
  5. 371-378行の `print(...)` 2箇所を `logger.info` に。
- **テスト**: `Config(run_dir=A)` 生成→`Config(run_dir=B)` 生成→`save_pickle("m.pkl")` が B に保存される。`calc_midplane_average()` が例外なく `rhogas_mid` 等を設定する。
- **受け入れ基準**: G1 不変。

### P1-9: `model_generator.py` の修正 ⚠物理変更の可能性(A-8, C-48, B-16の一部)
- **対象**: `envos/model_generator.py`
- **修正**:
  1. `__init__` に `self.f_dg = None` を追加。`calc_kinematic_structure` で未設定なら `f_dg=0.01` を警告つきで使用(C-48)。
  2. **ディスク合成(182-190行)を置換方式に統一(D1)**: `rho += self.disk.rho` を削除し、`rho[cond] = self.disk.rho[cond]` に変更(`cond = rho < self.disk.rho`)。速度の扱い(cond 領域のみ置換)は現状維持。⚠ `disk="powerlaw"` 使用時のみ結果が変わる。**本PRでディスク付きゴールデン G4(G1+`disk="powerlaw"`)を新規追加・生成**する。
  3. `read_model`(290-295行): `raise Exception("Still constructing...")` → `NotImplementedError("only .pkl is supported")`。到達不能 `return`(295行)を削除。
  4. `set_disk` 内 `print(config)`(258行)を `logger.info` に。
  5. **C-49(TSCスムージングの密度/速度の非整合、166-171行)は修正しない(D10)**: `calc_kinematic_structure` の docstring に「smoothing_TSC=True ではキャビティ内(rho==0)で速度のみ TSC とブレンドされる既知の非整合がある」と明記するに留める(物理コードの挙動固定の原則)。
- **テスト**: config 無しの手動組み立て(`set_grid`→`set_physical_parameters`→`set_inenv` 相当)で `calc_kinematic_structure()` が通る。`disk="powerlaw"` 付きスモーク。
- **受け入れ基準**: G1 不変。G4 新規追加。

### P1-10: `radmc3d.py` の修正(B-14, B-15, B-16, C-57, C-58, D6)
- **対象**: `envos/radmc3d.py`
- **修正**:
  1. `set_model`(97-103行):
     ```python
     if isinstance(model, str):
         if not os.path.isfile(model):
             raise FileNotFoundError(model)
         self.model = pd.read_pickle(model)
     elif model is not None:
         self.model = model
     else:
         raise ValueError("model is None")
     ```
  2. 116-121行: `exit()` を廃止。`rhodust` 判定を `if getattr(md, "rhodust", None) is not None:` に変更し、None なら `rhod = rhog * self.f_dg` + `logger.info` で通知(dataclass では `hasattr` が常に True のため)。
  3. 199-200行: `remove_file("gas_temperature.inp")` → `remove_file(os.path.join(self.radmc_dir, "gas_temperature.inp"))`。`dust_temperature.dat` も同様。
  4. `run_mctherm`(371-411行): `os.chdir` を `try/finally` で囲む。
  5. **non-LTE(D6)**: `__init__` で `config.nonlte` が truthy なら直ちに `NotImplementedError("nonlte mode is untested (lines.inp species count bug); see ISSUES C-57")` を送出。`set_lineobs_inpfiles` 内の nonlte 分岐(216-218行)と `set_numberdens_collpartners` は削除せず残す(将来の実装の足場)が、到達しない旨をコメント。
- **テスト**: `set_model(存在しないパス)` が `FileNotFoundError`。`rhodust=None` のモデルで `set_mctherm_inpfiles` が成功し `dust_density.inp` の値が `rhogas*f_dg` と一致(ファイルをパースして検証)。`Config(nonlte=1)` → `RadmcController` が `NotImplementedError`。
- **受け入れ基準**: `examples/make_userdefined_model.py` のモデル構築部(radmc3d 実行手前まで)が動く。

### P1-11: `obs.py` バグ修正(A-3〜A-7, C-45〜C-47, C-59〜C-61, 追補-71, 77)
- **対象**: `envos/obs.py`
- **修正**:
  1. **`or` パターンの除去**: 209-211行・288-292行の `incl = incl or self.incl` 等を全て `x = x if x is not None else self.x` 形式に。211行のタイポは `posang = posang if posang is not None else self.posang` に修正。`gen_radmc_cmd` 54行 `if vc_kms` → `if vc_kms is not None`。
  2. `observe_cont` 223行: `npixy=self.npixx` → `npixy=self.npixy`。
  3. `observe_line` 312行: `"iline": self.iline` → `"iline": iline`(ローカル変数)。
  4. `get_mom0_map`(1174-1200行)の vlim バグ(A-3)+ Iunit 非整合(追補-77):
     ```python
     _Ippv, _vkms = self.Ippv, self.vkms
     if vlim is not None:
         (len(vlim)==2 and vlim[0]<vlim[1] を検証、不正は ValueError)
         cond = (vlim[0] < _vkms) & (_vkms < vlim[1])
         _Ippv = _Ippv[..., cond]; _vkms = _vkms[cond]
     if method == "sum":         _Ipp = np.sum(_Ippv, axis=-1) * (_vkms[1] - _vkms[0])
     elif method == "integrate": _Ipp = integrate.simpson(_Ippv, x=_vkms, axis=-1)
     ```
     `shape[2]==1` の特例は維持。正規化は `_Ipp /= max` の直接除算をやめ、`Image` 構築後に `if normalize == "peak": img.norm_I("max")` とする(`norm_I` が `Iunit` も更新するため追補-77 が同時に解消)。⚠ vlim 指定時のみ結果が変わる(従来は黙って無視=バグ修正として記録)。
  5. ~~`refpos: RefPos = RefPos()`(1153, 1268, 1316行)→ `dataclasses.field(default_factory=RefPos)`~~ **P0-2 に繰り上げ済み**(Python 3.11 で import 不能のため)。本タスクでは refpos 共有が解消されていることのテスト((b))のみ担当。
  6. `_reset_positive_axes`(569-573行)の再設計: シグネチャを `_reset_positive_axes(self)` に変更。`for i, name in enumerate(self._axnames):` で `ax = getattr(self, name)` を取り、`ax is not None and len(ax) >= 2 and ax[1] < ax[0]` なら `setattr(self, name, ax[::-1])` + `self.set_I(np.flip(self.get_I(), axis=i))`。呼び出し元3箇所(`Cube/Image/PVmap.__post_init__`)を引数なし呼び出しに更新。
  7. `set_refpoint`(1011行): `self.refpos.dec = dec0` → `self.refpos.dec0 = dec0`。
  8. `convto_Tb`(804-806行): エラーメッセージを f-string 化、`self.repos` → `self.refpos`。
  9. `minmaxargs`(1772-1775行): 比較を `>=`/`<=` に変更し、該当なしは `ValueError(f"no points within {lim}")`。
  10. `get_pv_map` 1247行(追補-71): 存在しないメソッド `pv.save_fitsfile()` → `raise NotImplementedError`(P2-D で `save_fits` に接続して復活)。
  11. `Image.__post_init__`(1282行): InitVar 未宣言の `vkms0` 引数(常に None のデッド引数。しかも `tools.vkms_to_freq(vkms0)` は必須第2引数欠落)を削除。
  12. `_subcalc`(426行、C-59): `open(os.devnull, "w")` を `with` 文に変更してリークを解消(`with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):`)。
  13. `Convolver.__init__`(496-523行、C-60): `mode == "null"` ならカーネル構築をスキップして即 return。それ以外で `beam_maj_au` / `beam_min_au` が `None` なら `ValueError("beam size is required for convolution")`(現状は `None + 1e-100` の不親切な TypeError)。
  14. `Obreso.__post_init__`(1082-1087行、C-61)のバリデーションを厳密化。現状は**部分指定**(例: `beam_maj_au=50, beam_min_au=None`)でも au 側ペアの再計算に入り `np.deg2rad(None)` で TypeError になる。仕様を次のとおり明文化して実装:
      - `(beam_maj_au, beam_min_au)` が両方非None → deg 側を導出。
      - そうでなく `(beam_maj_deg, beam_min_deg)` が両方非None → au 側を導出。
      - どちらのペアも完全でない → `ValueError("specify both (beam_maj_au, beam_min_au) or both (beam_maj_deg, beam_min_deg); given: ...")`。
      - `set_dpc_from_obsdata` の `print("Something wrong in obsdata")`(1109行)は `logger.warning` に変更。
- **テスト**: 合成 Cube(解析的な3D配列)で (a) `get_mom0_map(vlim=...)` が範囲外チャネルを除外、(b) 2個の Cube の `refpos` が別オブジェクト、(c) 降順軸を与えた `Image` で軸とデータが整合して昇順化、(d) `trim` の往復、(e) `set_refpoint` 後に `refpos.dec0` が更新、(f) `Convolver(grid, mode="null")` がビーム未指定でも生成でき恒等変換、(g) `Obreso(ビーム情報なし)` が `ValueError`。
- **受け入れ基準**: テストパス。既存スモーク・ゴールデン不変。

### P1-12: デッドコード削除(横断)(B-17, B-18, B-20, B-27, C-62, 追補-74, 76)
- **対象**: §2.5 の一覧(grep で呼び出し元ゼロ確認済み)
- **修正**:
  1. `obs.py`: `observe_line_profile`、`set_radmc_input`、`convolve_image`、`find_proper_nthread`、`__main__` ブロック(個人パスのハードコード)を削除。
  2. `obs.py` `read_obsdata`(1429-1449行): fits 分岐 → `raise NotImplementedError("use read_cube_fits/read_image_fits/read_pv_fits")`(P2-D で再訪)。末尾の `sys.exit(1)` → `ValueError(f"cannot infer file type: {path}")`。
  3. `grid.py`: `get_interface_coord`(B-27 のバグはこの削除で同時クローズ)、`compressed_x2`、`thax_ver` 変数と `==2/3` 分岐を削除(`ti_ax = np.linspace(*theta_lim, ntheta+1)` に直書き)。
  4. `tsc.py`: `make_function_loglog` を削除。
  5. `envos/header.py` を削除(import 元ゼロ。C-62 も同時クローズ)。
  6. 大きなコメントアウト塊のうち誤解を招くもの(`obs.py:985-1006, 1252-1256`、`models.py:89-94`)を削除。
- **削除の安全確認**: 各シンボルについて削除PR内で `git grep -n <name>` の結果が定義行のみだったことをPR本文に記載(外部利用者向けの告知を兼ねる)。
- **受け入れ基準**: テスト全パス。`ruff` の per-file-ignores から該当ファイルを除去してもパス。

### P1-13: `streamline.py` の修正(B-23, B-24, B-25, D5)
- **対象**: `envos/streamline.py`
- **修正**:
  1. `Streamline.save_data`(88-90行): `global run_dir` を削除。`from . import gpath` を追加し `dpath = dpath if dpath is not None else gpath.run_dir`(恒久対応は P2-A)。
  2. `calc_streamline`(16-18行): デフォルトを `names=None, units=None, variables=None` にし、本体で `_variables = list(variables) if variables else []` とコピーしてから追記(55-56行の `+=` による共有デフォルト破壊を解消)。
  3. **ミラー対称モードを削除(D5)**: `StreamlineCalculator2.__init__` の `mirror` 引数、`self.mirror_symmetry`、`_func` 内の分岐(211-214行)。未使用かつ `vr = -self.vt_field(...)` のコピペ誤りと誤った鏡映式の二重バグのため。
  4. `print(pos0_list)`(65行)→ `logger.debug`。
- **テスト**: `calc_streamline` を同一プロセスで2回呼び、2回目の variables が累積しない。G1 モデルで `save=True, dpath=tmp_path` がファイルを出力する。
- **受け入れ基準**: `examples/trace_streamline.py` 相当の処理が通る。

### P1-14: `column_density.py` の修正(C-54, C-55)
- **対象**: `envos/column_density.py`
- **修正**:
  1. モジュールトップの `import matplotlib.pyplot as plt`(2行)と `import envos`(4行)を削除。plt は `test_column_density` と `__main__` 内の遅延 import に、`import envos` は `__main__` 内に移す。
  2. `column_density_z`(115行): ステップを `step = max(0.1 * z, zlim * 1e-4)` とし `np.arange(z, zlim, step)`(z=0 での無限ループ/即死を防止)。
- **テスト**: G1 モデルで `calc_column_density(model, "z")`(`double_interp` 両方)と `"r"`, `"theta"` が有限値を返す。
- **受け入れ基準**: `examples/trace_streamline_with_column_density.py` のモデル+列密度部分が通る。

### P1-15: `datacor.py` の修正(C-56)
- **対象**: `envos/datacor.py`
- **修正**:
  1. preprocess(46-48行): `fac1 = preprocess_func(im1, im2, axes_newgrid)` と `fac2 = preprocess_func(im2, im1, axes_newgrid)` を**両方先に計算**してから `im1 *= fac1; im2 *= fac2`(現状は変更済み im1 で im2 の係数を計算)。シグネチャは不変。
  2. 軸構築(25-30行): `_range is None` のとき `continue`(軸が脱落)→ `newax = _ax_data if _ax_user is None else _ax_user`(全範囲を採用)。
- **テスト**: 合成 Cube 2個で `ranges` 1軸のみ指定の `calc_datacor` が動き、3軸指定と整合。ZNCC 自己相関 = 1.0。
- **受け入れ基準**: テストパス。

### P1-16: `plot_funcs.py` の修正(B-34〜B-37)
- **対象**: `envos/plot_tools/plot_funcs.py`
- **修正**:
  1. `make_levels`(143-152行): `logger.debug` の複数位置引数を f-string 1引数に。
  2. `add_mass_estimate_plot`: 冒頭で `txt_Mip = ""`, `txt_Mvp = ""` を初期化し、末尾の結合を `"\n".join(filter(None, [txt_Mip, txt_Mvp]))` に(628-640行)。
  3. `get_subgrid_peaks`: (a) `find_subgrid_peak` の `optimize.minimize` 初期値を `coord_ini`(ピクセル添字)から `[xau_peak, vkms_peak]`(データ座標)に修正(751-762行)。(b) 740-747行の象限分岐の条件変数 `quadr` → `_quadr`。(c) `Peak(interpfun(*coord), ...)` の値を `float(interpfun(*coord)[0, 0])` でスカラー化。
  4. `add_peaks`(504-516行): `LocalPeak_Pax` を `zip(Ipv.T, vkms)`(速度ごとに x プロファイル)、`LocalPeak_Vax` を `zip(Ipv, xau)`(位置ごとに v プロファイル)に修正(現状は両方逆)。
- **テスト**: 合成 2D ガウシアン PV で `get_subgrid_peaks` がピーク位置を ±1セル以内で返す。`add_mass_estimate_plot(mass_ip=True, mass_vp=False)` が NameError を出さない(Agg)。
- **受け入れ基準**: テストパス。

### P1-17: `plot_tools/physical_structure.py` の修正(B-31〜B-33)
- **対象**: `envos/plot_tools/physical_structure.py`
- **修正**:
  1. 128行: `var[:, model.get_argmid, :]` → `var[:, model.get_argmid(), :]`。
  2. 321行: `Omega = model.vp / model.R,` の末尾カンマ削除。呼び出しを `plot_midplane_radial_profile(model, Omega, save_name="Omega_prof", ...)` とキーワード渡しに。
  3. 349行: `x, y = R * np.sin(tt) * [np.cos(pp), np.sin(pp)]` → `x, y = R * np.array([np.cos(pp), np.sin(pp)])`(R は既に円筒半径。sinθ の二重掛けを解消)。⚠図のみ変化(物理データ不変)。
  4. 269行: 未定義 `variable_name` 参照のデッド分岐 → `name = save_name` に簡約。
- **テスト**: G1 モデルで `plot_velocity_midplane_profile(model, midplane_average=False)`、`plot_losvelocity_midplane_map(model)`、`plot_angular_velocity_midplane_profile(model)` が Agg で完走し PDF 生成。
- **受け入れ基準**: テストパス。

### P1-18: `plot_tools/obs_output.py` の修正(B-38, B-39)
- **対象**: `envos/plot_tools/obs_output.py`
- **修正**:
  1. `_plot_image`: `_contopt` の定義(78-79行)を `if contour:` の外に移動(refimage 分岐 87行が `contour=False` でも参照)。
  2. `plot_pvdiagram`(380-383, 407-409行): `peaks` が `None`/空なら scatter をスキップ。
- **テスト**: 一様(ピークなし)PV で `plot_pvdiagram` 完走。`plot_image(im, refimage=ref, contour=False)` 完走。
- **受け入れ基準**: テストパス。

### P1-19: examples の修復(B-40)
- **対象**: `examples/*.py`, `example_run.py`
- **修正**: 存在しない関数を現行 API に置換:
  | 旧 | 新 |
  |---|---|
  | `plot_midplane_density_profile` | `plot_rhogas_midplane_profile` |
  | `plot_midplane_temperature_profile` | `plot_Tgas_midplane_profile` |
  | `plot_midplane_velocity_profile` | `plot_velocity_midplane_profile` |
  | `plot_midplane_velocity_map` | `plot_losvelocity_midplane_map` |
  | `plot_density_map` | `plot_rhogas_map` |
  | `plot_temperature_map` | `plot_Tgas_map` |
  | `plot_mom0_map(odat, pangle_deg=…, poffset_au=…)` | `plot_mom0_map(odat.get_mom0_map())` |
  - `trace_streamline_with_column_density.py` 末尾の `exit()` を削除。
  - `make_userdefined_model.py`: `calc_thermal_structure()` 前に `mg.model.set_dust_density(f_dg=config.f_dg)` を追加(P1-10 のフォールバックでも動くが明示)。
- **テスト**: `tests/test_examples.py` を新設。各 example の「モデル生成まで」を小グリッドにパッチして実行(`@pytest.mark.slow`)。radmc3d 必要部分は `@pytest.mark.radmc`。
- **受け入れ基準**: radmc3d のある環境で `python example_run.py` が完走(= M1 の最終確認)。

### P1-20: `mori2023.py` の最小ガード(B-41)
- **対象**: `mori2023.py`
- **修正**: (1) `synobs()` の PV 3ブロック(398-417行)を、`if plot_pv or plot_pv_for_mass_estimate or plot_pv_loglog:` で先に `pv = cube.get_pv_map(...)` を1回計算する形に整理。(2) `mg` 未定義経路(329-356行)に `else: raise FileNotFoundError("mg.pkl not found and calc_model=False")`。
- **受け入れ基準**: `ruff check`(F821)パス。実行検証は不要(研究スクリプト)。

---

## 6. Phase 2: 局所再設計

### P2-A: パス管理の再設計 — グローバル状態の排除(A-9, B-11, B-23, D-66, D-67 の根治)

- **依存**: Phase 1 完了(特に P1-3, P1-8, P1-13)
- **設計方針**:
  - 「パスの束」は **`Config` を単一情報源**とし、各コンポーネントへ明示的な引数として渡す。新しいクラスは増やさない。
  - `gpath` は**後方互換シム**として1リリース残す(`mori2023.py` が直接参照しているため)。シムは DeprecationWarning を出す。

#### P2-A-1: Config をパスの単一情報源にする
- `Config` に解決済みパスを返す read-only プロパティを追加: `run_path / fig_path / radmc_path / storage_path / log_path`。
- デフォルト解決規則を `Config` 内に一元化(現行 gpath の規則を踏襲): `run_dir` 未指定→`./run`、`radmc_dir`→`run/radmc`、`fig_dir`→`run/fig`、`logfile`→`run/log.dat`。
- **実装スケッチ**(プロパティは dataclass フィールドと衝突しない別名にする点が要点):
  ```python
  @property
  def run_path(self) -> Path:
      return Path(self.run_dir) if self.run_dir is not None else Path("./run")

  @property
  def radmc_path(self) -> Path:
      return Path(self.radmc_dir) if self.radmc_dir is not None else self.run_path / "radmc"

  # fig_path, log_path も同型。storage_path のみ D9 の解決規則(下記)。
  @property
  def storage_path(self) -> Path:
      if self.storage_dir is not None:
          return Path(self.storage_dir)
      legacy = Path(__file__).parents[1] / "storage"   # 旧: リポジトリ直下(1リリース維持)
      if legacy.is_dir():
          return legacy
      return importlib.resources.files("envos") / "storage"
  ```
- **storage のデフォルト(D9)**: `storage/` ディレクトリを `envos/storage/` へ移動し(`git mv`)、`pyproject.toml` の package-data に登録。デフォルト解決は `importlib.resources.files("envos") / "storage"`。これにより editable 以外のインストールでも動く。リポジトリ直下の `storage/` を参照していた既存ユーザー向けに、旧位置が存在する場合はそちらを優先するフォールバックを1リリース残す。
- ディレクトリの `mkdir` は「書き込む直前」に行う(`Config` 生成では作らない)。
- `Config.__post_init__` の gpath 書き換え(334-348行)は当面残す(P2-A-3 で削除)。

#### P2-A-2: 消費側を引数駆動に変更(§2.4 の一覧と1対1対応)
| 箇所 | 変更 |
|---|---|
| `RadmcController.set_dirs` | config から必ず受け取る。gpath フォールバック削除 |
| `tools.savefile` | `dirpath` 引数を**必須化**。それに伴い `ModelBase.save(..., dirpath=...)` と `ModelGenerator.save(dirpath=...)` のシグネチャに `dirpath` を追加し、`ModelGenerator.save` は `self.config.run_path` を渡す(config なしで組んだ場合は明示必須) |
| `tools.clean_radmcdir` | `dirpath` 引数必須化。呼び出し元 `mori2023.py:449` を更新 |
| `ModelBase.save_pickle/read_pickle` | `filepath` または `dirpath` を引数で受ける。gpath 参照(P1-8 の暫定対応)を削除 |
| `ObsSimulator.__init__` | `radmc_dir` は config / 引数からのみ。gpath フォールバック削除 |
| `BaseObsData.save` / `save_fits` | `filepath` 必須化(暗黙の run_dir 依存を廃止)。`mori2023.py:381` の `cube.save(filename=...)` を `filepath=` に更新 |
| `tsc.get_tsc / save_table / read_table` | `storage_dir` 引数を追加。`TerebeyOuterEnvelope.__init__` に `storage_dir` を追加し、`ModelGenerator.set_outenv` が config から渡す |
| `plot_tools.savefig` | `plot_tools.set_fig_dir(path)` を新設(**プロット層のみ module-default を許容**。全プロット関数への引数追加はAPI破壊が大きすぎるため)。`ModelGenerator`/`ObsSimulator` の `init_from_config` が呼ぶ |
| `log.set_logfile` | `filepath` 必須化(P2-C と同一PRでも可) |
| `streamline.Streamline.save_data` | `dpath` 必須化。デフォルト供給の責務は `calc_streamline(save=True, dpath=...)` 呼び出し側へ。`plot_funcs.add_trajectories` 経由の場合は呼び出し元の `trajectories_option` で渡す |
| `tsc.TscSolver`(`plot=True` の図出力 295行)と `tsc.__main__`(509-519行) | `TscSolver.__init__` に `fig_dir` 引数を追加(デフォルト `"."`)。`__main__` ブロックは引数化 or 削除 |
| `radmc3d.kappa.save`(482行) | `dirpath` 引数を追加(デバッグユーティリティのため必須化でよい) |
| `plot_tools/obs_output.plot_lineprofile`(249行) | `gpath.fig_dir` 直接参照をやめ `pfun.savefig()` 経由に統一(module-default の fig_dir を使う) |

#### P2-A-3: gpath をシム化
- モジュール本体を「`__getattr__(name)` で DeprecationWarning を出しつつ、最後に適用された Config 由来の値(なければ従来のデフォルト)を返す」だけに縮小。`set_*` / `make_dirs` / `remove_radmcdir` は削除。
- **実装スケッチ**(PEP 562 のモジュール `__getattr__` を使用。Python>=3.7 で利用可):
  ```python
  # envos/gpath.py(シム全体)
  import warnings
  _active_config = None   # Config.__post_init__ が _register(config) で設定

  def _register(config):
      global _active_config
      _active_config = config

  _LEGACY = {"run_dir": "run_path", "radmc_dir": "radmc_path", "fig_dir": "fig_path",
             "storage_dir": "storage_path", "logfile": "log_path", "home_dir": None}

  def __getattr__(name):
      if name not in _LEGACY:
          raise AttributeError(name)
      warnings.warn(f"envos.gpath.{name} is deprecated; use Config.{_LEGACY[name] or '...'}",
                    DeprecationWarning, stacklevel=2)
      if name == "home_dir":          # mori2023.py:38 互換: 従来どおりパッケージ親
          return Path(__file__).parents[1]
      if _active_config is not None:
          return getattr(_active_config, _LEGACY[name])
      return _defaults[name]          # Config 未生成時の従来デフォルト
  ```
- リポジトリ内スクリプトを新APIに更新: `mori2023.py:38, 303`(`envos.gpath.*` 参照)。
- `Config.__post_init__` の gpath 書き換えを削除し、シムへの登録(1行)に置換。

- **テスト(本タスクの完了判定の中心)**:
  - **多重実行テスト**: 同一プロセスで `Config(run_dir=A)` のモデル生成→保存→`Config(run_dir=B)` の生成→保存を行い、成果物が A・B に正しく配置される。
  - `envos.gpath.run_dir` が DeprecationWarning つきで動く。
  - 通常インストール(非 editable)を模した環境で `storage_path` が解決される。
- **受け入れ基準**: `grep -rn "gpath\." envos/ | grep -v gpath.py` が0件。多重実行テストパス。既存ゴールデン全パス。

### P2-B: `obs.py`(約1,800行)の分割

- **依存**: P2-A(シグネチャ確定後)。実施順は P2-C の後(§3.2)。
- **設計**: パッケージ `envos/obs/` を作り、**「移動のみ」で挙動を変えない**(`git mv` ベース+`git log --follow` でレビュー可能に):
  | 新モジュール | 移動する内容(現 obs.py 行) |
  |---|---|
  | `envos/obs/simulator.py` | `gen_radmc_cmd`(31)、`ObsSimulator`(59)、`format_array`(460)、`read_radmcdata`(1452。**呼び出し元が observe_* のみのため simulator に置く**) |
  | `envos/obs/convolve.py` | `convolve`(470)、`Convolver`(489) |
  | `envos/obs/data.py` | `BaseObsData`(562)、`Obreso`(1068)、`RefPos`(1112)、`Cube`(1123)、`Image`(1263)、`PVmap`(1311)、`minmaxargs`(1772) |
  | `envos/obs/fits_io.py` | `save_fits`(1345)、`read_obsdata`(1429)、`read_*_fits`(1491-1516)、`read_fits`(1524)、座標変換関数群(1741-1764) |
  | `envos/obs/__init__.py` | 上記全シンボルを re-export |
- **循環参照の扱い**: `data.py` の `BaseObsData.save(mode="fits")` → `fits_io.save_fits` は**メソッド内遅延 import** で解決(fits_io は data のクラスを import するため)。
- **pickle 互換性(重要)**: 既存の `.pkl`(Cube 等)はクラスパス `envos.obs.Cube` を記録している。re-export により `envos.obs.Cube` 属性は存在し続けるため**旧 pickle は読める**。これを保証するテストを追加: 分割**前**に合成 Cube の pickle を `tests/data/cube_legacy.pkl` としてコミットし、分割後コードで `read_obsdata` できることを確認する(このフィクスチャ作成は本タスクの最初のコミットで行う)。
- **受け入れ基準**: テスト全パス。旧 pickle 読み込みテストパス。各新モジュール 600 行以下。`from envos.obs import ObsSimulator` 等の既存 import が全て解決。

### P2-C: ロギングの簡素化

- **依存**: P2-A(`change_rundir` の唯一の呼び出し元が gpath のため)
- **設計**: `log.py` を以下だけに縮小(目安 100 行以下):
  - `logger = logging.getLogger("envos")`(propagate=False、デフォルトで StreamHandler + StandardFormatter を1つ)
  - `setup(level="INFO", logfile=None, file_level=None)` — **冪等**(envos 由来の既存ハンドラを除去してから付け直す)。
  - `StandardFormatter`(P1-5 修正済みのものを流用)
- **`setup()` の実装スケッチ**:
  ```python
  def setup(level="INFO", logfile=None, file_level=None):
      """envos のロギングを(再)構成する。何度呼んでも安全。"""
      for h in list(logger.handlers):
          h.close()
          logger.removeHandler(h)
      logger.setLevel(logging.DEBUG)            # 出力制御はハンドラ側で行う
      sh = logging.StreamHandler()
      sh.setFormatter(StandardFormatter())
      sh.setLevel(_to_level(level))
      logger.addHandler(sh)
      if logfile is not None:
          Path(logfile).parent.mkdir(parents=True, exist_ok=True)
          fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
          fh.setFormatter(StandardFormatter())
          fh.setLevel(_to_level(file_level or level))
          logger.addHandler(fh)
  ```
- **削除**: `loggers` 辞書、`set_logger`、`change_rundir`、`update_logfile` / `set_logfile` / `unset_logfile`(→`setup` に統合)、`set_level`(→`setup`)、`DebugFormatter`・`color`(未使用。grep 確認済み)。`set_logfile` の `filename` 引数はもともと本体で未使用(追補-73)なので互換考慮不要。
- **互換シム**: `update_logfile(**kw)` は DeprecationWarning + `setup` への転送として1リリース残す(`mori2023.py:322` が使用。同PRで mori2023 も `setup` に更新するが、外部ユーザー向けにシムは残す)。
- `Config.__post_init__` は `log.setup(level=self.level_stdout or "INFO", logfile=self.logfile, file_level=self.level_logfile)` を呼ぶ形に簡約(P1-4 の実装を置換)。
- **受け入れ基準**: `log.py` 100行以下。`setup` を3回呼んでもハンドラ数一定。Config 経由のレベル/ログファイル指定がテストで検証される。

### P2-D: FITS I/O の再構築

- **依存**: P2-B、決定 D4
- **背景**: 現実装は B-19〜B-22 のとおり動作実績がほぼ無い。「修正」ではなく仕様を決めて `fits_io.py` を書き直す。
- **設計(D4 推奨案: astropy のみで自前実装)**:
  - **データモデルの明文化**(fits_io.py の docstring):
    - 内部表現: `xau`(RAオフセット×cos(dec0) 補正済み・西向き正、au)、`yau`(au)、`vkms`(km/s、`refpos.freq0` 基準)、`refpos(ra0, dec0, freq0)`。
    - 書き出し: `CTYPE = RA---SIN / DEC--SIN / VRAD`、`CRVAL=(ra0, dec0, freq0)`、`CUNIT = deg / deg / m s-1`、`BUNIT`、ビームは `BMAJ/BMIN/BPA`(deg)。
    - 読み込み: `astropy.wcs.WCS(header)` から **軸タイプ別サブWCS**(`wcs.sub(...)`)で1次元ずつ座標列を得る(現実装の「全軸まとめて `all_pix2world` し列選択漏れ」設計を廃止。B-21 の根治)。fitstype ごとの軸の解釈:
      - **cube**: `[longitude, latitude, spectral]` サブWCS。
      - **image**: `[longitude, latitude]`。
      - **pv**: 経度軸は存在しない(`CTYPE1` は `OFFSET`/`ANGLE` 等)ため、第1軸は `wcs.sub([1])` の線形軸+`CUNIT1` で解釈し、第2軸を spectral として扱う。
      - スペクトル軸は `SPECTRAL` サブWCS + `RESTFRQ`(`RESTFREQ` 表記も受理)で `vkms` へ(B-22 の根治)。`BUNIT="Jy/beam"` は読み込み時に Jy/pix へ変換(`BMAJ` 必須チェック付き)。Stokes 軸など長さ1の軸は squeeze。
    - PVmap の書き出しは `CTYPE1="OFFSET"`(CASA の PV 出力に合わせる。現実装の "ANGLE" から変更)。
  - **書く関数**: `save_fits(obsdata, filepath)`(Cube/Image/PVmap で分岐)。`BaseObsData.save(mode="fits", filepath=...)` から呼ぶ(B-19 の根治)。
  - **読む関数**: `read_cube_fits / read_image_fits / read_pv_fits`(現名維持)。`read_fits` 本体は `_read_fits` に非公開化。`read_obsdata` の fits 分岐(P1-12 で NotImplementedError 化)をヘッダ判定によるディスパッチで復活(B-20 の根治)。
  - **公開シグネチャ(確定仕様)**:
    ```python
    def save_fits(obsdata, filepath, overwrite=True) -> None
    def read_cube_fits(filepath, dpc, unit1=None, unit2=None, unit3=None, v0_kms=0.0, Iunit=None, freq0=None) -> Cube
    def read_image_fits(filepath, dpc, unit1=None, unit2=None, Iunit=None, freq0=None) -> Image
    def read_pv_fits(filepath, dpc, unit1=None, unit2=None, v0_kms=0.0, Iunit=None, freq0=None) -> PVmap
    # unit* はヘッダ CUNIT* の上書き用(None ならヘッダから取得)。
    # 現行シグネチャとの差分: 未配線の unit*_cm/unit*_cms 引数を削除(呼び出し元なし確認済み)。
    ```
  - `Cube.get_pv_map(save=True)`(P1-11 で NotImplementedError 化)を `save_fits` 接続で復活。
- **テスト**:
  - **ラウンドトリップ**: Cube/Image/PVmap 各々で `save_fits → read_*_fits` の軸・データ・ビーム・refpos が `assert_allclose` で一致。
  - **実データ耐性**: CASA 風ヘッダ(NAXIS=4 + Stokes 軸、BUNIT=Jy/beam、deg 軸、RESTFRQ)のミニ FITS を `tests/data/` に合成・コミットし、読めること。
- **削除**: 旧 `read_fits` のデバッグ print 群、未配線の `unit*_cm/unit*_cms` 引数。
- **受け入れ基準**: 上記テストパス。B-19〜B-22 を ISSUES.md でクローズ。

---

## 6.5 Phase 3 以降:v2.0 完了後の継続作業計画(2026-06-14 追記)

### 現在地(この計画を書いた時点の事実)
- ブランチ `develop`、コミット `44c964f`、バージョン **2.0.0**。Phase 0〜2 は全タスク完了・統合済み。
- テスト: **198 件収集**(192 通常 + 6 slow)、CI(lint + Python 3.9/3.11/3.12)グリーン。`ruff check envos/` 除外なしで全パス。
- ISSUES.md: 79 件中 74 件クローズ。**未対応で残るのは意図的据え置きのみ**(C-44=D2、C-49=D10 注記済み、C-52、D-69)。
- **未充足の検証**: `radmc` マーカーのテストは **0 件**。radmc3d 輻射輸送(`observe_line`/`observe_cont` の実行)は一度も自動テストされていない。これが最大の盲点。
- 物理検証テスト(解析的極限との照合)・カバレッジ計測・型ヒント・APIドキュメントはいずれも未整備。

### 環境に関する確定事実(2026-06-14、ローカル clone 上で実地確認)
- gcc / make / **gfortran 13.3** が利用可能(root 権限あり、apt 可)。`git clone https://github.com/dullemond/radmc3d-2.0.git` 成功。**radmc3d バイナリのビルドと実 radmc3dPy の導入は技術的に可能**。
- **ただし** 外部 Fortran(約5万行)のコンパイル・実行と外部 radmc3dPy の import/実行は、過去にこの環境の権限分類器が拒否した操作に該当する。**P3-2 の実施にはオーナーの明示承認が必要**(§9 D11 参照)。承認なしに着手しないこと。

> ⚠ 凡例: **【承認要】**=外部コード実行のためオーナー承認が前提 / **【⚠物理変更】**=数値結果が変わるためゴールデン再生成と変化量記録が必須(§0 の物理変更ルール) / **【破壊的】**=公開API・サポート範囲が変わるため major バージョンを上げる。

### Phase 3: 検証の深化(最優先。承認不要なものは即着手可)

#### P3-1: 新発見バグ ISSUES 追補-78/79 の修正 — **完了済み(2026-06-12, 統合済み)**
- 追補-78(速度プロファイルの phi 軸ブロードキャスト)、追補-79(`PVmap` の InitVar 引数順逆転)を修正。記録のみ。

#### P3-2: radmc3d end-to-end テスト基盤 **【承認要】**
- **依存**: なし(ただしオーナー承認が前提)。
- **内容**: (1) `radmc3d-2.0` を clone し `src/` で `make` → バイナリを PATH へ。(2) 実 `radmc3dPy` を `pip install ./radmc3d-2.0/python/radmc3dPy`(共有 venv のスタブを置換)。(3) 極小グリッドで `ObsSimulator.observe_cont` / `observe_line` → `Cube` 生成までを通す `@pytest.mark.radmc` テストを `tests/test_radmc_e2e.py` に新設。
- **受け入れ基準**: `pytest -m radmc` が 0 件超で全パス。CI の test-full ジョブが同一経路でグリーン(P0-5 の「要動作確認」を完全クローズ)。スタブ前提のテスト(`tests/test_radmc3d.py` の入力ファイル検証)はそのまま維持。
- **誤解防止**: スタブはサンドボックス用の暫定物。本タスクは「スタブを実物に差し替えて初めて回る層」を対象とする。スタブ自体は削除せず、radmc3dPy 不在環境のフォールバックとして残す。

#### P3-3: 物理検証テスト(純Python。承認不要)
- **依存**: なし。
- **内容**: 解析的極限との照合テストを追加。内側領域の速度 → 自由落下 √(2GM/r) への漸近、ディスクのケプラー回転 √(GM/R)、質量保存 ∫ρ dV ≈ Ṁ·t(許容誤差つき)、TSC 解の遠方漸近形。
- **受け入れ基準**: 各テストが物理的に妥当な許容誤差(rtol を明記)で通る。**ゴールデンとは独立**(数値の固定ではなく性質の検証)。グループ A 級の「例外なく結果を歪める」回帰を将来検知できること。

#### P3-4: カバレッジ計測と穴埋め(純Python。承認不要)
- **依存**: P3-3 が入っていると効率的(必須ではない)。
- **内容**: `pytest-cov` を dev 依存に追加。`obs/simulator.py` と `tsc.py` を中心に未到達分岐を特定しテスト追加。
- **受け入れ基準**: カバレッジレポートが CI に出力される。主要モジュールの行カバレッジ目標(例: 80%)を設定し、明確に下回るモジュールにはテストを足すか除外理由を記載。

### Phase 4: 環境近代化(低コスト・高効率)

#### P4-1: Python 3.9 の廃止 **【破壊的】**
- **内容**: `requires-python>=3.10` に変更、CI マトリクスから 3.9 を除去。これにより CI の numpy 1.x pickle スキップ(`tests/test_pickle_compat.py`)が不要になり削除できる。
- **受け入れ基準**: CI から 3.9 が消え、pickle スキップ条件が外れて全環境でフル実行。`D3`(サポート Python)を更新。
- **誤解防止**: サポート範囲の縮小なので major(v3)で行う。3.9 利用者がいないことの確認はオーナー判断(§9 D12)。

#### P4-2: CI アクションの更新と Python 3.13 追加
- **内容**: `actions/checkout` / `setup-python` を最新へ(Node 20 廃止警告の解消)。マトリクスに 3.13 を追加。
- **受け入れ基準**: 警告なしで CI グリーン。

#### P4-3: ruff ルール拡大と format 導入
- **内容**: `ruff.toml` の select を段階的に拡大(F401 未使用 import 等)、`ruff format` を導入し CI に format チェックを追加。
- **受け入れ基準**: 拡大後のルールでクリーン(直せない箇所は per-file-ignores に理由つきで)。format 差分ゼロ。

### Phase 5: 据え置き物理課題の決着 **【⚠物理変更】**(ゴールデン再生成セット)
- **依存**: Phase 3(検証網)完了後が望ましい。
- **方針**: D2 / C-49 / C-52 を個別 PR ではなく **1つの「物理レビュー」作業群**として扱い、各々で論文(Mori+)・元実装意図と照合し、ゴールデン再生成+変化量記録を必須とする。
  - **P5-1 [D2]**: 速度チャネル幅の定義不整合(実幅 `vfw/(nlam-1)` と `Convolver` の `dv_kms` 不一致。最大 ±半チャネル幅の速度ズレ)。C-44 クローズ。
  - **P5-2 [C-49/D10]**: TSC スムージング時のキャビティ内(rho==0)密度・速度の非整合(現状 docstring 注記のみ)。修正は G2 ゴールデン再生成を伴う。
  - **P5-3 [C-52]**: UCM の μ クリップ(`tsc.py`/`models.py` の物理コード)。
- **受け入れ基準**: 各変更で影響を受けるゴールデン(G1/G2/G4 等)を同一 PR で再生成し、変化率を PR 本文と §11 に記録。`physics-change` ラベル。

### Phase 6: ドキュメントとリリース
- **P6-1**: Sphinx または MkDocs による API リファレンス + チュートリアルノートブック。docstring 補強。
- **P6-2**: `CHANGELOG.md` 新設。`develop` → `master` 統合 PR(**オーナー指示待ち**。本計画では未実施)。GitHub Release で `v2.0.0` タグを公開(リモートはタグ push 不許可のため Release UI で作成)。
- **P6-3**: PyPI 公開ワークフロー(`pip install envos` を可能に)。
- **受け入れ基準**: ドキュメントがビルドできてCIで生成、CHANGELOG が SemVer に沿う、Release が公開される。

### Phase 7: アーキテクチャ整理(v3 に向けて)
- **P7-1**: `Config` 神オブジェクト(100超フィールド)を `GridConfig` / `RadmcConfig` / `ObsConfig` 等に分割。全面的な型ヒント付与、`envos/py.typed` 追加、嘘の型注釈(例: かつての `molabun: float = None`)の修正、mypy または pyright を CI に追加。**【破壊的】**(Config の構造変更)。
- **P7-2**: Deprecation シムの削除 — `envos.gpath`(P2-A-3 で導入)と `envos.log.update_logfile`(P2-C で導入)。**【破壊的】**。1リリース猶予の約束に従い v3 で実施。

### Phase 3 以降のマイルストーン
- **M3**: `radmc` テストが 0 件超で CI(test-full 含む)グリーン【P3-2 は承認後】、物理検証テスト群が通る、カバレッジが CI に出る。
- **M4**: CI が最新アクション・サポート Python のみで警告なくグリーン。ruff 拡大ルールでクリーン。
- **M5**: 据え置き物理課題が決着(ゴールデン再生成済み)、API ドキュメント公開、PyPI 配布。

### Phase 3 以降の依存関係
```
[即着手可・並列] P3-3(物理検証) ∥ P3-4(カバレッジ) ∥ P4-2 ∥ P4-3
[承認後] P3-2(radmc3d e2e)  ← オーナー承認が前提
[破壊的・major] P4-1(3.9廃止) → P7-1(Config分割) → P7-2(シム削除)
[⚠物理変更] Phase 5 は Phase 3 の検証網完成後に着手
[随時] Phase 6(docs/release)
```

---

## 7. タスク依存関係(まとめ)

```
P0-1 → P0-2 → P0-3 ─┬→ P0-4
                     └→ P0-5 → P0-6
[Phase 1] P1-2 を最優先。P1-5 → P1-4。他は並列可。
[Phase 2] P1-* 全完了 → P2-A → P2-C → P2-B → P2-D(D4 決定が前提)
```

---

## 8. 完了定義(Definition of Done)

- **M0**: CI グリーン。ゴールデン回帰が PR ごとに走る。CONTRIBUTING.md 完備。
- **M1**: ISSUES.md の A・B 全項目(**P2 に割り付け済みの B-19〜B-22 を除く**)と C の対象項目がクローズ。radmc3d 環境で `python example_run.py` 完走(radmc3d バイナリの無い開発環境では CI の test-full ジョブまたは利用者環境での実行に委譲)。`ruff check envos/`(F821 含む)が per-file-ignores なしで全パス。
  **【2026-06-12 達成記録】** 148 passed + slow 6 passed、ruff 除外なしで全クリーン。⚠物理変更2件(P1-9 ディスク合成: ディスク領域で最大49.6%、P1-1 year定数: smpy系で≈0.33%)はゴールデン G4 追加・G3b 再生成で固定。タグ `v1.1.0`。example_run の radmc3d 実行部のみ CI test-full に委譲。
- **M2**: (1) gpath 直接参照ゼロ+多重 run_dir テストパス、(2) obs 分割後も全テスト・旧 pickle 互換テストパス、(3) log.py 100 行以下、(4) FITS ラウンドトリップ+CASA風読み込みテストパス。
  **【2026-06-12 達成記録】** 4条件すべて充足: (1) gpath参照0件・test_paths 12件パス、(2) obs 5モジュール分割・レガシーpickle互換8テストパス、(3) log.py 80行、(4) ラウンドトリップ精度~1e-13・CASA風NAXIS=4読込パス。総計 **193 passed + slow 6**、ruff 全クリーン。バージョン 2.0.0、タグ `v2.0.0`(ローカル)。残課題: D2(チャネル幅、先送り)、ISSUES 追補-78/79(新規発見、未着手)、Deprecationシム(gpath / update_logfile)の v3 での削除。

---

## 9. 未決事項(オーナー確認推奨。確認できない場合は推奨案で進める)

| ID | 論点 | 推奨案 |
|---|---|---|
| D1 | ディスク密度の合成: `rho += disk.rho`(現状)か `rho[cond] = disk.rho[cond]`(置換)か | **置換方式**。速度場の扱いと整合し、コメントアウト履歴から元の意図と推定。⚠disk 使用時のみ結果が変わる |
| D2 | 速度チャネル幅: `nlam=round(vfw/dv)` + `linspace(-vfw/2, vfw/2, nlam)` では実幅が `vfw/(nlam-1)` で、`Convolver` に渡す `dv_kms` と不一致 | **Phase 1〜2 では挙動維持**(修正は⚠物理変更)。M2 以降の独立タスクとし、ドキュメントに実チャネル幅の定義のみ明記 |
| D3 | サポート Python | 3.9〜3.12(CI マトリクスで担保) |
| D4 | FITS I/O: 自前(astropy のみ)か spectral-cube 依存か | **自前**。PVmap という非標準データ型があり spectral-cube は Cube にしか合わない。依存も重い |
| D5 | streamline の mirror モード | **削除**(未使用・二重誤実装・ドキュメントなし) |
| D6 | non-LTE モード | **NotImplementedError 化**(本人コメントで Not tested、lines.inp の種数不整合あり、動作実績なしと判断) |
| D7 | `nconst.Lsun=3.848e33`(IAU 3.828e33 と不一致) | **据え置き+コメント**(直すなら⚠物理変更の独立タスク) |
| D8 | `rot_ccw`(未配線) | **docstring に「未実装」明記のみ**(P1-4)。実装するなら `vp` 符号反転の独立タスク |
| D9 | `storage/` の配置とデフォルト解決(現状は editable install 前提) | **`envos/storage/` へ移動し importlib.resources で解決**(P2-A-1)。旧位置フォールバックを1リリース維持 |
| D10 | C-49: TSCスムージング(`smoothing_TSC=True`)でキャビティ内(rho==0)は密度に外側エンベロープを混ぜないが速度は全域で混ぜる非整合 | **据え置き**(物理コードの挙動固定の原則。修正は⚠物理変更で TSC 使用時の全結果に影響)。P1-9 で docstring に既知の限界として明記。修正する場合は M2 以降に G2 ゴールデン再生成つきの独立タスク |
| D11 | P3-2(radmc3d e2e テスト)の実施可否 — 外部 Fortran(約5万行)のビルド・実行と外部 radmc3dPy の import を伴い、過去に権限分類器が拒否した操作に該当 | **オーナー承認を得てから実施**。承認なしに着手しない。環境(gfortran・ソース取得)は確認済みで、承認さえあれば技術的障壁はない |
| D12 | P4-1(Python 3.9 廃止)の可否 — サポート範囲の縮小(破壊的) | **オーナー判断**。3.9 実利用者がいなければ廃止して `>=3.10` に。CI の numpy 1.x pickle スキップ撤去とセット |

---

## 付録A: 互換性維持対象の公開API

リポジトリ内スクリプトの実使用に基づく(変更する場合は同一PRで使用側も更新):

- `envos.Config`(全フィールド名)、`.replaced()`、`.log()`
- `envos.ModelGenerator(config)` / `(readfile=...)`、`.calc_kinematic_structure()`、`.calc_thermal_structure()`、`.get_model()`、`.get_meshgrid()`、`.set_grid(ri,ti,pi)`、`.set_gas_density()`、`.set_gas_velocity()`、`.save()`、`.model`
- `envos.read_model(path)`、`envos.read_obsdata(path)`
- `envos.ObsSimulator(config)`、`.set_model(model)`、`.observe_line(obsdust=...)`、`.observe_cont(...)`
- `Cube.get_mom0_map(...)`、`.get_pv_map(pangle_deg=...)`、`.trim(xlim=...)`、`.save(...)`; `PVmap/Image` の `.norm_I()`、属性 `xau/yau/vkms/Ipv/Ippv/data/dpc/obreso`
- `CircumstellarModel`: `.rhogas/.Tgas/.vr/.vt/.vp/.rr/.tt/.R/.z/.rc_ax/.tc_ax/.ppar/.mu0`、`.save_pickle()`、`.calc_column_density()`、`.get_midplane_profile()`、`.get_gasmask()`、`.get_argmid()`
- `envos.nc.*`(定数名)、`envos.tools.clean_radmcdir()`、`envos.log.update_logfile()`(P2-C で `setup` へ。シム要)、`envos.gpath.run_dir/home_dir`(P2-A-3 で Deprecation シム)
- `envos.streamline.calc_streamline(...)`(`names/units/variables/save/label/dpath` 引数)
- `envos.plot_tools.*`: `plot_rhogas_map, plot_Tgas_map, plot_mom0_map, plot_pvdiagram, plot_velocity_midplane_profile, plot_losvelocity_midplane_map, plot_density_velocity_midplane_profile, plot_variable_meridional_map, plot_rhogas_midplane_profile, plot_Tgas_midplane_profile`
- `envos.datacor.calc_datacor`(外部利用の可能性があるためシグネチャ維持)

## 付録B: ISSUES.md 項目 → タスク対応表

| ISSUES | タスク | ISSUES | タスク |
|---|---|---|---|
| A-1 | P1-1 | B-27 | P1-12(関数ごと削除) |
| A-2 | P1-2 | B-28 | P1-8 |
| A-3〜A-7 | P1-11 | B-29, B-30 | P1-5 |
| A-8 | P1-9 | B-31〜B-33 | P1-17 |
| A-9 | P1-8(根治 P2-A) | B-34〜B-37 | P1-16 |
| A-10 | P0-2 | B-38, B-39 | P1-18 |
| B-11 | P1-3(根治 P2-A) | B-40 | P1-19 |
| B-12, B-13 | P1-4 | B-41 | P1-20 |
| B-14〜B-16 | P1-10 | C-42, C-43 | P1-2 |
| B-17, B-18 | P1-12 | C-44 | **先送り(D2)**(⚠物理変更のため M2 以降の独立タスク) |
| B-19〜B-22 | P2-D | C-45〜C-47 | P1-11 |
| B-23〜B-25 | P1-13 | C-48 | P1-9 |
| B-26 | P1-6(関数削除でクローズ) | C-49 | **据え置き(D10)**(P1-9 で docstring 注記のみ) |
| C-50, C-51 | P1-7 | — | — |
| C-52 | **据え置き**(tsc.py の物理コード。非ゴール原則) | C-53 | P1-8(NaN警告のみ。cubicsolver 本体は据え置き) |
| C-54, C-55 | P1-14 | C-56 | P1-15 |
| C-57, C-58 | P1-10 | C-59〜C-61 | P1-11(項12〜14) |
| C-62 | P1-12(header.py 削除でクローズ) | D-63 | P1-4 |
| D-64 | P1-4(注記)/ D8 | D-65 | P1-5 |
| D-66, D-67 | P2-A | D-68 | P1-2 |
| D-69 | 据え置き(設計判断。pickle 継続) | D-70 | P0-3〜P0-5 |
| 追補-71〜76 | P1-11 / P1-4 / P2-C / P1-12 / P1-2 / P1-12 | 追補-77 | P1-11(項4) |

---

## 10. 改訂履歴

- **v3.0-draft**: Phase 0〜2 を全完了し v2.0.0 に到達(M0/M1/M2 達成記録は §8)。v2.0 以降の継続作業計画を §6.5 に新設(Phase 3 検証深化 / Phase 4 近代化 / Phase 5 物理課題決着 / Phase 6 docs・release / Phase 7 アーキテクチャ)。各タスクに【承認要】【⚠物理変更】【破壊的】の凡例を付し誤解を防止。マイルストーン M3〜M5 と Phase 3 以降の依存関係を追加。§9 に D11(radmc3d e2e の承認要件)・D12(Python 3.9 廃止判断)を追加。ローカル clone 環境で gfortran 13.3・radmc3d ソース取得可を実地確認(P3-2 は環境的に可能だが承認が前提)。
- **v1.0**: 初版。
- **v1.1**: 検証パス1の結果を反映(§11 ログ参照)。主な変更: デッドコード一覧の確定(§2.5)と P1-12 の横断タスク化、`shell()` を Popen 単一実装への書き直しに変更(進捗表示の維持)、G3 を (a)/(b) に分割(year 修正の影響を捕捉可能に)、nonlte の早期失敗位置を `__init__` に変更、`read_radmcdata` の移動先を simulator.py に訂正、P2-B に pickle 互換性の保証とテストを追加、D9(storage 配置)を新設、P2-A-2 の表に呼び出し元更新(mori2023)を明記。
- **v1.2**: 検証パス2の結果を反映(§11 ログ参照)。主な変更: C-59〜C-61 を P1-11 の修正項目(項12〜14)として追加、`header.py` 削除を P1-12 に追加(C-62 クローズ)、P1-4 を `update_logfile()` 方式に変更、P2-A-2 の表に gpath 参照3箇所(tsc 図出力・kappa.save・plot_lineprofile)を追加、P2-D に fitstype 別の軸解釈(PV には経度軸が無い)と `CTYPE1="OFFSET"` の決定を追記、付録Bの C 系・D 系マッピングを全面訂正。検証パス3で収束を確認。
- **v2.2**: 実環境検証(検証パス6)の結果: A-7(refpos 共有デフォルト)は Python 3.11 以降 dataclass の制約により **import 時クラッシュ**となることが判明。P1-11 項5 を P0-2 に繰り上げ。P0-2 を「現代環境互換修正」に改題。
- **v2.1**: ブランチ戦略を §0-2 に明文化(統合ブランチ `develop`、タスクブランチ命名規則 `task/<タスクID>-<説明>`、マイルストーンごとに `master` へ統合)。
- **v2.0**: 検証パス4(深掘り)の結果を反映(§11 ログ参照)。主な変更: §0 に PR テンプレ・削除判断基準・ロールバック方針を追加、§2.5 に `calc_dependent_params` を追加し P1-6 を「削除+ガード」に再定義(B-26 は削除でクローズ)、§2.6(未使用だが残す公開API一覧)新設、§3.4(バージョニング)・§3.5(テストファイル台帳)新設、Phase 1 にリスク・規模一覧を追加、P1-2 に実機確認の受け入れ基準を追加、P1-5 にハンドラ close を追加、P1-9 に C-49 の据え置き注記(D10 新設)、P1-11 項4 を `norm_I` 経由に変更(追補-77 解消)・項14 の Obreso 仕様を厳密化、P2-A-1/A-3/C/D に実装スケッチと確定シグネチャを追加、付録Bの誤割り付け(C-44 → D2、C-49 → D10)を訂正、P0-1 のバージョン管理を §3.4 と整合化。検証パス5で収束を確認。

## 11. 検証・自己批判ログ

### 検証パス1(v1.0 → v1.1)
コードと照合して見つけた v1.0 の不備と対応:

1. **[誤り] `get_interface_coord` の「修正」を指示していたが、呼び出し元がゼロのデッドコードだった。** grep で確認(定義のみ)。`compressed_x2`・`thax_ver==2/3` 分岐・`make_function_loglog`・`find_proper_nthread`・`convolve_image` も同様にデッド。→ 修正ではなく削除に変更(P1-12 を横断タスク化、§2.5 新設)。
2. **[誤り] `read_radmcdata` を fits_io.py に分類していた。** 実際は FITS と無関係(radmc3dPy の image オブジェクト→Cube/Image 変換)で、呼び出し元は `observe_*` の3箇所のみ。→ simulator.py へ訂正。
3. **[見落とし] P2-B のモジュール分割が既存 pickle を壊すリスク。** pickle はクラスの `__module__` を記録するため、`envos.obs.Cube` → `envos.obs.data.Cube` の移動は旧データの読み込みに影響しうる。re-export で属性解決が維持されるため実際は読めるが、**保証するテストが計画に無かった**。→ レガシー pickle フィクスチャと互換テストを P2-B に追加。
4. **[見落とし] storage/ のデフォルト解決が editable install 前提。** `gpath.storage_dir = パッケージ親/storage` は通常の pip install では site-packages を指して壊れる。→ D9 新設、P2-A-1 に importlib.resources 方式を明記、P0-1 に「editable を標準とする」注記。
5. **[不正確] P1-1 のゴールデン影響予測。** G3 の T+CR+Ms ケースは `year` に依存せず**不変**(cgs 内部値のみ保存するため)。`Mdot_smpy` 入力ケースだけが `nc.smpy` 経由で変わる。→ G3 を (a)/(b) に分割し、(a) 不変・(b) 変化という予言の検証をタスクに組み込み。
6. **[設計不備] P1-2 の `shell()` 修正案(subprocess.run + capture)では radmc3d 長時間実行の進捗が見えなくなる。** mctherm は数分かかるため UX 後退。→ Popen で行ストリーミングしつつ keyword 照合する単一実装に変更。`simple` 引数は呼び出し元4箇所が未使用と確認の上で削除。
7. **[設計不備] P1-10 の nonlte 対応(`lines.inp` の種数修正)は中途半端。** 動作実績がない機能を「一部だけ」直すとテスト不能なコードが残る。→ `__init__` での早期 `NotImplementedError` に変更(D6)。
8. **[記述の混乱] P1-9 のディスク合成の疑似コードに無意味な式が混入、D2 の文言が「P1-11 に含める」と「先送り」で矛盾。** → どちらも整理(D2 は完全に先送りで統一)。
9. **[漏れ] 新規発見4件(`save_fitsfile` 不存在、`level_stdout/level_logfile` 未配線、`set_logfile(filename=)` 未使用、デッドコード群)が ISSUES.md に未記載。** → ISSUES.md に追補節(71〜76)を追加し、付録Bにマッピング。
10. **[確認済みで問題なし] 引用行番号のスポットチェック**(obs.py:54/209-211/223/312/1247、config.py:348、tsc.py:92/345、models.py:61/69、mori2023.py:38/303/322、grid.py:100-113)— 全て一致。

### 検証パス2(v1.1 → v1.2)
v1.1 を頭から通読し、各タスクを「新規参加者が質問せずに実行できるか」「受け入れ基準は修正を実際に検証するか」の観点で再点検した結果:

1. **[誤り] 付録Bの対応表で C-59(devnull リーク)を「P1-12 削除内」としていた。** `_subcalc` はマルチスレッド観測の現役コードでありデッドコードではない。→ P1-11 に修正項目(項12)として追加し、対応表を訂正。
2. **[漏れ] C-60(Convolver の null モード/ビーム None)と C-61(Obreso の両 None)に対応するタスクが無かった**(P2-D 設計内と誤記)。どちらも obs.py の小修正。→ P1-11 に項13・14として追加。
3. **[発見] `envos/header.py` はどこからも import されていない**(grep で確認)。Python2 向けバージョンチェックの残骸。→ P1-12 の削除対象に追加し、C-62 をこの削除でクローズ。
4. **[設計不備] P1-4 で `set_logfile()` を使う指示は、Config を複数回生成するとファイルハンドラが重複する。** → `update_logfile()`(unset→set)に変更。P1-5(unset_logfile 修正)への依存を明記。
5. **[漏れ] P2-A-2 の変更表に gpath 参照3箇所が未掲載だった**: `tsc.TscSolver` の図出力(tsc.py:295, 509-519)、`radmc3d.kappa.save`(482行)、`plot_lineprofile` の直接参照(obs_output.py:249)。受け入れ基準「gpath 参照ゼロ」と矛盾していた。→ 表に3行追加。
6. **[設計不備] P2-D の「軸タイプ別サブWCS」方針は PV FITS に適用できない**(PV には経度軸が無く `CTYPE1=OFFSET` 等)。→ fitstype ごとの軸解釈を明文化(cube=lon/lat/spectral、pv=線形第1軸+spectral)。PVmap 書き出しの `CTYPE1` は CASA に合わせ "OFFSET" と決定。
7. **[確認済み] `shell()` の戻り値は呼び出し元4箇所すべてで未使用** → P1-2 に「戻り値 None でよい」と明記。
8. **[確認済み] P1-17 の `plot_midplane_radial_profile(model, Omega配列, save_name=...)` 経路**は ndarray 分岐+`get_midplane_profile(ndarray)` で成立する(コード確認済み)。
9. **[確認済み] P0-4 G2 の TSC テーブル同梱方式**: `tscsol.pkl` は `TscData`(envos.tsc のdataclass)の pickle であり、モジュールが存続する限り読める。生成は `make_golden.py` 内で `storage_dir=tests/golden` を指す Config で1回だけ実施。

### 検証パス3(v1.2 の収束確認)
v1.2 全体を通読し、(1) タスク間の依存関係の整合(P1-5→P1-4、P2-A→C→B→D)、(2) ISSUES.md 全項目(70+追補6)がいずれかのタスクまたは明示的な「据え置き」判断に割り付けられていること、(3) 受け入れ基準とテストの対応、(4) 公開API(付録A)と各タスクのシグネチャ変更の整合(変更箇所はすべて同一PRでの呼び出し元更新かシムを明記)を確認した。実質的な修正は発生せず、**v1.2 で収束**と判断する。

### 検証パス4(v1.2 → v2.0、深掘り)
「他の人が作業しても大丈夫」の水準を上げるため、(i) 各タスクの修正対象が本当に最適な処置か(修正/削除/据え置きの判断根拠)、(ii) 対応表の整合、(iii) Phase 2 の実装が一意に決まるか、の3観点で再検証した結果:

1. **[矛盾] 付録Bで C-44(チャネル幅)を P1-11 に割り付けていたが、D2 では「先送り」と決定済み。** 自己矛盾。→ C-44 → D2(先送り)に訂正し、C-45〜C-47 のみ P1-11 に。
2. **[矛盾] C-49(TSCスムージングの密度/速度非整合)を P1-9 に割り付けていたが、P1-9 の修正項目に存在しなかった。** 物理コードのため挙動固定の原則と衝突する。→ D10 を新設して「据え置き+docstring 注記」と明示。
3. **[発見] `calc_dependent_params` は呼び出し元ゼロ**(grep 確認)。クラス版と完全重複でバグ持ち。「修正」より「削除」が `get_interface_coord` の扱いと一貫する。→ P1-6 を再定義。削除/維持の判断がタスクごとにぶれないよう、**判断基準を §0-9 に成文化**した(これにより `calc_midplane_average` は「動く公開APIは残す」側と明確化)。
4. **[発見] obs.py の解析系メソッド群(mask/reverse_ax/move_center 等)もリポジトリ内未使用。** 削除はしないが、計画に記録がないと後続作業者が再調査することになる。→ §2.6 として一覧化し、テスト優先度の判断も付した。
5. **[発見] `get_mom0_map(normalize="peak")` が `Iunit` を更新しない**(`norm_I` と非整合)。→ ISSUES 追補-77 として記録し、P1-11 項4 の実装を `norm_I("max")` 経由に変更して同時解消。
6. **[不備] Obreso の検証仕様(P1-11 項14)が「両方 None」の場合しか想定しておらず、部分指定(maj_au のみ等)で従来どおり TypeError になる。** コードを再読して全分岐を確認。→ ペア完全性の仕様として厳密化。
7. **[不備] P1-5 の `unset_logfile` 修正にハンドラの `close()` が無く、`update_logfile` の繰り返しで fd リーク。** → close を明記。
8. **[不整合] §3.4(新設)の dynamic version と P0-1 の「version="1.0.0" を一致させる」が衝突。** → P0-1 を dynamic 方式に統一。
9. **[不足] リスクの高いタスク(P1-2, P1-11)の扱いが他と同列だった。** → リスク・規模一覧を §5 冒頭に新設し、P1-2 に実機確認を受け入れ基準として追加、P1-11 にコミット分割の指示を追加。
10. **[不足] Phase 2 は方針記述のみで、実装者の裁量が大きすぎた。** → P2-A-1(Config プロパティ)、P2-A-3(gpath シム、PEP 562)、P2-C(`setup()`)、P2-D(公開シグネチャ)に実装スケッチを追加。

### 検証パス6(v2.2、実環境での検証)
開発着手時の環境構築(Python 3.11.15、numpy 2.4.6、scipy 1.17.1、astropy 等最新)で初めて実行レベルの検証が可能になり、以下が判明:
1. **[重大な訂正] A-7(`refpos: RefPos = RefPos()`)は「インスタンス間の状態共有」ではなく、Python 3.11 以降では `ValueError: mutable default ... is not allowed` による import 時クラッシュ。** dataclass の mutable-default 検査が 3.11 で unhashable 型全般に拡大されたため。`import envos` が一切通らないことを実機で確認し、P1-11 項5 を P0-2 に繰り上げた。3箇所の `default_factory` 化のみで import が通ることも実機確認済み。
2. **[確認] A-10(scipy.simps/cumtrapz)は予測どおり import 時ではなく実行時の障害**(関数参照が遅延のため)。P0-2 の位置づけは不変。
3. **[制約] radmc3dPy は PyPI に存在せず**(PLAN の記載どおり)、サンドボックスでは公式 git からのインストールが権限ポリシーで不許可。テスト実行用に最小スタブ(import 時シンボルのみ、呼べば NotImplementedError)を venv に配置して対応。CI(P0-5)では実物をインストールする計画に変更なし。

### 検証パス5(v2.0 の収束確認)
v2.0 全体を機械的に照合(grep による相互参照チェック+付録B全行の目視)した。発見と対応:
1. **[残存矛盾] 付録Bに v1.2 由来の旧行 `C-44〜C-47 → P1-11` が新行(C-44 → D2)と並存していた。** パス4の編集が表の一部しか置換していなかった。→ 表を再構成し、全項目が一意に割り付くことを行単位で確認。
2. **[残存矛盾] P1-11 の見出しが旧範囲 `C-44〜C-47` のままだった。** → `C-45〜C-47, C-59〜C-61, 追補-71, 77` に訂正。

上記2件の修正後、(1) 付録Bの全項目(A-1〜D-70、追補-71〜77)が「タスク割り付け」「先送り(D2)」「据え置き(D10 等)」のいずれかに**重複なく一意に**解決されること、(2) §0-9 の削除基準が P1-6 / P1-8 / P1-12 / §2.6 の判断と矛盾しないこと、(3) §3.5 テスト台帳が各タスクのテスト要件と対応すること、(4) 実装スケッチが既存コードの事実(dataclass フィールド名、mori2023 の gpath 使用箇所、PEP 562 の利用可否)と整合することを確認。grep による `C-44〜C-47` 残存参照の検索は0件。これ以上の照合で新たな矛盾は出なくなったため、**v2.0(本修正込み)で収束**と判断する。

残存する既知の限界(計画として許容):
- 行番号は HEAD 時点のもの。先行タスクでずれるため、引用コードでの特定を §0-7 で義務付けている。
- `radmc3dPy` の CI インストール方法(P0-5)は実環境での動作未確認のため「要動作確認」フラグつき。P0-5 の作業者が最初に検証する。
- 外部ユーザーの API 使用状況は把握できないため、互換性判断はリポジトリ内スクリプトの使用実態に基づく(付録A)。削除系タスクは PR 本文での告知を義務付けた(P1-12)。
