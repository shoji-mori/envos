# envos コードレビュー: バグ・懸念点リスト

コードベース全体(envosパッケージ + スクリプト類、約8,000行)を通読して検出した問題のリスト。
「結果の正しさに関わるもの」「確実にクラッシュするもの」「特定の使い方で壊れるもの」「設計・ドキュメント上の懸念」の4段階に分類している。

---

## 対応状況

| 項目 | タスク(PLAN.md) | 状態 |
|---|---|---|
| **グループ A: 結果の正しさに関わる問題** | | |
| A-1 | P1-1 | 未着手 |
| A-2 | P1-2 | **対応済み** |
| A-3〜A-6 | P1-11 | 未着手 |
| A-7 | P0-2 | **対応済み**(develop に統合済み) |
| A-8 | P1-9 | 未着手 |
| A-9 | P1-8(根治 P2-A) | **対応済み**(根治はP2-A) |
| A-10 | P0-2 | **対応済み**(develop に統合済み) |
| **グループ B: 確実にクラッシュする・呼べば必ず壊れるもの** | | |
| B-11 | P1-3(根治 P2-A) | **対応済み**(根治はP2-A) |
| B-12, B-13 | P1-4 | **対応済み** |
| B-14〜B-16 | P1-10 | 未着手 |
| B-17, B-18 | P1-12 | 未着手 |
| B-19〜B-22 | P2-D | 未着手 |
| B-23〜B-25 | P1-13 | 未着手 |
| B-26 | P1-6(関数削除でクローズ) | **対応済み** |
| B-27 | P1-12(関数ごと削除) | **対応済み** |
| B-28 | P1-8 | **対応済み** |
| B-29, B-30 | P1-5 | **対応済み** |
| B-31〜B-33 | P1-17 | 未着手 |
| B-34〜B-37 | P1-16 | 未着手 |
| B-38, B-39 | P1-18 | 未着手 |
| B-40 | P1-19 | **対応済み** |
| B-41 | P1-20 | **対応済み** |
| **グループ C: 特定の使い方で壊れる・サイレントに誤るもの** | | |
| C-42, C-43 | P1-2 | **対応済み** |
| C-44 | 先送り(D2) | 未着手 |
| C-45〜C-47 | P1-11 | 未着手 |
| C-48 | P1-9 | 未着手 |
| C-49 | 据え置き(D10) | 未着手 |
| C-50, C-51 | P1-7 | **対応済み** |
| C-52 | 据え置き(物理コード非ゴール原則) | 未着手 |
| C-53 | P1-8(NaN警告のみ) | **対応済み** |
| C-54, C-55 | P1-14 | 未着手 |
| C-56 | P1-15 | 未着手 |
| C-57, C-58 | P1-10 | 未着手 |
| C-59〜C-61 | P1-11 | 未着手 |
| C-62 | P1-12(header.py 削除でクローズ) | **対応済み** |
| **グループ D: 設計・ドキュメント上の懸念** | | |
| D-63 | P1-4 | **対応済み** |
| D-64 | P1-4(注記)/D8 | **注記済み**(実装はD8判断待ち) |
| D-65 | P1-5 | **対応済み** |
| D-66, D-67 | P2-A | 未着手 |
| D-68 | P1-2 | **対応済み** |
| D-69 | 据え置き(設計判断。pickle 継続) | 未着手 |
| D-70 | P0-3〜P0-5 | 未着手 |
| **追補(2026-06-12 発見)** | | |
| 追補-71 | P1-11 | 未着手 |
| 追補-72 | P1-4 | **対応済み** |
| 追補-73 | P2-C | 未着手 |
| 追補-74 | P1-12 | 未着手 |
| 追補-75 | P1-2 | **対応済み** |
| 追補-76 | P1-12 | 未着手 |
| 追補-77 | P1-11 | 未着手 |

---

## A. 最重要: 結果の正しさに関わる問題

例外を出さずに科学的結果を静かに歪めるタイプのバグ。最優先での修正を推奨。

1. **`envos/nconst.py:28` — 1年の秒数が間違っている。**
   `year = yr = 3.1454e7` だが、正しくは約 `3.1557e7` 秒(365.25日)。約0.33%のずれが
   `Msun_per_yr`、`Mdot`、`t` など全ての時間関連の物理量に伝播する。`Myr`(29行目)も同様。

2. **`envos/tools.py:262` — `shell()` の `simple = 1` がハードコードされ、エラー検出が全て死んでいる。**
   引数 `simple` を直後に上書きしており、`error_keyword=["ERROR", "STOP"]` によるRADMC-3D実行エラーの
   検出コード(279行目以降)には到達不能。さらに `subprocess.run` に `check=True` がないため、
   radmc3dが失敗しても例外にならず、古い/壊れた出力を読んで計算が続行する。
   なお到達不能側のコードは `re` をimportしていないため、復活させてもNameErrorになる。

3. **`envos/obs.py:1188-1192 — `Cube.get_mom0_map()` で `vlim` が無視される。**
   `vlim` で速度範囲を切り出した `_Ipp` を作った直後、`method=="sum"` 分岐が全速度域の
   `self.Ippv` で `_Ipp` を上書きする。速度範囲を指定したモーメント0マップは黙って全範囲の積分になる。

4. **`envos/obs.py:569-573` — `_reset_positive_axes()` が何もしていない。**
   `img = np.flip(img, i)` はローカル変数の再代入、`axs[i] = ax[::-1]` も呼び出し側で作った
   一時リストの変更であり、`self.Ippv` / `self.xau` 等には一切反映されない。
   軸が降順のFITSを読んだ場合、軸とデータの対応が壊れたままになる。

5. **`envos/obs.py:211` — `observe_cont()` の `posang = phi or self.posang`。**
   タイポで `posang` 引数が無視され `phi` が使われる。また223-224行目は `npixy=self.npixx`
   (長方形画像で誤り)。さらに `incl or self.incl` パターン(209行目ほか、`observe_line` も同様)は
   `incl=0` など0を明示指定すると無視される。

6. **`envos/obs.py:312` — `observe_line()` がradmc3dコマンドに引数の `iline` ではなく `self.iline` を使う。**
   一方で出力データの `freq0` やメタ情報には引数 `iline` を使うため(373行目)、引数で指定すると
   計算した遷移と記録される周波数が食い違う可能性がある。

7. **`envos/obs.py:1153, 1268, 1316` — `Cube`/`Image`/`PVmap` の `refpos: RefPos = RefPos()` が共有ミュータブルデフォルト。**
   全インスタンスが同一の `RefPos` オブジェクトを共有するため、1つのキューブに `freq0` や `ra0` を
   設定すると他の全データに波及する。
   **【2026-06-12 追記】Python 3.11 以降では dataclass が unhashable なデフォルト値を拒否するため、
   これは `import envos` 自体を `ValueError` で失敗させる(実機確認済み)。事実上の最重要クラッシュ。
   PLAN.md P0-2 に修正を繰り上げ。**

8. **`envos/model_generator.py:182-190` — ディスク合成のロジックが不整合。**
   密度は `rho += self.disk.rho`(全域に加算)なのに、速度は `cond`(ディスクが卓越する領域)のみ置換。
   コメントアウトされた `rho[cond] = ...` と方針が混在しており、エンベロープ領域の密度が
   ディスク分だけ過大になる。

9. **`envos/models.py:9` — `from .gpath import run_dir` をimport時に束縛。**
   その後 `Config(run_dir=...)` で実行ディレクトリを変えても、`ModelBase.save_pickle()` /
   `read_pickle()` のデフォルト保存先は古いパスのまま。

10. **SciPy互換性: 削除済みAPIの使用。**
    `integrate.simps`(`tsc.py:345`、`obs.py:1192`、`plot_funcs.py:405`)と
    `integrate.cumtrapz`(`tsc.py:92`)は SciPy 1.14 で削除済み。一方 `tools.py` / `models.py` は
    新しい `simpson` を使っており混在。新しいSciPyでは `AttributeError` でクラッシュする。

---

## B. 確実にクラッシュする・呼べば必ず壊れるもの

11. **`envos/gpath.py:34-37` — `set_radmcdir()` が `radmc_dir` ではなく `run_dir` を設定**
    (`global run_dir; run_dir = Path(path)`)。明白なコピペバグ。

12. **`envos/config.py:348` — `set_logfile("on")`。**
    `log.set_logfile` の第1引数はロガー名なので `loggers["on"]` → `KeyError`。
    `Config(logfile=...)` を指定すると必ずクラッシュする。

13. **`envos/__init__.py:10-23` — `__all__` に存在しない `read_mg` と `column_density`。**
    `from envos import *` が `AttributeError` になる。

14. **`envos/radmc3d.py:97-103` — `set_model()` の二重バグ。**
    モデルがファイルパスの場合 `self.model_pkl`(未定義属性)を参照して `AttributeError`。
    仮に直しても、直後の `if model is not None:` が elif でないため、読み込んだモデルが
    パス文字列で上書きされる。

15. **`envos/radmc3d.py:120-121` — ライブラリ内で `exit()`。**
    rhodustがない場合にインタープリタごと殺す。また `CircumstellarModel` はdataclassなので
    `hasattr(md, "rhodust")` は常にTrueで、`rhodust=None` のとき(ユーザー定義モデル例)は
    後段の `rhod.ravel()` で不可解な `AttributeError` になる。

16. **`envos/radmc3d.py:199-200` — `remove_file("gas_temperature.inp")` がradmc_dirではなく
    カレントディレクトリのファイルを消そうとする(パス結合漏れ)。**
    `set_temperature()` で書いた古い温度ファイルが消えず、mcthermの結果に影響する可能性がある。

17. **`envos/obs.py:117-122` — `set_radmc_input()` は二重に壊れている。**
    `RadmcController(**conf.__dict__)` はシグネチャ不一致で `TypeError`、
    `clean_radmc_dirs()` というメソッドも存在しない。

18. **`envos/obs.py:242-283` — `observe_line_profile()` の本体全体が文字列リテラル。**
    呼んでも何もせず `None` を返す。

19. **`envos/obs.py:1345-1425` — `save_fits()`。**
    冒頭で `filepath = gpath.run_dir / filename` を作るのに、最後は `hdulist.writeto(filename, ...)`
    でカレントディレクトリに書き込む。さらに `od.beam_maj_au`(1400行目)は
    `od.obreso.beam_maj_au` にあるべき属性なので、ビーム情報付きデータでは `AttributeError`。
    `BaseObsData.save(mode="fits", filepath=...)` 経由だと `filename=None` が渡って即死。

20. **`envos/obs.py:1429-1449` — `read_obsdata()` のfits分岐は `print("do fits")` して `None` を返すだけ。**
    それ以外の拡張子では `sys.exit(1)`(ライブラリでprocess kill)。

21. **`envos/obs.py:1590-1604` — `read_fits._get_ax()` が軸の1次元配列ではなく `(n, naxis)` の
    2次元配列を返す。** 列を選択する行(`[:, axnum-1]`)がコメントアウトされたままで、
    生成される `xau` 等が2次元になる。

22. **`envos/obs.py:1664` — `unit_name.lower() in ["Hz"]` は絶対にマッチしない**
    (小文字化した文字列と "Hz" を比較)。周波数軸のFITSが読めない。

23. **`envos/streamline.py:88-90` — `Streamline.save_data()` の `global run_dir`。**
    モジュール内に `run_dir` は存在しない(importが8行目でコメントアウト済み)ので、
    `dpath=None` で `NameError`。`save=True` のストリームライン保存は必ず失敗する。

24. **`envos/streamline.py:16-18, 55-56` — ミュータブルデフォルト引数の蓄積。**
    `variables=[]` に `_variables += [...]` で破壊的に追加するため、`calc_streamline()` を
    複数回呼ぶと変数リストが呼び出しをまたいで増殖する。

25. **`envos/streamline.py:211-214` — ミラー対称モードのコピペバグ。**
    `vr = -self.vt_field(...)`(vr_fieldであるべき)で、ミラー座標 `-0.5*np.pi + pos[1]` も
    鏡映(`π - θ`)として正しくない。

26. **`envos/physical_params.py:25-41` — `calc_dependent_params()`: `t_yr` 指定時に `Ms` が未定義。**
    `Ms_Msun = Mdot * t / nc.Msun` を設定するだけで `Ms` を作らないため、続く `CR_au` 分岐や
    戻り値の `"Ms": Ms` で `NameError`。

27. **`envos/grid.py:148` — `theta_lim > np.pi / 2 + 1e-8` はタプルとfloatの比較で `TypeError`**
    (`get_interface_coord` を `dr_to_r` 付きで呼ぶと即死)。

28. **`envos/models.py:203-207` — `calc_midplane_average()` が存在しない
    `self.take_midplane_average` を呼ぶ** → `AttributeError`。

29. **`envos/log.py:71-79` — `change_rundir()` が無効。**
    `fn.replace(...)` の戻り値を捨てている(strはイミュータブル)ので、ログファイルは古いパスのまま
    再オープンされる。さらに `FileHandler.__init__(fn)` の再呼び出しでmode/encodingがデフォルトに戻る。

30. **`envos/log.py:123-129` — `unset_logfile()` が `loggers[name]` ではなくグローバル `logger` から
    ハンドラを除去**し、かつイテレーション中のリストを変更。`set_level(ver=1)`(150行目)も
    同じくグローバル `logger` を誤用。

31. **`envos/plot_tools/physical_structure.py:128` — `var[:, model.get_argmid, :]`:
    メソッドを呼ばずにインデックスに使っている** → `TypeError`(`midplane_average=False` で必ず発生)。

32. **`envos/plot_tools/physical_structure.py:321` — `Omega = model.vp / model.R,` の末尾カンマで
    タプルになる。** 後続の `isinstance` 判定を両方すり抜けて `UnboundLocalError`。
    引数の渡し方(`"Omega_prof"` が `save_name` 位置)も意図と不一致の疑い。

33. **`envos/plot_tools/physical_structure.py:349` — `x, y = R * np.sin(tt) * [cos(pp), sin(pp)]`:
    `R` は既に `r sinθ` なので sinθ を二重に掛けており、`plot_losvelocity_midplane_map` の
    座標が歪む。**

34. **`envos/plot_tools/plot_funcs.py:628-640` — `add_mass_estimate_plot()` で `mass_ip`/`mass_vp` の
    片方だけ指定すると `txt_Mip` または `txt_Mvp` が未定義のまま参照され `NameError`。**

35. **`envos/plot_tools/plot_funcs.py:751-762` — `get_subgrid_peaks()`。**
    最適化の初期値に座標値ではなくピクセル添字 `coord_ini` を渡しており、境界(データ座標)と矛盾。
    また740行目以降の象限マスクは `_quadr` ではなく `quadr` を比較しているため、
    `quadr="vmax"` 指定時はマスクが一切効かない。

36. **`envos/plot_tools/plot_funcs.py:143-152` — `logger.debug("max is ", np.max(...), ...)`:
    logging APIの誤用**(複数位置引数)。DEBUGレベルを有効にするとロギングエラーを吐く。

37. **`envos/plot_tools/plot_funcs.py:504-516` — `add_peaks()` の転置が逆。**
    `LocalPeak_Pax` は `Ipv.transpose(0,1)`(無変換、x軸イテレート)を `vkms` とzip、
    `LocalPeak_Vax` はその逆で、両方とも軸が食い違っている。

38. **`envos/plot_tools/obs_output.py:82-91` — `_plot_image()` で `refimage` を渡しつつ
    `contour=False` の場合、`_contopt` 未定義で `NameError`。**

39. **`envos/plot_tools/obs_output.py:380-383` — `get_subgrid_peaks()` は `None` を返しうるのに
    `peaks[0]` を無条件参照**(ピークなしPVで `TypeError`)。

40. **`examples/` の4本中3本が存在しない関数を呼ぶ。**
    `plot_midplane_density_profile`、`plot_density_map`、`plot_temperature_map` 等は
    `plot_tools` に存在しない(実名は `plot_rhogas_midplane_profile`、`plot_rhogas_map` 等)。
    `make_userdefined_model.py` の `plot_mom0_map(odat, pangle_deg=..., poffset_au=...)` も
    現行シグネチャと不一致。ドキュメントとして提供されている例が現コードで動かない。

41. **`mori2023.py:398-412` — `plot_pv=False` かつ `plot_pv_for_mass_estimate=True` だと `pv` 未定義で
    `NameError`。** 同様に `calc_model=False` で `mg.pkl` が無い場合、`calc_obs` 分岐で `mg` が未定義。

---

## C. 特定の使い方で壊れる・サイレントに誤るもの

42. **`envos/tools.py:323-333` — `filecopy()`:コピー先が既にあると「Do nothing.」とログしつつ、
    実際にはコピーを実行する**(早期returnがない)。挙動とログが矛盾。

43. **`envos/tools.py:204-233` — `dataclass_str()`:リスト値は最初の `if` と最後の `else` の
    両方で出力される**(`elif` 漏れ)。

44. **`envos/obs.py:295/320` — チャネル幅の不整合。**
    `nlam = round(vfw/dv)` で `linspace(-vfw/2, vfw/2, nlam)` を使うため実際のチャネル幅は
    `vfw/(nlam-1)`。`Convolver` は `dv_kms` でカーネルを作るので、速度方向の畳み込み幅が
    1チャネル分ずれる。

45. **`envos/obs.py:1008-1011` — `set_refpoint()` が `refpos.dec0` ではなく `refpos.dec` に代入**
    (存在しない属性が黙って生える)。

46. **`envos/obs.py:804-806` — `convto_Tb()` のエラーメッセージがf-string化されておらず、
    `self.repos` というタイポも含む。**

47. **`envos/obs.py:1772-1775` — `minmaxargs()`:範囲内に点が無いと `IndexError`、
    境界値ちょうどは除外**(`>=`でなく`>`)。`trim()` や `get_Imax_pos(interp=True)` から呼ばれる。

48. **`envos/model_generator.py` — `f_dg` は `init_from_config()` でしか設定されない。**
    configなしで組み立てて `calc_kinematic_structure()` を呼ぶと `AttributeError`。

49. **`envos/model_generator.py:166-171` — TSCスムージングで `rho==0` の領域(キャビティ含む)には
    外側エンベロープを混ぜないが、速度は全域で混ぜる。** 密度と速度場の整合性が崩れる。

50. **`envos/grid.py:44-45` — グリッド指定が不足すると `Grid.__init__` が黙って `return None`。**
    属性なしオブジェクトが返り後段で不可解なエラーになる。また `nr=None`/`dr_to_r=None` の
    デフォルト `Config()` では `np.geomspace(*rau_lim, nr+1)` で `TypeError`
    (エラーメッセージが不親切)。

51. **`envos/grid.py:47-51` — ringhost:ログは「4セル追加」だが実際に挿入されるのは3セル**
    (`[:-1]`)。内縁 `4*Rsun` もハードコード(`Rstar_Rsun` 設定と無関係)。

52. **`envos/tsc.py:155-171` — `f_dal0dy`/`f_dV0dy`:`(y-V0)**2 != 1` という浮動小数の完全一致比較。**
    `np.where` は両分岐を評価するため音速点近傍でゼロ除算警告/infが出る。

53. **`envos/cubicsolver.py` — `f==0 and g==0 and h==0` の完全一致比較、`h<=0` かつ `i==0` で
    ゼロ除算などのエッジケース。** `CassenMoosmanInnerEnvelope._sol_with_cubic` は解なしのとき
    `NaN` を返し、密度にNaNが混入しうる。

54. **`envos/column_density.py:4` — `import envos`(パッケージの自己import、循環)と、
    モジュールトップでの `matplotlib.pyplot` import**(計算専用モジュールがGUIバックエンドに依存)。

55. **`envos/column_density.py:115` — `np.arange(z, zlim, 0.1*z)`:`z=0`(赤道面ちょうど)で
    ステップ0となりクラッシュ。**

56. **`envos/datacor.py:46-48` — `preprocess_func` を `im1` に適用した後、その変更済み `im1` を使って
    `im2` 用の係数を計算**(対称であるべき前処理が非対称に)。また `ranges` が軸数より短いと
    `zip_longest` の `None` で軸ごとスキップされ、次元不一致で `interpn` が失敗する。

57. **`envos/radmc3d.py:216-227` — non-LTEモード時、`lines.inp` の分子数が `"1"` 固定なのに
    speclines は3行**(コードコメントでも「Not tested」と明記)。

58. **`envos/radmc3d.py:385-411` — `run_mctherm()` の `os.chdir` が try/finally で守られていない。**
    読み込み失敗時にカレントディレクトリが radmc_dir のまま残る。

59. **`envos/obs.py:426` — `open(os.devnull, "w")` を閉じていない**(マルチプロセス実行ごとにリーク)。

60. **`envos/obs.py:496-523` — `Convolver`:`convmode="null"` でもカーネルを構築するため、
    ビームサイズ `None` だと `None + 1e-100` で `TypeError`。**
    また `Gaussian2DKernel` の `theta` はx軸基準の反時計回りで、天文のPA(北基準)とは
    90°の規約差がある点は要確認。

61. **`envos/obs.py:1082-1087` — `Obreso.__post_init__`:au も deg も両方 `None` の場合
    `np.deg2rad(None)` で `TypeError`**(バリデーションなし)。

62. **`envos/header.py:4` — バージョン判定 `major + minor*0.1 < 3.3`。**
    Python 3.10 で `3+1.0=4.0` となりたまたま通るが、判定式として壊れている。

---

## D. 設計・ドキュメント上の懸念

63. **`envos/config.py` のdocstringと実装の不一致。**
    `disk` のオプションは docstring では `"exptail"`(140行目)だが、コードが受けるのは
    `"powerlaw"`(`model_generator.py:248`)。`molabun: float = ""`(277行目)は型がfloatなのに
    デフォルトが空文字列で、未設定のまま `set_lineobs_inpfiles` に渡ると `nh2 * ""` で
    実行時エラー。`nphot: int = 1e6` もfloat。

64. **`rot_ccw`(`config.py:264`)はコード中で一切参照されておらず、設定しても何も起きない**
    (READMEにも記載があるのに)。

65. **`envos/log.py:186` — `StandardFormatter` のフォールバック書式にデバッグ残骸
    `"... is this used? "`。** CRITICALレベルのログにこの文字列が出る。

66. **グローバル状態への依存が広範囲。**
    `gpath` のモジュール変数、`Config.__post_init__` の副作用、`models.py` のimport時パス束縛。
    同一プロセスで複数の run_dir を扱うと壊れやすい構造。`Config` の生成だけでディレクトリが
    作られる(`gpath.set_rundir` → `make_dirs`)のも副作用として強め。

67. **`gpath.remove_radmcdir()` / `RadmcController.clean_radmc_dir()` は `shutil.rmtree` を
    無確認で実行。** (11)の `set_radmcdir` バグと組み合わさると、ユーザーが指定したつもりの
    ディレクトリと違う場所を消すリスクがある。

68. **`envos/tools.py:329` — `logger.warn` は非推奨**(`warning` を使うべき)。

69. **pickleベースの保存/読み込みが全面的に使われている**(`tsc.read_table` は例外を全て
    握りつぶして `None`)。バージョン間互換性と、信頼できないファイルを読む際の
    任意コード実行リスクという2点で脆い設計。

70. **テストが1つも存在しない。**
    「呼べば必ず落ちる」コードが多数残っているのはこのため。最低限のスモークテスト
    (import、`Config()` 生成、小さなグリッドでの `calc_kinematic_structure`)だけでも
    かなりの数を検出できる。

---

## まとめ

特に優先度が高いのは以下の5点。いずれも例外を出さずに科学的結果を静かに歪めるタイプのバグ。

1. `nconst.year` の定数誤り(A-1)
2. `tools.shell` のエラー検出無効化 — radmc3d失敗のサイレント続行(A-2)
3. `get_mom0_map` の `vlim` 無視(A-3)
4. `_reset_positive_axes` の無効化(A-4)
5. `refpos` 共有ミュータブルデフォルト(A-7)

クラッシュ系(`set_logfile("on")`、`set_radmcdir`、streamline保存、fits保存など)は気づきやすい一方、
観測データ読み込み(`read_fits`)とexamplesは現状ほぼ動作しない状態。修正に着手する場合は
A → B → C → D の順での対応を推奨。

---

## 追補(PLAN.md 作成時の再検証で発見。2026-06-12)

71. **`envos/obs.py:1247` — `Cube.get_pv_map(save=True)` が存在しないメソッド `pv.save_fitsfile()` を呼ぶ**
    → `AttributeError`(Bランク相当)。

72. **`envos/config.py:227-228` — `level_stdout` / `level_logfile` はどこからも読まれていない。**
    設定しても何も起きない(Cランク相当。`rot_ccw`(D-64)と同種の未配線パラメータ)。

73. **`envos/log.py:109-120` — `set_logfile()` の `filename` 引数が本体で未使用**
    (docstringには説明があるのに無視される。Dランク相当)。

74. **呼び出し元ゼロのデッドコード(grep確認済み、Dランク相当)**:
    `grid.get_interface_coord`(B-27のバグ箇所)、`grid.compressed_x2`、
    `grid.Grid.calc_interface_coord` の `thax_ver==2/3` 分岐(`thax_ver=1` 固定で到達不能)、
    `tsc.make_function_loglog`、`obs.find_proper_nthread`、`obs.BaseObsData.convolve_image`。

75. **`envos/tools.py:290` — `shell()` 内のデバッグ残骸 `print("line:", _line)`**
    (到達不能コード側だが、A-2 の修正時に併せて除去。Dランク相当)。

76. **`envos/obs.py:1778-1787` — `__main__` ブロックに個人環境の絶対パスがハードコード**
    (`/home/smori/...`。Dランク相当)。

77. **`envos/obs.py:1174-1200` — `Cube.get_mom0_map(normalize="peak")` が正規化後も `Iunit` を
    更新しない。** 返される `Image` は無次元化されているのに単位は `Jy/pix` のまま
    (`norm_I()` は `Iunit` を更新するのに対し非整合。Cランク相当)。

対応計画は `PLAN.md` を参照(タスクID対応は PLAN.md 付録B)。
