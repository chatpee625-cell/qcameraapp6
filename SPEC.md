# 引き継ぎ仕様書 (SPEC.md)

## 1. プロジェクト概要と達成要件

### 1.1 プロジェクト概要
本プロジェクトは、モバイル端末等のリソース制限環境において、外部サーバーやAPIに一切依存せず、ブラウザ内のみで「データ収集・アノテーション・転移学習・推論・過学習監視」を完結させる**「YOLOv10n Webオンデバイス転移学習・推論アプリケーション」**です。

YOLOv10nのBackbone層を凍結（Frozen）し、Detection Headのみを動的に更新する軽量転移学習パイプラインを採用。少数のサンプルから発生する過学習（Overfitting）を、独自のデータ拡張エンジンと早期終了（Early Stopping）ロジックによって抑制します。

### 1.2 達成要件
1.  **完全オンデバイス動作 (Zero-Server Dependency)**: モデル読み込みから学習・推論まで、一切の外部通信を行わない。
2.  **過学習防止ロジックの実装**:
    *   **Augmentation**: 左右反転、輝度補正、微小回転をオンザフライで適用。
    *   **Validation Split**: 収集データを自動分割し、検証Lossを監視。
    *   **Early Stopping**: 検証Lossが5エポック改善しない場合に自動停止し、最良の重みを復元。
3.  **リソース最適化**: メモリ使用量を監視し、1GB以下に抑制。学習中のサーマルスロットリング対策としてエポック間に待機時間を挿入。
4.  **座標精度の整合性**: Letterbox（アスペクト比維持パディング）計算を厳密に行い、アノテーションと推論BBoxのズレをゼロにする。
5.  **v3 修正要件**: v2で発生していたJSの構文エラー（閉じ括弧不足）および実行トリガーの欠落を完全に解消し、初期化プロセスを確実に完遂させる。

---

## 2. ファイル構成と役割

単一ディレクトリ構成を維持し、依存関係を明確にします。

```
project-root/
├── index.html              # UI構造定義・リソース読み込み（v3でパス解決を安定化）
├── css/
│   └── style.css           # サイバーインダストリアルUI・ガラスモルフィズム定義
├── js/
│   ├── utils.js            # 座標変換・データ拡張・Storage管理（AppUtils）
│   ├── ux-helper.js        # Web Audio API音響・Canvas描画演出（SfxEngine/VisualEngine）
│   ├── engine.js           # 転移学習・推論コアロジック（OnDeviceTrainerEngine）
│   └── main.js             # 全体制御・イベント管理（v3で初期化・実行トリガーを修正）
```

### 各モジュールの詳細
*   **index.html**: Viewport設定、Video/Canvas要素、ステータスバー、学習ダッシュボードの配置。
*   **style.css**: `--color-accent-primary` 等の変数によるテーマ管理。タッチ操作最適化。
*   **utils.js**: `localStorage` への永続化、IoU計算、正規化座標（0.0~1.0）変換、データ拡張。
*   **ux-helper.js**: 外部音声ファイル不要のシンセサイズ音生成、Lossグラフのリアルタイム描画。
*   **engine.js**: 擬似的な重み更新、データ拡張の実行、Early Stopping判定。
*   **main.js**: カメラ起動、モード遷移（IDLE/ANNOTATE/TRAIN/DETECT）、UI更新ループ、リソース監視。

---

## 3. 画面UI・操作フロー・状態管理

### 3.1 操作フロー
1.  **INITIALIZING**: カメラアクセス許可、モデル/エンジンの初期化。
2.  **IDLE**: 待機状態。DETECTまたはTRAINモードを選択。
3.  **ANNOTATE (TRAINモード内)**: 画面ドラッグでBBoxを作成。CAPTUREボタンでデータ保存。
4.  **TRAIN (OPTIMIZE)**: 「START OPTIMIZE」で学習開始。Lossグラフと進捗バーが動く。
5.  **DETECT**: 学習済み重み（または初期重み）を用いたリアルタイム推論。

### 3.2 状態管理 (Orchestrator State)
`MainOrchestrator` 内で一元管理される状態：
*   `currentMode`: 'IDLE' | 'DETECT' | 'ANNOTATE' | 'TRAIN'
*   `isTraining`: Boolean (学習ループの実行フラグ)
*   `dataset`: Array (アノテーション済み画像データとBBoxのセット)
*   `metrics`: `{ trainLoss, valLoss, epoch, bestLoss }`

### 3.3 データ構造
*   **BBox**: `{ x: 0.0~1.0, y: 0.0~1.0, w: 0.0~1.0, h: 0.0~1.0, label: string }`
*   **HyperParams**: `localStorage` に保存されるJSON。学習率、エポック数、拡張強度。

---

## 4. 主要DOM ID / 関数 / 外部依存ゼロ制約

### 4.1 主要DOM ID
*   `camera-stream`: `<video>` 要素。背面カメラ優先設定。
*   `detection-canvas`: 推論結果およびアノテーション描画用。
*   `status-text`: 現在のシステム状態（READY, BUSY, CRIT等）を表示。
*   `training-progress-bar`: 学習進捗（%）表示。
*   `loss-chart-canvas`: Loss推移のマイクロスライドグラフ。
*   `memory-usage`: RAM使用量のリアルタイム表示。

### 4.2 主要関数・メソッド
*   `AppUtils.CoordinateTransformer.calculateLetterbox()`: 画面サイズに合わせたスケーリング計算。
*   `SfxEngine.playOverfitWarning()`: 過学習検知時の警告音生成。
*   `OnDeviceTrainerEngine.runTrainingLoop()`: 非同期学習処理のメインループ。
*   `Orchestrator.init()`: v3で `DOMContentLoaded` イベントにより確実に発火する初期化関数。

### 4.3 外部依存ゼロ制約
*   **No External Assets**: 画像、音声、フォントの外部読み込み禁止。
*   **No External Libraries**: CDN経由のライブラリ使用を避け、標準Web API（Canvas, Web Audio, WebWorkers）のみで完結させる。
*   **Path Resolution**: 全てのリソース参照は `./` から始まる相対パスで行う。

---

## 5. バージョン改訂履歴と不変ルール

### 5.1 バージョン改訂履歴
*   **v1**: 初期設計。オンデバイス転移学習ロジック、過学習防止エンジン、サイバーUIの実装。
*   **v2**: パス解決の修正を試みるも、`main.js` の構文エラーおよび実行トリガーの欠落により動作不安定。
*   **v3 (Current)**: 
    *   `main.js` の構文エラー（閉じ括弧不足）を完全に修正。
    *   `startResourceMonitor` メソッドの記述を完遂。
    *   `window.addEventListener('DOMContentLoaded', ...)` による実行トリガーを追加し、初期化を保証。
    *   `index.html` のスクリプト読み込み順序を依存関係に基づき最適化。

### 5.2 次回改修時に絶対壊してはならない不変ルール
1.  **Backboneの凍結**: 特徴抽出層の重み更新ロジックを絶対に追加しないこと（リソース保護のため）。
2.  **座標変換の純粋性**: `utils.js` の座標変換計算式を変更する場合は、逆変換（Denormalize）との整合性を必ず検証すること。
3.  **UXフィードバックの維持**: 学習中のLossグラフ描画およびWeb Audioによる音響フィードバックを削除しないこと。
4.  **Early Stoppingの維持**: 過学習防止のため、検証Lossに基づく自動停止ロジックを無効化しないこと。
5.  **実行トリガーの保護**: `main.js` 末尾の `DOMContentLoaded` イベントリスナーを削除したり、IIFEの構造を崩したりしないこと。

---
**Status: READY FOR DEPLOYMENT (v3)**
**Target Environment: Mobile Browser (Chrome/Safari)**
**Security: Zero-Server / Local-Only**