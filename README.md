# Mistral OCR Web App (Seminar Edition)

参加者向けの軽量OCR体験アプリです。Streamlit Community Cloud で公開できます。

使い方（Streamlit Community Cloud）
- このフォルダ一式を GitHub の公開リポジトリに push します。
- share.streamlit.io にログイン → New app → リポとブランチを選択します。
- Main file に _mistral_ocr_webapp.py を指定して Deploy します。
- 発行された URL を参加者に配布します（QRコードが便利）。

ポイント
- 参加者は自分の Mistral API キーをサイドバーで入力して利用します。
- 共通キーのSecrets配布は推奨しません（課金・レート制限の観点）。

ローカル実行（講師PCや個人PC）
- 仮想環境を作成し、requirements.txt をインストールします。
- 実行コマンド例: streamlit run _mistral_ocr_webapp.py
- 同一LANで公開: streamlit run _mistral_ocr_webapp.py --server.address 0.0.0.0 --server.port 8501
- 参加者アクセス例: http://<講師PCのIP>:8501

設定ファイル
- .streamlit/config.toml の server.maxUploadSize を必要に応じて調整（例: 100〜200MB）。

注意事項
- 本アプリは Mistral API を利用するためインターネット接続が必要です。
- 大きなPDFは処理に時間がかかるため、実習では小〜中サイズから開始してください。
- API キーはブラウザのセッション内でのみ使用し、サーバー側に保存しません。

