"""
百科全書・啓蒙研究会用 フランス語テキスト近代化（正書法・表記の標準化）Webアプリケーション（Gemini版）

セミナー参加者向け - 各自のGemini APIキーで利用

使用方法：
1. streamlit run historical_text_modernizer_Gemini_webapp.py
2. ブラウザで http://localhost:8501 にアクセス
3. 自分のGemini APIキーを入力（gemini-2.5-flash-lite を利用）
4. Mistral OCRの結果ファイル（Markdown もしくはテキスト）をアップロード
5. 近代化（綴り正規化）結果をダウンロード

特徴：
- 百科全書・啓蒙研究に特化したインターフェース
- フランス語文献対応（17–19世紀の古典に配慮）
- 大きなテキストの自動分割処理（段落単位の安全なチャンク化）
- セッション管理（APIキーはメモリ内のみ保持）
- 進捗表示（チャンク単位）とエラーハンドリング
"""

import os
import time
import json
import streamlit as st
from pathlib import Path
from typing import List
import concurrent.futures

# 既存のロジック（チャンク分割、Gemini呼び出し）を再利用
import _historical_text_modernizer_Gemini as htm


# ページ設定
st.set_page_config(
    page_title="百科全書・啓蒙研究会用 フランス語テキスト近代化 (Gemini)",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_api_key_from_sidebar() -> str:
    """サイドバーでGemini APIキーを取得（secrets/env からの自動補完にも対応）。"""
    st.sidebar.header("🔑 API キー設定 (Gemini)")

    # secrets / 環境変数の自動検出
    detected_source = None
    preset_key = ""
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            if key_name in st.secrets and st.secrets[key_name]:
                preset_key = st.secrets[key_name]
                detected_source = f"st.secrets['{key_name}']"
                break
        except Exception:
            pass
        if os.environ.get(key_name):
            preset_key = os.environ.get(key_name, "")
            detected_source = f"os.environ['{key_name}']"
            break

    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="secrets または環境変数から自動設定がある場合は上書き可能です",
        value=preset_key,
    )

    if api_key:
        if detected_source:
            st.sidebar.caption(f"検出済みのキーを使用中: {detected_source}")
        else:
            st.sidebar.caption("ユーザー入力のキーを使用中")
    else:
        st.sidebar.warning("Gemini APIキーを入力してください")

    # セッション状態に保持（メモリ内のみ）
    st.session_state["gemini_api_key"] = api_key
    return api_key


def process_text_with_progress(modernizer: htm.TextModernizer, input_text: str, max_workers: int) -> str:
    """チャンク進捗をStreamlitで可視化しつつ近代化処理を実行。"""
    chunks: List[str] = modernizer.chunk_text(input_text)
    if not chunks:
        return ""

    st.info(f"テキストを {len(chunks)} チャンクに分割しました。")
    progress_bar = st.progress(0)
    status = st.empty()
    error_list = []
    results: List[str] = [""] * len(chunks)

    # 並列処理（I/Oバウンド）
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(modernizer.process_chunk, chunk, idx): idx
            for idx, chunk in enumerate(chunks)
        }
        done = 0
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = ""
                error_list.append(idx + 1)
                st.warning(f"チャンク {idx+1} でエラー: {e}")
            finally:
                done += 1
                progress_bar.progress(done / len(chunks))
                status.text(f"処理済み: {done}/{len(chunks)} チャンク")

    if error_list:
        st.warning(
            f"一部のチャンクに失敗がありました: {', '.join(map(str, error_list))}"
        )

    # 空のチャンクはプレースホルダを入れておく（後から手動確認しやすく）
    combined = "".join([r if r else "【処理失敗】\n\n" for r in results]).rstrip()
    return combined


def main():
    st.title("🖋️ 百科全書・啓蒙研究会用 フランス語テキスト近代化 (Gemini)")
    st.markdown("*17–19世紀フランス語資料の正書法・表記の標準化支援ツール*")

    # クイックガイド
    with st.expander("使い方", expanded=True):
        st.markdown(
            """
            1) 左サイドバーで自分の Gemini API キーを入力
            2) 下の「ファイルアップロード」に Mistral OCR の結果（.md / .txt）を追加
            3) 「近代化を実行」を押す → プレビュー → ダウンロード
            """
        )

    # サイドバー: APIキーと設定
    api_key = get_api_key_from_sidebar()
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 設定")

    # モデル（既定は gemini-2.5-flash-lite）
    model_name = st.sidebar.selectbox(
        "モデル",
        options=["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="軽量高速なら flash-lite、精度重視なら pro を選択",
    )
    # 並列数
    max_workers = st.sidebar.slider("並列処理数", 1, 8, value=min(5, htm.MAX_WORKERS))

    # メイン：ファイル入出力
    st.header("📤 ファイルアップロード")
    uploaded_files = st.file_uploader(
        "Mistral OCRの結果ファイル（Markdown / テキスト）を選択してください",
        type=["md", "txt"],
        accept_multiple_files=True,
    )

    st.markdown("---")
    if uploaded_files and api_key:
        if st.button("🚀 近代化を実行", type="primary"):
            # モデルを指定（モジュールの定数を上書きして使う）
            htm.GEMINI_MODEL_NAME = model_name

            try:
                modernizer = htm.TextModernizer(api_key, output_path=os.path.join(os.getcwd(), "French_Modernization_Output_Gemini"))
            except Exception as e:
                st.error(f"初期化エラー: {e}")
                return

            results = {}
            overall = st.progress(0)
            overall_status = st.empty()

            for i, f in enumerate(uploaded_files):
                overall_status.text(f"処理中: {i+1}/{len(uploaded_files)} - {f.name}")
                with st.expander(f"📄 {f.name} の結果", expanded=True):
                    try:
                        text = f.read().decode("utf-8")
                    except Exception:
                        f.seek(0)
                        text = f.read().decode("utf-8", errors="ignore")

                    if not text.strip():
                        st.warning("入力テキストが空です。スキップします。")
                        results[f.name] = ""
                    else:
                        start = time.time()
                        try:
                            modernized = process_text_with_progress(modernizer, text, max_workers)
                            duration = time.time() - start
                            st.success(f"完了 ({duration:.1f}秒)")

                            # プレビュー
                            preview = modernized[:800] + ("..." if len(modernized) > 800 else "")
                            st.subheader("📋 プレビュー")
                            st.text_area("プレビュー", preview, height=240, key=f"preview_{i}")

                            # 保存用データ
                            results[f.name] = modernized
                        except Exception as e:
                            st.error(f"処理エラー: {e}")
                            results[f.name] = ""

                overall.progress((i + 1) / len(uploaded_files))

            # ダウンロード
            st.markdown("---")
            st.header("📥 結果ダウンロード")

            for idx, (name, content) in enumerate(results.items()):
                if not content:
                    st.caption(f"{name}: 出力なし（エラー/空入力）")
                    continue
                base = Path(name).stem
                out_name = f"{base}_modernized_gemini.md"
                st.download_button(
                    f"📄 {name} をダウンロード",
                    content,
                    file_name=out_name,
                    mime="text/markdown",
                    key=f"dl_{idx}_{out_name}",
                )
    else:
        if not uploaded_files:
            st.info("👆 まずは Mistral OCR の結果ファイル（.md / .txt）をアップロードしてください")
        if not api_key:
            st.info("👈 サイドバーで Gemini API キーを設定してください")

    # 補足情報
    st.markdown("---")
    st.markdown(
        """
        ヒント：
        - 入力は Mistral OCR 由来の Markdown を想定（ページ区切りタグ <pb>n</pb> 等は保持）
        - 語彙・綴りの現代化を行いますが、固有名詞は原則保持します
        - レイアウト保持を重視するため、段落単位で安全に分割・処理します
        - APIキーはセッションメモリのみで扱い、永続保存しません
        """
    )


if __name__ == "__main__":
    main()

