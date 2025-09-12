"""
百科全書・啓蒙研究会用 Mistral OCR Webアプリケーション
セミナー参加者向け - 各自のMistral APIキーで利用

使用方法：
1. streamlit run mistral_ocr_webapp.py
2. ブラウザでhttp://localhost:8501にアクセス
3. 自分のMistral APIキーを入力
4. 18世紀文献のPDFや画像をアップロード
5. OCR結果をダウンロード

特徴：
- 百科全書・啓蒙研究に特化したインターフェース
- フランス語文献対応
- 大きなPDFの自動分割処理
- セッション管理（APIキーはメモリ内のみ）
- 進捗表示とエラーハンドリング
"""

import streamlit as st
import os
import base64
import requests
import json
import time
from pathlib import Path
import re
from typing import Optional, Tuple
from io import BytesIO
import random
from PIL import Image, ImageFilter, ImageOps
import qrcode

# ページ設定
st.set_page_config(
    page_title="百科全書・啓蒙研究会用 Mistral OCR",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定数設定
API_URL = "https://api.mistral.ai/v1/ocr"
DEFAULT_MODEL = "mistral-ocr-latest"
IMAGE_TIMEOUT_SEC = 60
PDF_TIMEOUT_SEC = 120
MAX_RETRIES = 3
LARGE_PDF_THRESHOLD_MB = 5

class MistralOCRWeb:
    """Webアプリ用のMistral OCRクラス"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = API_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MistralOCRWebApp/1.0 (+streamlit)"
        }

    def image_to_base64(self, file_bytes: bytes) -> str:
        """ファイルバイトをbase64エンコードする"""
        return base64.b64encode(file_bytes).decode('utf-8')

    def process_image(self, file_bytes: bytes, filename: str = "image", mime_type: Optional[str] = None) -> str:
        """画像をOCR処理する"""
        max_retries = MAX_RETRIES
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                base64_image = self.image_to_base64(file_bytes)
                # MIMEタイプの決定（既定はimage/jpeg）
                mt = (mime_type or "image/jpeg").lower()
                if mt == "image/jpg":
                    mt = "image/jpeg"
                
                payload = {
                    "model": DEFAULT_MODEL,
                    "document": {
                        "type": "image_url",
                        "image_url": f"data:{mt};base64,{base64_image}"
                    },
                    "include_image_base64": False
                }

                with st.spinner(f"OCR処理中... (試行 {attempt+1}/{max_retries})"):
                    try:
                        response = requests.post(
                            self.api_url,
                            headers=self.headers,
                            json=payload,
                            timeout=IMAGE_TIMEOUT_SEC
                        )
                    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as req_err:
                        if attempt < max_retries - 1:
                            st.warning(f"ネットワークエラー/タイムアウト。リトライ中... ({str(req_err)})")
                            # ジッター付き指数バックオフ
                            sleep_time = retry_delay + random.uniform(0, retry_delay * 0.5)
                            time.sleep(sleep_time)
                            retry_delay *= 2
                            continue
                        else:
                            raise

                    if response.status_code == 200:
                        result = response.json()
                        if "pages" in result and len(result["pages"]) > 0:
                            return result["pages"][0]["markdown"]
                        else:
                            return "テキストが抽出できませんでした。"
                    else:
                        # リトライ対象ステータス: 429, 5xx
                        if response.status_code == 429 or 500 <= response.status_code < 600:
                            if attempt < max_retries - 1:
                                st.warning(f"API一時エラー {response.status_code}。リトライします。")
                                sleep_time = retry_delay + random.uniform(0, retry_delay * 0.5)
                                time.sleep(sleep_time)
                                retry_delay *= 2
                                continue
                        # それ以外の4xxは即時失敗
                        raise Exception(f"API呼び出しエラー: {response.status_code} - {response.text}")
                            
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"エラーが発生しました。リトライ中... ({str(e)})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    def process_pdf(self, file_bytes: bytes, filename: str = "document.pdf", add_page_breaks: bool = True) -> str:
        """PDFをOCR処理する"""
        max_retries = MAX_RETRIES
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                base64_pdf = self.image_to_base64(file_bytes)
                
                payload = {
                    "model": DEFAULT_MODEL,
                    "document": {
                        "type": "document_url",
                        "document_url": f"data:application/pdf;base64,{base64_pdf}"
                    },
                    "include_image_base64": False
                }

                with st.spinner(f"PDF OCR処理中... (試行 {attempt+1}/{max_retries})"):
                    try:
                        response = requests.post(
                            self.api_url,
                            headers=self.headers,
                            json=payload,
                            timeout=PDF_TIMEOUT_SEC
                        )
                    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as req_err:
                        if attempt < max_retries - 1:
                            st.warning(f"ネットワークエラー/タイムアウト。リトライ中... ({str(req_err)})")
                            sleep_time = retry_delay + random.uniform(0, retry_delay * 0.5)
                            time.sleep(sleep_time)
                            retry_delay *= 2
                            continue
                        else:
                            raise

                    if response.status_code == 200:
                        result = response.json()
                        if "pages" in result:
                            all_text = ""
                            for i, page in enumerate(result["pages"]):
                                page_content = page.get("markdown", "テキストが抽出できませんでした。")
                                if add_page_breaks and i > 0:
                                    all_text += f"\n\n<pb>{i+1}</pb>\n\n"
                                all_text += page_content
                            return all_text
                        else:
                            return "テキストが抽出できませんでした。"
                    else:
                        if response.status_code == 429 or 500 <= response.status_code < 600:
                            if attempt < max_retries - 1:
                                st.warning(f"API一時エラー {response.status_code}。リトライします。")
                                sleep_time = retry_delay + random.uniform(0, retry_delay * 0.5)
                                time.sleep(sleep_time)
                                retry_delay *= 2
                                continue
                        raise Exception(f"API呼び出しエラー: {response.status_code} - {response.text}")
                            
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"エラーが発生しました。リトライ中... ({str(e)})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

def process_large_pdf_web(file_bytes: bytes, filename: str, ocr: MistralOCRWeb, add_page_breaks: bool = True) -> str:
    """大きなPDFを分割して処理する（Webアプリ用）"""
    try:
        from PyPDF2 import PdfReader, PdfWriter
        # メモリ内でPDFを読み込み
        pdf = PdfReader(BytesIO(file_bytes))
        total_pages = len(pdf.pages)

        st.info(f"📄 PDFの総ページ数: {total_pages}")

        # 進捗バー
        progress_bar = st.progress(0)
        progress_text = st.empty()

        all_text_parts = []

        for page_num in range(total_pages):
            progress = (page_num + 1) / total_pages
            progress_bar.progress(progress)
            progress_text.text(f"ページ {page_num+1}/{total_pages} を処理中...")

            # 1ページずつメモリ内で抽出
            pdf_writer = PdfWriter()
            pdf_writer.add_page(pdf.pages[page_num])
            buf = BytesIO()
            pdf_writer.write(buf)
            page_bytes = buf.getvalue()

            try:
                part_text = ocr.process_pdf(page_bytes, f"{filename}_page_{page_num+1}", add_page_breaks=add_page_breaks)

                # ページ番号タグを追加
                if add_page_breaks and not re.search(r'<pb>\d+</pb>', part_text):
                    part_text = f"<pb>{page_num+1}</pb>\n\n{part_text}"

                all_text_parts.append(part_text)

            except Exception as e:
                st.warning(f"ページ {page_num+1} の処理中にエラー: {str(e)}")
                all_text_parts.append(f"<pb>{page_num+1}</pb>\n\n**Error:** {str(e)}\n\n")

        progress_bar.progress(1.0)
        progress_text.text("✅ 全ページの処理が完了しました！")

        return "\n".join(all_text_parts)
                
    except Exception as e:
        st.error(f"PDF分割処理エラー: {str(e)}")
        raise


def preprocess_image_bytes(
    file_bytes: bytes,
    *,
    long_edge_px: int,
    grayscale: bool,
    denoise: bool,
    binarize: bool,
    threshold: int,
    orig_mime: Optional[str] = None,
) -> Tuple[bytes, str]:
    """PILで軽量な画像前処理を行い、(バイト列, MIME) を返す。
    - 長辺リサイズ、グレースケール、メディアンノイズ除去、閾値二値化
    """
    with Image.open(BytesIO(file_bytes)) as im:
        im.load()

        # リサイズ（長辺を指定ピクセル以内に）
        w, h = im.size
        long_edge = max(w, h)
        if long_edge_px > 0 and long_edge > long_edge_px:
            scale = long_edge_px / long_edge
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            im = im.resize(new_size, Image.LANCZOS)

        # グレースケール
        if grayscale:
            im = ImageOps.grayscale(im)

        # ノイズ除去（軽め）
        if denoise:
            im = im.filter(ImageFilter.MedianFilter(size=3))

        # 二値化（必要時）
        if binarize:
            if im.mode != "L":
                im = ImageOps.grayscale(im)
            thr = max(0, min(255, int(threshold)))
            im = im.point(lambda p: 255 if p > thr else 0, mode="L")

        # 保存形式の決定（既定は入力MIMEを尊重。なければPNG）
        if orig_mime and orig_mime.lower() in ("image/jpeg", "image/jpg"):
            fmt = "JPEG"
            mime_out = "image/jpeg"
            if im.mode not in ("L", "RGB"):
                im = im.convert("RGB")
            params = {"quality": 90, "optimize": True}
        else:
            fmt = "PNG"
            mime_out = "image/png"
            params = {"optimize": True}

        buf = BytesIO()
        im.save(buf, format=fmt, **params)
        return buf.getvalue(), mime_out

def main():
    # ヘッダー
    st.title("📚 百科全書・啓蒙研究会用 Mistral OCR")
    st.markdown("*18世紀フランス文献のデジタル化支援ツール*")
    
    # クイックスタートとQRコード
    with st.container():
        col_qs, col_qr = st.columns([2, 1])
        with col_qs:
            st.info(
                """
                使い方（概要）
                1) 左のサイドバーに自分の Mistral APIキーを入力
                2) 右の「ファイルアップロード」に PDF/PNG/JPEG を追加
                3) 必要なら「画像前処理」「ページ区切り」をON
                4) 「OCR処理を開始」をクリック → プレビュー → ダウンロード
                """
            )
            # サンプルファイルへの案内（任意リンク）
            sample_link = os.environ.get("SAMPLE_FILES_URL") or st.secrets.get("SAMPLE_FILES_URL", None)
            if sample_link:
                st.markdown(f"📎 サンプルファイル: [{sample_link}]({sample_link})")
            else:
                st.caption("サンプルファイルは配布資料のリンクからダウンロードしてください。")
        with col_qr:
            # アプリURLのQRコード（Secrets/環境変数または編集可能な既定値）
            default_app_url = os.environ.get("APP_URL") or st.secrets.get("APP_URL", "https://tatsuohemmi-geel--mistral-ocr-webapp-kxxcwt.streamlit.app/")
            with st.expander("QRコード（アクセス用）", expanded=True):
                app_url = st.text_input("配布用URL", value=default_app_url, help="参加者に配布するURL。変更するとQRも更新されます。")
                try:
                    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=4, border=2)
                    qr.add_data(app_url)
                    qr.make(fit=True)
                    qr_img = qr.make_image(fill_color="black", back_color="white")
                    buf = BytesIO()
                    qr_img.save(buf, format="PNG")
                    st.image(buf.getvalue(), caption="このQRからアクセスできます")
                except Exception as _:
                    st.warning("QRコードの生成に失敗しました。URLをそのまま配布してください。")
    
    # サイドバー：設定
    with st.sidebar:
        st.header("🔧 設定")
        
        # APIキー入力
        st.subheader("Mistral API設定")
        detected_source = None
        if not st.session_state.get("api_key"):
            if "MISTRAL_API_KEY" in st.secrets:
                st.session_state["api_key"] = st.secrets["MISTRAL_API_KEY"]
                detected_source = "secrets"
            elif os.environ.get("MISTRAL_API_KEY"):
                st.session_state["api_key"] = os.environ.get("MISTRAL_API_KEY")
                detected_source = "env"

        api_key = st.text_input(
            "Mistral APIキー",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="https://mistral.ai/ でAPIキーを取得してください"
        )
        if api_key != st.session_state.get("api_key", ""):
            st.session_state["api_key"] = api_key

        if not st.session_state.get("api_key"):
            st.warning("⚠️ APIキーを入力してください")
            st.markdown("""
            **APIキーの取得方法：**
            1. https://mistral.ai/ にアクセス
            2. アカウント作成・ログイン
            3. API Keysページでキーを生成
            4. 上記フィールドに入力
            """)
            return

        if detected_source:
            st.success("✅ APIキーを自動設定しました（Secrets/環境変数）")
        else:
            st.success("✅ APIキーが設定されました")
        
        # ファイルサイズ制限の説明
        st.subheader("📋 使用方法")
        st.markdown("""
        **対応フォーマット：**
        - PDF（18世紀稀覯書原本等）
        - PNG, JPEG（刊本画像等）
        
        **ファイルサイズ：**
        - 5MB以下：高速処理
        - 5MB以上：自動分割処理
        
        **用途例：**
        - Gallicaからの稀覯書PDF原本
        - 18世紀 Google Books 本などのデジタル化
        - 古典籍画像のテキスト化
        """)
    
    # メインエリア
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 ファイルアップロード")
        
        uploaded_files = st.file_uploader(
            "OCR処理したい文献ファイルを選択してください",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="複数ファイルの同時処理が可能です"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} ファイルが選択されました")
            
            # ファイル情報表示
            for file in uploaded_files:
                file_size_mb = len(file.read()) / (1024 * 1024)
                file.seek(0)  # ポインタを先頭に戻す
                
                st.markdown(f"""
                **{file.name}**
                - サイズ: {file_size_mb:.2f} MB
                - タイプ: {file.type}
                """)
    
    with col2:
        st.header("⚙️ OCR処理オプション")
        
        # 処理オプション
        st.subheader("処理設定")
        
        auto_split = st.checkbox(
            "大きなPDFの自動分割",
            value=True,
            help="5MB以上のPDFを自動的にページごとに分割して処理します"
        )
        
        add_page_breaks = st.checkbox(
            "ページ区切りタグ挿入",
            value=True,
            help="複数ページのドキュメントに <pb>n</pb> タグを挿入します"
        )
        
        # 出力形式
        st.subheader("出力形式")
        output_format = st.selectbox(
            "ファイル形式",
            ["Markdown (.md)", "テキスト (.txt)", "JSON (.json)"],
            help="OCR結果の保存形式を選択してください"
        )

        # 画像前処理
        st.subheader("画像前処理")
        enable_preprocess = st.checkbox(
            "画像前処理を有効化",
            value=False,
            help="OCR前に画像をリサイズ/グレースケール/ノイズ除去/二値化します"
        )
        if enable_preprocess:
            resize_long_edge = st.slider("長辺ピクセル数", min_value=800, max_value=3000, value=1800, step=100)
            pp_grayscale = st.checkbox("グレースケール", value=True)
            pp_denoise = st.checkbox("軽いノイズ除去", value=True)
            pp_binarize = st.checkbox("二値化（単純閾値）", value=False)
            pp_threshold = st.slider("二値化しきい値", min_value=50, max_value=230, value=180, step=5)
        else:
            resize_long_edge = 0
            pp_grayscale = False
            pp_denoise = False
            pp_binarize = False
            pp_threshold = 180
    
    # 処理実行ボタン
    st.markdown("---")
    
    if uploaded_files and api_key:
        if st.button("🚀 OCR処理を開始", type="primary"):
            ocr = MistralOCRWeb(api_key)
            
            # 結果格納用
            results = {}
            
            # 全体の進捗
            total_files = len(uploaded_files)
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                overall_status.text(f"ファイル {i+1}/{total_files}: {uploaded_file.name}")
                
                try:
                    file_bytes = uploaded_file.read()
                    file_size_mb = len(file_bytes) / (1024 * 1024)
                    
                    # ファイル処理
                    with st.expander(f"📄 {uploaded_file.name} の処理結果", expanded=True):
                        if uploaded_file.type == "application/pdf":
                            if file_size_mb > LARGE_PDF_THRESHOLD_MB and auto_split:
                                st.info(f"📊 大きなPDFファイル ({file_size_mb:.2f} MB) - 分割処理を実行中...")
                                extracted_text = process_large_pdf_web(file_bytes, uploaded_file.name, ocr, add_page_breaks=add_page_breaks)
                            else:
                                st.info("📄 PDF処理中...")
                                extracted_text = ocr.process_pdf(file_bytes, uploaded_file.name, add_page_breaks=add_page_breaks)
                        else:
                            st.info("🖼️ 画像処理中...")
                            img_bytes = file_bytes
                            mime_for_ocr = uploaded_file.type
                            if enable_preprocess:
                                try:
                                    img_bytes, mime_for_ocr = preprocess_image_bytes(
                                        file_bytes,
                                        long_edge_px=resize_long_edge,
                                        grayscale=pp_grayscale,
                                        denoise=pp_denoise,
                                        binarize=pp_binarize,
                                        threshold=pp_threshold,
                                        orig_mime=uploaded_file.type,
                                    )
                                    st.caption("前処理を適用しました。")
                                except Exception as pe:
                                    st.warning(f"画像前処理でエラー: {str(pe)}。元画像で処理を続行します。")
                            extracted_text = ocr.process_image(img_bytes, uploaded_file.name, mime_type=mime_for_ocr)
                        
                        # 結果保存
                        results[uploaded_file.name] = extracted_text
                        
                        # プレビュー表示
                        st.subheader("📋 抽出テキストプレビュー")
                        preview_text = extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
                        st.text_area(
                            f"{uploaded_file.name} - プレビュー",
                            preview_text,
                            height=200,
                            key=f"preview_{i}"
                        )
                        
                        # 統計情報
                        char_count = len(extracted_text)
                        line_count = len(extracted_text.split('\n'))
                        page_count = len(re.findall(r'<pb>\d+</pb>', extracted_text))
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("文字数", f"{char_count:,}")
                        with col_b:
                            st.metric("行数", f"{line_count:,}")
                        with col_c:
                            st.metric("ページ数", f"{page_count}" if page_count > 0 else "1")
                
                except Exception as e:
                    st.error(f"❌ {uploaded_file.name} の処理中にエラー: {str(e)}")
                    results[uploaded_file.name] = f"処理エラー: {str(e)}"
                finally:
                    # 個別ファイル処理後に全体進捗を更新
                    overall_progress.progress((i + 1) / total_files)
            
            # 全体処理完了
            overall_progress.progress(1.0)
            overall_status.text("✅ 全ファイルの処理が完了しました！")
            
            # ダウンロードセクション
            st.markdown("---")
            st.header("📥 結果ダウンロード")
            
            for idx, (filename, content) in enumerate(results.items()):
                if not content.startswith("処理エラー"):
                    # ファイル名生成
                    base_name = Path(filename).stem
                    
                    if output_format == "Markdown (.md)":
                        download_filename = f"{base_name}_ocr_mistral.md"
                        download_content = content
                        mime_type = "text/markdown"
                    elif output_format == "テキスト (.txt)":
                        download_filename = f"{base_name}_ocr_mistral.txt"
                        download_content = content
                        mime_type = "text/plain"
                    else:  # JSON
                        download_filename = f"{base_name}_ocr_mistral.json"
                        download_content = json.dumps({
                            "filename": filename,
                            "extracted_text": content,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "character_count": len(content),
                            "line_count": len(content.split('\n'))
                        }, ensure_ascii=False, indent=2)
                        mime_type = "application/json"
                    
                    st.download_button(
                        f"📄 {filename} をダウンロード",
                        download_content,
                        file_name=download_filename,
                        mime=mime_type,
                        key=f"download_{idx}_{download_filename}"
                    )
    else:
        if not uploaded_files:
            st.info("👆 まずはファイルをアップロードしてください")
        if not api_key:
            st.info("👈 サイドバーでMistral APIキーを設定してください")

    # フッター
    st.markdown("---")
    st.markdown("""
    **💡 ヒント：**
    - 18世紀の古典籍は解像度300dpi以上を推奨
    - フランス語文献は特殊文字（é, è, ç等）も正確に認識されます
    - 大きなファイルは分割処理により時間がかかる場合があります
    
    **📚 百科全書・啓蒙研究会セミナーシリーズ**
    - 第1回：Zotero + AI引用管理
    - 第2回：Mistral OCRで古典籍解読（本セッション）
    - 第3回：AI翻訳で多言語文献研究
    """)

if __name__ == "__main__":
    main()
