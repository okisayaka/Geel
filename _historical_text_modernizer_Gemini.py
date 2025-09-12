#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to modernize old French text using the Gemini API while preserving layout.
Based on the superior DeepSeek implementation structure.

Requirements:
- Python 3.7+
- Install dependencies: pip install google-generativeai tqdm
- (Optional for GUI file dialog) tkinter:
    - Linux: sudo apt-get install python3-tk
    - Mac: brew install python-tk
    - Windows: Ensure tkinter is selected during Python installation.

Setup:
1. API Key is directly configured in the script (security risk acknowledged).
2. To change the API key, modify the GOOGLE_API_KEY variable in the script.
"""

import os
import json
import time
import logging
import argparse
import sys
import re
import concurrent.futures
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple

# --- Configuration ---
# API Key (直接設定 - リスクを認識の上)
GOOGLE_API_KEY = "AIzaSyCyvbyq5KVSsM4IUBMSBqwKqUNHzSMZb-A"

# Available Gemini models (use one of these):
# - gemini-2.5-flash-lite (recommended: high volume, cost efficient, low latency)
# - gemini-2.5-flash (fast performance on everyday tasks)
# - gemini-2.5-pro (best for coding and highly complex tasks)
# - gemini-1.5-flash (legacy model)
# - gemini-1.5-pro (legacy model)
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
DEFAULT_OUTPUT_DIR = "French_Modernization_Output_Gemini"
LOG_FILE_NAME = "modernizer_gemini.log"

# API Settings
MAX_RETRIES = 6  # DeepSeek版に合わせて6に増加
INITIAL_RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 30  # seconds
REQUEST_TIMEOUT = 120  # seconds

# Rate Limiting (Gemini API limits)
# Gemini Free Tier Limits (approximate, check official docs):
# - RPM (Requests Per Minute): ~60
# - TPM (Tokens Per Minute): ~40,000 (input + output)
MAX_REQUESTS_PER_MINUTE = 40
TOKENS_PER_MINUTE_LIMIT = 40000

# Chunking & Token Estimation
# Note: These are rough estimates. Using a dedicated tokenizer is more accurate.
# Factors based roughly on OpenAI tokenizers for similar languages.
CHAR_TOKEN_RATIO_LATIN = 0.35  # French, English, etc.
CHAR_TOKEN_RATIO_CJK = 1.0  # Chinese, Japanese, Korean (adjust if needed)
# Gemini 2.5 Flash-Lite supports 1M context window, so we can use larger chunks
TARGET_CHUNK_TOKENS = 2000  # Increased for better efficiency
MAX_CHUNK_TOKENS = 5000  # Increased for better efficiency
MAX_COMPLETION_TOKENS = 8000  # Increased for better efficiency

# Parallel Processing
# Gemini 2.5 Flash-Lite has low latency, so we can increase parallel processing
MAX_WORKERS = 5  # Increased for better throughput

# --- Dependency Check ---
REQUIRED_PACKAGES = ["google.generativeai", "tqdm"]
missing_packages = []
for package in REQUIRED_PACKAGES:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("エラー: 以下の必要なライブラリがインストールされていません:")
    for package in missing_packages:
        print(f"- {package}")
    print("\n以下のコマンドを実行してインストールしてください:")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)

import google.generativeai as genai
from tqdm import tqdm

# --- Tkinter Check (for GUI) ---
HAS_TKINTER = False
try:
    import tkinter as tk
    from tkinter import filedialog

    # Try creating a root window to ensure display is available
    try:
        root = tk.Tk()
        root.withdraw()
        root.destroy()
        HAS_TKINTER = True
    except tk.TclError:
        print(
            "警告: tkinterはインストールされていますが、表示環境が利用できない可能性があります (例: SSH接続)。"
        )
        print(
            "ファイル/ディレクトリ選択ダイアログは利用できません。コマンドライン引数を使用してください。"
        )
except ImportError:
    # Warning only printed if GUI is attempted later without args
    pass


# --- Rate Limiter (Simplified for TPM Estimation) ---
class TokenRateLimiter:
    """Estimates token usage to proactively avoid hitting TPM limits."""

    def __init__(self, tpm_limit: int = TOKENS_PER_MINUTE_LIMIT, time_window: int = 60):
        self.tpm_limit = tpm_limit
        self.time_window = time_window
        self.token_usage: List[Tuple[int, datetime]] = []  # (tokens, timestamp)

    def _cleanup_old_records(self, now: datetime):
        """Remove token records older than the time window."""
        cutoff_time = now - timedelta(seconds=self.time_window)
        self.token_usage = [
            (tokens, ts) for tokens, ts in self.token_usage if ts > cutoff_time
        ]

    def wait_if_needed(self, estimated_request_tokens: int):
        """Waits if the estimated token usage might exceed the TPM limit."""
        now = datetime.now()
        self._cleanup_old_records(now)

        current_tokens_in_window = sum(tokens for tokens, _ in self.token_usage)
        potential_total = current_tokens_in_window + estimated_request_tokens

        if potential_total > self.tpm_limit:
            # Calculate how long to wait for the oldest record(s) to expire
            tokens_to_remove = potential_total - self.tpm_limit
            tokens_removed = 0
            wait_until = now  # Default to now if no wait needed

            # Sort by timestamp to find oldest records
            self.token_usage.sort(key=lambda x: x[1])

            for tokens, ts in self.token_usage:
                tokens_removed += tokens
                if tokens_removed >= tokens_to_remove:
                    wait_until = ts + timedelta(seconds=self.time_window)
                    break

            sleep_duration = (wait_until - now).total_seconds()
            if sleep_duration > 0:
                logging.info(
                    f"TPM推定: {potential_total} > {self.tpm_limit}. {sleep_duration:.2f}秒待機します..."
                )
                time.sleep(sleep_duration)
                # Re-cleanup after waiting
                self._cleanup_old_records(datetime.now())

    def record_usage(self, tokens_used: int):
        """Records the actual token usage for a request."""
        now = datetime.now()
        self.token_usage.append((tokens_used, now))
        self._cleanup_old_records(now)  # Keep the list tidy


# --- Text Modernizer Class ---
class TextModernizer:
    """Modernizes French text using Gemini API, preserving layout."""

    def __init__(self, api_key: str, output_path: Optional[str] = None):
        """Initialize the TextModernizer."""
        if not api_key:
            raise ValueError("APIキーが提供されていません。")

        # Configure Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        self.token_rate_limiter = TokenRateLimiter()
        self.total_tokens_used = 0

        # Setup output directory
        self.output_path = output_path or os.path.join(os.getcwd(), DEFAULT_OUTPUT_DIR)
        os.makedirs(self.output_path, exist_ok=True)

        # Setup logging
        log_file = os.path.join(self.output_path, LOG_FILE_NAME)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),  # Also print logs to console
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("=" * 20 + " Text Modernizer (Gemini) Initialized " + "=" * 20)
        self.logger.info(f"Model: {GEMINI_MODEL_NAME}")
        self.logger.info(f"Output Directory: {self.output_path}")
        self.logger.info(f"タイムアウト設定: {REQUEST_TIMEOUT}秒")
        self.logger.info(f"最大リトライ回数: {MAX_RETRIES}")
        self.logger.info(f"並列処理数: {MAX_WORKERS}")
        self.logger.info(
            f"チャンクサイズ: 目標{TARGET_CHUNK_TOKENS}トークン, 最大{MAX_CHUNK_TOKENS}トークン"
        )

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in the input text.
        NOTE: This is a rough estimation. Accuracy varies.
        """
        import re

        if not text:
            return 0

        # Basic CJK character detection (adjust ranges if needed)
        cjk_pattern = re.compile(
            r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+"
        )
        cjk_chars = "".join(cjk_pattern.findall(text))
        cjk_count = len(cjk_chars)
        latin_count = len(text) - cjk_count

        estimated_tokens = (
            cjk_count * CHAR_TOKEN_RATIO_CJK + latin_count * CHAR_TOKEN_RATIO_LATIN
        )
        # Add a small buffer
        return int(estimated_tokens) + 10

    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks based on estimated tokens, preserving paragraphs."""
        if not text:
            return []

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk_paragraphs = []
        current_chunk_tokens = 0

        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():  # Keep empty paragraphs as separators
                current_chunk_paragraphs.append(paragraph)
                continue

            paragraph_tokens = self.estimate_tokens(paragraph)

            # Check if paragraph itself is too large
            if paragraph_tokens > MAX_CHUNK_TOKENS:
                self.logger.warning(
                    f"段落 {i+1} は単独で大きすぎます ({paragraph_tokens} トークン推定)。"
                    f"強制的に分割しますが、レイアウトが崩れる可能性があります。"
                )
                # Simple split by sentences (can be improved with NLTK etc.)
                # Be cautious with abbreviations (e.g., "M.", "etc.")
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                temp_chunk = ""
                temp_tokens = 0
                for sentence in sentences:
                    sentence_tokens = self.estimate_tokens(sentence)
                    if temp_tokens + sentence_tokens > MAX_CHUNK_TOKENS and temp_chunk:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = sentence + " "
                        temp_tokens = sentence_tokens
                    else:
                        temp_chunk += sentence + " "
                        temp_tokens += sentence_tokens
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
                # Reset current chunk after handling oversized paragraph
                current_chunk_paragraphs = []
                current_chunk_tokens = 0
                continue

            # Check if adding this paragraph would exceed the target
            if (
                current_chunk_tokens + paragraph_tokens > TARGET_CHUNK_TOKENS
                and current_chunk_paragraphs
            ):
                # Finalize current chunk
                chunks.append("\n\n".join(current_chunk_paragraphs))
                current_chunk_paragraphs = [paragraph]
                current_chunk_tokens = paragraph_tokens
            else:
                # Add to current chunk
                current_chunk_paragraphs.append(paragraph)
                current_chunk_tokens += paragraph_tokens

        # Add the last chunk if it has content
        if current_chunk_paragraphs:
            chunks.append("\n\n".join(current_chunk_paragraphs))

        # Ensure each chunk ends with double newline for consistency
        chunks = [chunk + "\n\n" if not chunk.endswith("\n\n") else chunk for chunk in chunks]

        self.logger.info(f"テキストを {len(chunks)} チャンクに分割しました。")
        for i, chunk in enumerate(chunks):
            estimated_tokens = self.estimate_tokens(chunk)
            self.logger.debug(f"チャンク {i+1}: {estimated_tokens} トークン推定")

        return chunks

    def create_prompt(self, input_chunk: str) -> str:
        """Create the prompt for the Gemini API."""
        prompt = f"""Tu es un expert en modernisation de textes français anciens (XVIIe-XIXe siècles).
Ta tâche est de moderniser l'orthographe, la grammaire et le vocabulaire du texte fourni vers un français contemporain STANDARD, tout en respectant IMPÉRATIVEMENT la mise en page originale.

Instructions STRICTES :
1.  **Conservation de la Mise en Page :** Reproduis EXACTEMENT la structure des paragraphes, les sauts de ligne, les retraits (indentations), les espaces et les lignes vides de l'original. Chaque élément doit être à sa place exacte.
2.  **Métadonnées Intactes :** Ne JAMAIS modifier les en-têtes, pieds de page, numéros de page, titres ou toute autre métadonnée. Reproduis-les tels quels.
3.  **Modernisation du Contenu :** Mets à jour UNIQUEMENT l'orthographe (ex: 'estoit' -> 'était'), la grammaire (accords, conjugaisons) et le vocabulaire obsolète du texte principal. Remplace les mots anciens par leurs équivalents modernes courants.
4.  **Pas de Réécriture :** Ne reformule pas les phrases, n'ajoute pas ou ne supprime pas d'informations. Modernise le texte existant, sans altérer le sens ou la structure phrastique fondamentale.
5.  **Format de Sortie :** Renvoie UNIQUEMENT le texte modernisé complet, en respectant toutes les règles de mise en page. N'ajoute aucune explication, commentaire ou balise supplémentaire.

Modernise le texte suivant en français contemporain standard, en conservant scrupuleusement sa mise en page originale (paragraphes, sauts de ligne, retraits, espacements, métadonnées). Applique les instructions systèmes à la lettre.

--- DEBUT DU TEXTE ORIGINAL ---
{input_chunk}
--- FIN DU TEXTE ORIGINAL ---

--- DEBUT DU TEXTE MODERNISÉ (ne rien écrire avant cette ligne) ---"""

        return prompt

    def process_chunk(self, chunk: str, chunk_index: int) -> str:
        """Process a single chunk of text."""
        max_retries = MAX_RETRIES
        retry_delay = 5  # seconds
        last_error = None

        for attempt in range(max_retries):
            try:
                # Estimate tokens for this chunk
                estimated_tokens = self.estimate_tokens(chunk)
                self.logger.info(
                    f"[チャンク {chunk_index}] 処理開始。入力トークン推定: {estimated_tokens}"
                )

                # Wait if needed based on rate limits
                self.token_rate_limiter.wait_if_needed(estimated_tokens)

                # Create the prompt
                prompt = self.create_prompt(chunk)

                # Make the API request
                start_time = time.time()
                response = self.model.generate_content(
                    prompt,
                    stream=False,
                    request_options={
                        "timeout": REQUEST_TIMEOUT,
                    },
                )
                end_time = time.time()

                # Check if we got a valid response
                if not response or not response.candidates:
                    error_msg = "API response missing 'candidates' field"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    error_msg = "API response missing 'content.parts' field"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

                output_text = candidate.content.parts[0].text.strip()

                # Check for empty response
                if not output_text:
                    error_msg = "APIから空のテキストが返されました。"
                    self.logger.warning(error_msg)
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                    raise Exception(error_msg)

                # Calculate token usage (Gemini doesn't provide detailed token info)
                # Use estimation for now
                estimated_output_tokens = self.estimate_tokens(output_text)
                total_tokens = estimated_tokens + estimated_output_tokens

                # Record token usage
                self.token_rate_limiter.record_usage(total_tokens)
                self.total_tokens_used += total_tokens

                # Log success
                self.logger.info(
                    f"[チャンク {chunk_index}] 成功 (試行 {attempt + 1}/{max_retries}, {end_time - start_time:.2f}秒). "
                    f"トークン推定: Input={estimated_tokens}, Output={estimated_output_tokens}, Total={total_tokens}. "
                    f"総トークン使用量: {self.total_tokens_used}"
                )

                # Check for significant line count changes
                original_lines = len(chunk.splitlines())
                new_lines = len(output_text.splitlines())
                if abs(original_lines - new_lines) > 5:  # Increased threshold
                    self.logger.warning(
                        f"[チャンク {chunk_index}] 行数が大幅に変化しました (元: {original_lines}, 新: {new_lines})。レイアウトが崩れている可能性があります。"
                    )

                return output_text

            except Exception as e:
                last_error = e
                self.logger.error(f"[チャンク {chunk_index}] エラー (試行 {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                raise

        # If we get here, all retries failed
        self.logger.error(f"[チャンク {chunk_index}] 最大リトライ回数に達しました。最後のエラー: {last_error}")
        raise last_error

    def process_text(self, input_text: str) -> str:
        """Process the complete text by chunking and using parallel processing."""
        if not input_text or input_text.isspace():
            self.logger.warning("入力テキストが空または空白のみです。")
            return ""

        try:
            chunks = self.chunk_text(input_text)
            if not chunks:
                self.logger.warning("有効なチャンクが生成されませんでした。")
                return ""

            modernized_chunks = [None] * len(chunks)  # Use None as placeholder
            failed_chunks = []  # 失敗したチャンクを記録するリスト

            # Use ThreadPoolExecutor for I/O-bound tasks (API calls)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:
                future_to_index = {
                    executor.submit(self.process_chunk, chunk, i): i
                    for i, chunk in enumerate(chunks)
                }

                # 改良された進捗表示
                with tqdm(
                    total=len(chunks),
                    desc="チャンク処理中",
                    unit="chunk",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [残り: {remaining}]",
                ) as pbar:
                    for future in concurrent.futures.as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            result = future.result()
                            modernized_chunks[index] = result
                            if not result:  # 空の結果は失敗とみなす
                                failed_chunks.append(index + 1)  # 1-indexed
                        except Exception as e:
                            # Error already logged in process_chunk
                            self.logger.error(
                                f"チャンク {index+1} の処理中にエラーが発生しました: {e}"
                            )
                            failed_chunks.append(index + 1)  # 1-indexed
                            modernized_chunks[index] = (
                                ""  # 失敗したチャンクは空文字列で埋める
                            )
                        finally:
                            pbar.update(1)

            # 失敗したチャンクの報告
            if failed_chunks:
                error_msg = f"以下のチャンクの処理に失敗しました: {', '.join(map(str, failed_chunks))}"
                self.logger.warning(error_msg)

                # 失敗数が一定以下なら続行
                if len(failed_chunks) <= len(chunks) * 0.1:  # 10%以下なら続行
                    self.logger.info(
                        f"失敗したチャンク数は全体の {len(failed_chunks)/len(chunks)*100:.1f}% で許容範囲内です。処理を続行します。"
                    )
                else:
                    self.logger.error(
                        f"失敗したチャンク数が多すぎます ({len(failed_chunks)}/{len(chunks)})。中断します。"
                    )
                    raise Exception(error_msg)

            # Join chunks. They should already have trailing newlines from chunk_text
            full_modernized_text = "".join(
                [chunk if chunk else "【処理失敗】\n\n" for chunk in modernized_chunks]
            ).rstrip()
            self.logger.info(
                f"すべてのチャンクの処理が完了しました。成功: {len(chunks) - len(failed_chunks)}/{len(chunks)}チャンク"
            )
            return full_modernized_text

        except Exception as e:
            self.logger.exception(f"テキスト全体の処理中にエラーが発生しました: {e}")
            raise  # Re-raise the exception to be caught in main()

    def save_output(self, output_text: str, base_filename: str) -> str:
        """Save the modernized text to a file."""
        # Use .md extension for markdown output
        output_filename = f"{os.path.splitext(base_filename)[0]}_modernized_gemini.md"
        full_path = os.path.join(self.output_path, output_filename)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(output_text)
            self.logger.info(f"出力ファイルを保存しました: {full_path}")
            return full_path
        except IOError as e:
            error_msg = (
                f"出力ファイル '{full_path}' の保存中にエラーが発生しました: {e}"
            )
            self.logger.error(error_msg)
            raise IOError(error_msg) from e

    def process_file(self, file_path: str) -> Optional[str]:
        """Read a file, process its content, and save the output."""
        self.logger.info(f"--- ファイル処理開始: {file_path} ---")
        try:
            # Read the input file
            with open(file_path, "r", encoding="utf-8") as f:
                input_text = f.read()

            if not input_text.strip():
                self.logger.warning(f"ファイル '{file_path}' は空または空白のみです。")
                return None

            # Process the text
            modernized_text = self.process_text(input_text)

            if not modernized_text.strip():
                self.logger.warning(f"ファイル '{file_path}' の処理結果が空です。")
                return None

            # Save the output
            base_filename = os.path.basename(file_path)
            output_file = self.save_output(modernized_text, base_filename)

            self.logger.info(f"--- ファイル処理完了: {file_path} -> {output_file} ---")
            return output_file

        except Exception as e:
            self.logger.exception(f"ファイル '{file_path}' の処理中にエラーが発生しました: {e}")
            raise


def select_files_dialog() -> List[str]:
    """Open a file dialog to select input files."""
    if not HAS_TKINTER:
        print("エラー: tkinterが利用できません。")
        return []

    try:
        root = tk.Tk()
        root.withdraw()
        root.title("入力ファイルを選択")
        file_paths = filedialog.askopenfilenames(
            title="処理するテキストファイルを選択してください",
            filetypes=[
                ("Text files", "*.txt"),
                ("Markdown files", "*.md"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return list(file_paths)
    except Exception as e:
        print(f"ファイル選択ダイアログでエラーが発生しました: {e}")
        return []


def select_output_dir_dialog(initial_dir: str) -> Optional[str]:
    """Open a directory dialog to select output directory."""
    if not HAS_TKINTER:
        print("エラー: tkinterが利用できません。")
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.title("出力ディレクトリを選択")
        selected_dir = filedialog.askdirectory(
            title="出力ディレクトリを選択してください",
            initialdir=initial_dir,
        )
        root.destroy()
        return selected_dir
    except Exception as e:
        print(f"ディレクトリ選択ダイアログでエラーが発生しました: {e}")
        return None


def main():
    """Main function to parse arguments and run the modernizer."""
    global MAX_WORKERS, REQUEST_TIMEOUT, TARGET_CHUNK_TOKENS  # 関数の先頭に移動

    print("--- フランス語テキスト近代化ツール (Gemini版) ---")

    parser = argparse.ArgumentParser(
        description="古いフランス語のテキストファイルをGemini APIを使用して現代化します。",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_files",
        nargs="+",  # Allows multiple files
        help="処理する入力テキストファイルへのパス。\n指定しない場合、ファイル選択ダイアログが表示されます (GUIが利用可能な場合)。",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help=f"近代化されたファイルを保存するディレクトリ。\n指定しない場合、デフォルトの '{DEFAULT_OUTPUT_DIR}' または\nディレクトリ選択ダイアログが表示されます (GUIが利用可能な場合)。",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=MAX_WORKERS,
        help=f"並列処理数（デフォルト: {MAX_WORKERS}）",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=REQUEST_TIMEOUT,
        help=f"API呼び出しのタイムアウト秒数（デフォルト: {REQUEST_TIMEOUT}）",
    )
    parser.add_argument(
        "--target_chunk_tokens",
        type=int,
        default=TARGET_CHUNK_TOKENS,
        help=f"チャンク分割の目標トークン数（デフォルト: {TARGET_CHUNK_TOKENS}）",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="一部チャンクの処理に失敗しても処理を継続する",
    )

    args = parser.parse_args()

    # Update global variables from command line arguments
    MAX_WORKERS = args.max_workers
    REQUEST_TIMEOUT = args.request_timeout
    TARGET_CHUNK_TOKENS = args.target_chunk_tokens

    # --- Determine Input Files ---
    input_files = args.input_files
    if not input_files:
        if HAS_TKINTER:
            print(
                "入力ファイルが指定されていません。ファイル選択ダイアログを開きます..."
            )
            input_files = select_files_dialog()
            if not input_files:
                print("ファイルが選択されませんでした。終了します。")
                return 1  # Exit code for no files selected
        else:
            print("エラー: 入力ファイルが指定されておらず、GUIも利用できません。")
            print("--input_files 引数でファイルを指定してください。")
            parser.print_help()
            return 1  # Exit code for missing input

    if not input_files:  # Double check after potential dialog cancellation
        print("処理するファイルがありません。終了します。")
        return 1

    # --- Determine Output Directory ---
    output_dir = args.output_dir
    if not output_dir:
        if HAS_TKINTER:
            print(
                f"出力ディレクトリが指定されていません。デフォルトは '{DEFAULT_OUTPUT_DIR}' です。"
            )
            print("変更する場合はディレクトリ選択ダイアログを開きます...")
            selected_dir = select_output_dir_dialog(
                os.path.join(os.getcwd(), DEFAULT_OUTPUT_DIR)
            )
            if selected_dir:  # If user selected a directory
                output_dir = selected_dir
            else:  # User cancelled or dialog failed
                print(
                    "ディレクトリ選択がキャンセルされたか失敗しました。デフォルトを使用します。"
                )
                output_dir = os.path.join(os.getcwd(), DEFAULT_OUTPUT_DIR)
        else:
            # No GUI, use default
            output_dir = os.path.join(os.getcwd(), DEFAULT_OUTPUT_DIR)
            print(
                f"出力ディレクトリが指定されておらず、GUIも利用できません。デフォルトの '{output_dir}' を使用します。"
            )

    # Create output directory if it doesn't exist (robustness)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(
            f"\033[91mエラー: 出力ディレクトリ '{output_dir}' の作成に失敗しました: {e}\033[0m"
        )
        return 1

    # --- Initialize Modernizer ---
    try:
        modernizer = TextModernizer(GOOGLE_API_KEY, output_path=output_dir)
    except ValueError as e:
        print(f"\033[91m初期化エラー: {e}\033[0m")
        return 1
    except Exception as e:
        print(f"\033[91m予期せぬ初期化エラーが発生しました: {e}\033[0m")
        logging.exception("初期化エラーの詳細:")  # Log traceback
        return 1

    # --- Process Files ---
    success_count = 0
    fail_count = 0
    start_time_all = time.time()

    print(f"\n{len(input_files)} 個のファイルを処理します...")
    for file_path in input_files:
        if not os.path.isfile(file_path):
            print(
                f"\033[93m警告: ファイルが見つからないか、ファイルではありません。スキップします: {file_path}\033[0m"
            )
            fail_count += 1
            continue

        file_start_time = time.time()
        try:
            output_file = modernizer.process_file(file_path)
            if output_file:
                success_count += 1
                file_duration = time.time() - file_start_time
                print(
                    f"\033[92m✓ 成功: {os.path.basename(file_path)} -> {os.path.basename(output_file)} ({file_duration:.2f}秒)\033[0m"
                )
            else:
                fail_count += 1
                file_duration = time.time() - file_start_time
                print(
                    f"\033[91m✗ 失敗 (空の結果): {os.path.basename(file_path)} ({file_duration:.2f}秒) - 詳細はログ '{LOG_FILE_NAME}' を確認してください。\033[0m"
                )

        except Exception as e:
            fail_count += 1
            file_duration = time.time() - file_start_time
            # Error details should already be logged by process_file or lower functions
            print(
                f"\033[91m✗ 失敗 (エラー): {os.path.basename(file_path)} ({file_duration:.2f}秒) - 詳細はログ '{LOG_FILE_NAME}' を確認してください。\033[0m"
            )
            # 全体の処理を停止せず続行 (ユーザーオプションに応じて)
            if not args.continue_on_error:
                print(
                    "\n\033[91m処理中にエラーが発生したため中断します。残りのファイルはスキップされます。\n--continue_on_error オプションを使用すると、エラーが発生しても処理を続行できます。\033[0m"
                )
                break

    total_duration = time.time() - start_time_all
    print("\n--- 処理結果 ---")
    print(f"成功: {success_count} ファイル")
    print(f"失敗: {fail_count} ファイル")
    print(f"総所要時間: {total_duration:.2f} 秒")
    modernizer.logger.info(
        f"総トークン使用量 (推定含む): {modernizer.total_tokens_used}"
    )
    print(
        f"詳細はログファイル '{os.path.join(output_dir, LOG_FILE_NAME)}' を確認してください。"
    )

    return 0 if fail_count == 0 else 1  # Return 0 if all successful, 1 otherwise


if __name__ == "__main__":
    # Ensure UTF-8 encoding for stdout/stderr, especially on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8")

    exit_code = main()
    sys.exit(exit_code)