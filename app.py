import io
import json
import os
import re
import traceback 
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
from google.cloud import vision
from google.cloud import storage 
from dotenv import load_dotenv

import vertexai # type: ignore
from vertexai.generative_models import GenerativeModel, Part, Tool, grounding, HarmCategory, HarmBlockThreshold, ToolConfig # type: ignore

# --- 初始化 Flask 應用 ---
app = Flask(__name__)
CORS(app) # 開發階段允許所有來源，生產環境應配置具體來源

# 加載環境變數
load_dotenv() 

# --- 配置 ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME")

DATASTORE_ID = os.environ.get("DATASTORE_ID")
DATASTORE_COLLECTION_LOCATION = "global"
DATASTORE_RESOURCE_NAME = f"projects/{GCP_PROJECT_ID}/locations/{DATASTORE_COLLECTION_LOCATION}/collections/default_collection/dataStores/{DATASTORE_ID}"

# GCS Bucket 用於存儲 Prompt 文件
GCS_PROMPT_BUCKET_NAME = os.environ.get("GCS_PROMPT_BUCKET_NAME")

# --- 初始化 Google Cloud 客戶端 ---
try:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client(project=GCP_PROJECT_ID)

    gemini_model = GenerativeModel(GEMINI_MODEL_NAME)
    search_tool = Tool.from_retrieval(
        grounding.Retrieval(grounding.VertexAISearch(datastore=DATASTORE_RESOURCE_NAME))
    )
    tools_list = [search_tool]
    print("Vertex AI and Google Cloud clients initialized successfully.")

except Exception as e:
    print(f"Error initializing Google Cloud clients: {e}")
    traceback.print_exc()

# --- 輔助函數 ---
def get_size(obj):
    """遞歸計算物件在記憶體中的大小（以MB為單位）。"""
    seen_ids = set()
    
    def sizeof_detail(o):
        if id(o) in seen_ids:
            return 0
        seen_ids.add(id(o))
        size = sys.getsizeof(o)
        if isinstance(o, dict):
            size += sum(sizeof_detail(v) for v in o.values())
            size += sum(sizeof_detail(k) for k in o.keys())
        elif hasattr(o, '__dict__'):
            size += sizeof_detail(o.__dict__)
        elif hasattr(o, '__iter__') and not isinstance(o, (str, bytes, bytearray)):
            size += sum(sizeof_detail(i) for i in o)
        return size

    return sizeof_detail(obj) / (1024 * 1024)

def process_and_compress_image(file_bytes, max_size_mb=20, max_dimension=1200, quality=85):
    """
    檢查、壓縮並調整圖片大小。
    返回壓縮後的圖片二進制數據，如果檔案過大或非圖片則返回 None。
    """
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        print(f"ERROR: File is too large ({file_size_mb:.2f}MB), limit is {max_size_mb}MB.")
        return None
        
    try:
        img = Image.open(io.BytesIO(file_bytes))
        
        # 轉換為 RGB 以避免處理 RGBA 或 P 模式時的儲存問題
        if img.mode not in ('RGB', 'L'): # L for grayscale
            img = img.convert('RGB')
            
        img.thumbnail((max_dimension, max_dimension))
        
        output_buffer = io.BytesIO()
        img.format = 'JPEG' # 強制存為 JPEG 以確保壓縮
        img.save(output_buffer, format='JPEG', quality=quality)
        compressed_bytes = output_buffer.getvalue()
        
        compressed_size_mb = len(compressed_bytes) / (1024 * 1024)
        print(f"Image compressed from {file_size_mb:.2f}MB to {compressed_size_mb:.2f}MB.")
        return compressed_bytes
    except Exception as e:
        print(f"ERROR: Could not process or compress image. It might not be a valid image file. Error: {e}")
        return None
    
def perform_ocr(image_file_storage): # <--- 恢復 perform_ocr 函數
    """使用 Google Cloud Vision API 對圖片文件執行 OCR。"""
    if not image_file_storage:
        return "OCR_ERROR: No image file provided."
    try:
        print(f"Performing OCR on image: {image_file_storage.filename}")
        content = image_file_storage.read()
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        return response.text_annotations[0].description if response.text_annotations else ""
    except Exception as e:
        print(f"Error during OCR: {e}")
        traceback.print_exc()
        return f"OCR_ERROR: {str(e)}"


def get_prompt_from_gcs(bucket_name, file_path_in_bucket):
    """從 GCS 讀取 Prompt 文本。"""
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path_in_bucket)
        print(f"bucket_name:{bucket_name},file_path_in_bucket:{file_path_in_bucket}")
        if not blob.exists():
            print(f"Error: Prompt file gs://{bucket_name}/{file_path_in_bucket} not found.")
            return None
        prompt_text = blob.download_as_text()
        print(f"Successfully loaded prompt from gs://{bucket_name}/{file_path_in_bucket}")
        return prompt_text
    except Exception as e:
        print(f"Error fetching prompt from GCS (gs://{bucket_name}/{file_path_in_bucket}): {e}")
        traceback.print_exc()
        return None
    
def get_standard_answer_for_lesson_from_gcs(bucket_name, base_file_path, grade_level, learnsheets_key, worksheet_category):
    """
    從 GCS 讀取指定年級和學習單主題的完整標準答案 JSON，
    並提取該主題 (Lesson X) 的數據。
    
    Args:
        bucket_name (str): GCS 存儲桶名稱。
        base_file_path (str): 基礎檔案路徑，例如 'ai_english_tutor/ai_english_file/'。
        grade_level (str): 年級，例如 '七年級'。
        learnsheets_key (str): 學習單主題，例如 'Lesson 1'。
        
    Returns:
        dict: 該 Lesson 的標準答案數據字典，如果找不到則返回 None。
    """
    answer_file_map = {
        "七年級全英提問學習單參考答案":"全英提問學習單參考答案(01_1下).txt",
        "八年級全英提問學習單參考答案":"全英提問學習單參考答案(01_2下).txt",
        "九年級全英提問學習單參考答案":"全英提問學習單參考答案(01_3下).txt",
        "七年級差異化學習單參考答案":"差異化學習單參考答案(01_1下).txt",
        "八年級差異化學習單參考答案":"差異化學習單參考答案(01_2下).txt",
        "九年級差異化學習單參考答案":"差異化學習單參考答案(01_3下).txt",
    }

    # 根據 grade_level 和 worksheet_category 找到對應的檔案
    full_file_key = f"{grade_level}{worksheet_category}"
    target_filename = answer_file_map.get(full_file_key)
    
    if not target_filename:
        print(f"Error: No standard answer file found for grade '{grade_level}' and category '{worksheet_category}'.")
        return None
    
    try:
        bucket = storage_client.bucket(bucket_name)
        base_file_path = f"{base_file_path}{target_filename}"
        blob = bucket.blob(base_file_path)
        print(f"base_file_path:{base_file_path},target_filename:{target_filename},bucket_name:{bucket_name},blob:{blob}")
        if not blob.exists():
            print(f"Error: Standard answer file gs://{bucket_name}{base_file_path} not found.")
            return None
        
        json_content = blob.download_as_text()
        all_answers_data = json.loads(json_content)
        
        # 提取指定 learnsheets_key (例如 "Lesson 1") 的數據
        lesson_data = all_answers_data.get(learnsheets_key)
        
        if not lesson_data:
            print(f"Warning: '{learnsheets_key}' not found in standard answer file {target_filename}.")
            return None
            
        print(f"Successfully loaded standard answers for {learnsheets_key} from gs://{bucket_name}/{target_filename}")
        return lesson_data
    
    except Exception as e:
        print(f"Error fetching standard answers from GCS (gs://{bucket_name}/{target_filename}): {e}")
        traceback.print_exc()
        return None
    
def get_standard_answer_for_reading_writing_from_gcs(bucket_name, base_file_path, grade_level, bookrange):
    """
    從 GCS 讀取指定年級和冊次的讀寫習作標準答案 JSON，
    並提取該冊次 (例如 Book 5) 的數據。
    
    Args:
        bucket_name (str): GCS 存儲桶名稱。
        base_file_path (str): 基礎檔案路徑，例如 'ai_english_tutor/ai_english_file/'。
        grade_level (str): 年級，例如 '七年級'。
        bookrange (str): 冊次主題，例如 'Book 5'。
        
    Returns:
        dict: 該冊次的標準答案數據字典，如果找不到則返回 None。
    """
    # 假設讀寫習作的答案檔名格式
    answer_file_map = {
        "七年級讀寫習作參考答案": "113_1習作標準答案.txt", 
        "八年級讀寫習作參考答案": "113_2習作標準答案.txt", 
        "九年級讀寫習作參考答案": "113_3習作標準答案.txt", 
    }

    # 根據 grade_level 找到對應的檔案
    full_file_key = f"{grade_level}讀寫習作參考答案"
    target_filename = answer_file_map.get(full_file_key)
    
    if not target_filename:
        print(f"Error: No standard answer file found for reading/writing grade '{grade_level}'.")
        return None
    
    try:
        bucket = storage_client.bucket(bucket_name)
        full_file_path = f"{base_file_path}{target_filename}"
        blob = bucket.blob(full_file_path)
        print(f"Attempting to load reading/writing answers from: gs://{bucket_name}/{full_file_path}")

        if not blob.exists():
            print(f"Error: Standard answer file gs://{bucket_name}/{full_file_path} not found.")
            return None
        
        json_content = blob.download_as_text()
        all_answers_data = json.loads(json_content)
        
        # 提取指定 bookrange_key (例如 "Book 5") 的數據
        book_data = all_answers_data.get(bookrange)
        
        if not book_data:
            print(f"Warning: '{bookrange}' not found in standard answer file {target_filename}.")
            return None
            
        print(f"Successfully loaded standard answers for {bookrange} from gs://{bucket_name}/{target_filename}")
        return book_data
    
    except Exception as e:
        print(f"Error fetching reading/writing standard answers from GCS (gs://{bucket_name}/{target_filename}): {e}")
        traceback.print_exc()
        return None

def get_json_format_example(submission_type, mock_paragraph, mock_quiz, mock_learning_sheet, mock_reading_writing):
    """根據提交類型返回對應的 JSON 格式範例字符串。"""
    # 這裡我們仍然使用 mock data 來定義 JSON 結構範例
    # 您也可以將這些 JSON 結構範例存儲為 GCS 文件或其他配置
    if submission_type == '測驗寫作評改':
        return json.dumps(mock_quiz, ensure_ascii=False, indent=2)
    elif submission_type == '段落寫作評閱':
        return json.dumps(mock_paragraph, ensure_ascii=False, indent=2)
    elif submission_type == '學習單批改':
        return json.dumps(mock_learning_sheet, ensure_ascii=False, indent=2)
    elif submission_type == '讀寫習作評分':
        return json.dumps(mock_reading_writing, ensure_ascii=False, indent=2)
    else:
        print(f"Warning: Unknown submission_type '{submission_type}' for JSON example. Defaulting to paragraph.")
        return json.dumps(mock_paragraph, ensure_ascii=False, indent=2) # 默認

# --- 主要路由 ---
@app.route('/api/grade', methods=['POST'])
def grade_writing():
    print(f"\n--- New request to /api/grade ---")
    print(f"Request Content-Type: {request.content_type}")

    try:
        submission_type = None
        grade_level = None
        text_input = None
        bookrange = None
        learnsheets = None
        worksheet_category = None
        essay_image_files = []
        learning_sheet_files = []
        reading_writing_files = []
        standard_answer_text = None
        standard_answer_image_files = []
        scoring_instructions = None

        if request.content_type and 'multipart/form-data' in request.content_type.lower():
            print("Processing as multipart/form-data")
            submission_type = request.form.get('submissionType')
            grade_level = request.form.get('gradeLevel')
            text_input = request.form.get('text')
            if submission_type == '讀寫習作評分':
                bookrange = request.form.get('bookrange')
                print(f"Extracted bookrange from form: {bookrange}")
            if submission_type == '學習單批改':
                learnsheets = request.form.get('learnsheets')
                worksheet_category = request.form.get('worksheetCategory')
                print(f"Extracted learnsheets from form: {learnsheets},Category: {worksheet_category}")
            essay_image_files = request.files.getlist('essayImage')
            learning_sheet_files = request.files.getlist('learningSheetFile')
            reading_writing_files = request.files.getlist('readingWritingFile')
            if submission_type == '測驗寫作評改':
                standard_answer_text = request.form.get('standardAnswerText', '')
                standard_answer_image_files = request.files.getlist('standardAnswerImage') # get 返回 None 如果不存在
                scoring_instructions = request.form.get('scoringInstructions', '')
        elif request.is_json:
            print("Processing as JSON")
            data = request.get_json()
            submission_type = data.get('submissionType')
            grade_level = data.get('gradeLevel')
            bookrange = data.get('bookrange')
            learnsheets = data.get('learnsheets')
            worksheet_category = data.get('worksheetCategory')
            text_input = data.get('text') 
            if submission_type == '測驗寫作評改':
                standard_answer_text = data.get('standardAnswerText', '')
                scoring_instructions = data.get('scoringInstructions', '')
        else:
            print(f"Warning: Received request with unhandled Content-Type: {request.content_type}")
            submission_type = request.form.get('submissionType')
            grade_level = request.form.get('gradeLevel')


        print(f"Parsed Submission Type: {submission_type}, Grade Level: {grade_level}")

        # --- 後續的驗證和處理邏輯 ---
        if not submission_type or not grade_level: # 現在這個判斷更關鍵
            return jsonify({"error": "Missing submissionType or gradeLevel after parsing request"}), 400

        # --- 準備傳遞給 Gemini 的內容列表 ---
        # Gemini 的 generate_content 可以接受一個 Part 物件列表 (文本和圖片)
        contents_for_gemini = []

        essay_content = "" # 儲存給 Prompt 字符串的內容 (可以是文本或指示性文本)
        
        # 處理學生作業內容：優先文本輸入，其次圖片文件
        if submission_type == '段落寫作評閱' or submission_type == '測驗寫作評改':
            if text_input:
                essay_content = text_input
            elif essay_image_files:
                # 使用 Vision API 執行 OCR，結果用於填入 Prompt 中的 {essay_content}
                ocr_results = []
                contents_for_gemini.append(Part.from_text("學生提交的原始作業圖片內容，供您參考理解其版面和手寫內容："))
                for file_storage in essay_image_files:
                    file_storage.seek(0) # 重置文件指針，確保 OCR 和圖片 Part 都能讀取到完整內容
                    ocr_text = perform_ocr(file_storage)
                    if "OCR_ERROR:" in ocr_text:
                        print(f"OCR failed for {file_storage.filename}: {ocr_text}. Skipping this file for OCR content.")
                        # 即使 OCR 失敗，我們仍嘗試將原始圖片作為 Part 傳遞
                        # 這裡可以選擇報錯或繼續
                    elif ocr_text.strip():
                        ocr_results.append(ocr_text)
                    
                    # 將原始圖片添加到 Gemini 的 Part 列表中
                    file_storage.seek(0) # 再次重置指針，確保讀取完整的圖片數據
                    try:
                        image_data = file_storage.read()
                        contents_for_gemini.append(Part.from_data(data=image_data, mime_type=file_storage.content_type))
                        print(f"Added student essay image {file_storage.filename} as Part.")
                    except Exception as e:
                        print(f"Error reading student essay image {file_storage.filename} for Gemini: {e}")
                        traceback.print_exc()
                        return jsonify({"error": f"Failed to read student essay image file {file_storage.filename} for Gemini: {str(e)}"}), 400

                if not ocr_results and not text_input: # 如果所有 OCR 都失敗或沒有純文本
                    return jsonify({"error": "OCR failed or returned empty for all provided images, and no text input."}), 400
                
                essay_content = "\n\n".join(ocr_results) if ocr_results else "（圖片內容 OCR 失敗或為空，請參考隨後提供的原始圖片）" # 即使 OCR 失敗，也給個提示

            else:
                return jsonify({"error": f"For {submission_type}, text input or an essay image is required."}), 400
        elif submission_type == '學習單批改':
            if learning_sheet_files:
                ocr_results = []
                contents_for_gemini.append(Part.from_text("學生提交的原始學習單圖片內容，供您參考理解其版面和手寫內容："))
                for file_storage in learning_sheet_files:
                    file_storage.seek(0)
                    ocr_text = perform_ocr(file_storage)
                    if "OCR_ERROR:" in ocr_text:
                        print(f"OCR failed for {file_storage.filename}: {ocr_text}. Skipping this file for OCR content.")
                    elif ocr_text.strip():
                        ocr_results.append(ocr_text)

                    file_storage.seek(0)
                    try:
                        image_data = file_storage.read()
                        contents_for_gemini.append(Part.from_data(data=image_data, mime_type=file_storage.content_type))
                        print(f"Added learning sheet image {file_storage.filename} as Part.")
                    except Exception as e:
                        print(f"Error reading learning sheet image {file_storage.filename} for Gemini: {e}")
                        traceback.print_exc()
                        return jsonify({"error": f"Failed to read learning sheet file {file_storage.filename} for Gemini: {str(e)}"}), 400

                if not ocr_results:
                    return jsonify({"error": "OCR failed or returned empty for all provided learning sheet images."}), 400
                essay_content = "\n\n".join(ocr_results)
            else:
                return jsonify({"error": "Learning sheet file is required for '學習單批改'."}), 400
        elif submission_type == '讀寫習作評分':
            if reading_writing_files:
                ocr_results = []
                contents_for_gemini.append(Part.from_text("學生提交的原始讀寫習作圖片內容，供您參考理解其版面和手寫內容："))
                for file_storage in reading_writing_files:
                    file_storage.seek(0)
                    ocr_text = perform_ocr(file_storage)
                    if "OCR_ERROR:" in ocr_text:
                        print(f"OCR failed for {file_storage.filename}: {ocr_text}. Skipping this file for OCR content.")
                    elif ocr_text.strip():
                        ocr_results.append(ocr_text)

                    file_storage.seek(0)
                    try:
                        image_data = file_storage.read()
                        contents_for_gemini.append(Part.from_data(data=image_data, mime_type=file_storage.content_type))
                        print(f"Added reading/writing worksheet image {file_storage.filename} as Part.")
                    except Exception as e:
                        print(f"Error reading reading/writing worksheet image {file_storage.filename} for Gemini: {e}")
                        traceback.print_exc()
                        return jsonify({"error": f"Failed to read reading/writing worksheet file {file_storage.filename} for Gemini: {str(e)}"}), 400

                if not ocr_results:
                    return jsonify({"error": "OCR failed or returned empty for all provided reading/writing worksheet images."}), 400
                essay_content = "\n\n".join(ocr_results)
            else:
                return jsonify({"error": "Reading/writing worksheet file is required for '讀寫習作評分'."}), 400
        else:
            return jsonify({"error": f"Unsupported submission type for content processing: {submission_type}"}), 400

        if not essay_content.strip() and not text_input and not contents_for_gemini: # 確保有任何形式的內容
            return jsonify({"error": "Essay content is empty after processing input"}), 400
        print(f"Essay content (first 100 chars): {essay_content[:100]}...")
        
        # 處理標準答案（僅測驗寫作評改）
        processed_standard_answer = ""
        if submission_type == '測驗寫作評改':
            if standard_answer_text:
                processed_standard_answer = standard_answer_text
            elif standard_answer_image_files:
                ocr_standard_answer_results = []
                contents_for_gemini.append(Part.from_text("\n測驗的原始標準答案圖片內容，供您參考理解其版面和手寫內容："))
                for file_storage in standard_answer_image_files:
                    file_storage.seek(0) # 重置指針
                    ocr_text = perform_ocr(file_storage)
                    if "OCR_ERROR:" in ocr_text:
                        print(f"OCR for standard answer image {file_storage.filename} failed: {ocr_text}. Skipping OCR content for this file.")
                    elif ocr_text.strip():
                        ocr_standard_answer_results.append(ocr_text)
                    
                    file_storage.seek(0) # 再次重置指針
                    try:
                        image_data = file_storage.read()
                        contents_for_gemini.append(Part.from_data(data=image_data, mime_type=file_storage.content_type))
                        print(f"Added standard answer image {file_storage.filename} as Part.")
                    except Exception as e:
                        print(f"Error reading standard answer image {file_storage.filename} for Gemini: {e}")
                        traceback.print_exc()
                        print(f"Failed to read standard answer image file {file_storage.filename} for Gemini: {str(e)}")

                if ocr_standard_answer_results:
                    processed_standard_answer = "\n\n".join(ocr_standard_answer_results)
                else:
                    processed_standard_answer = "（標準答案圖片內容 OCR 失敗或為空，請參考隨後提供的原始圖片）"
                    print("OCR for all standard answer images failed or returned empty.")
            print(f"Processed Standard Answer (first 100 chars): {processed_standard_answer[:100]}...")
        # --- 新增：載入學習單標準答案的邏輯 ---
        standard_answers_json_str = "" 

        # 邏輯 1：處理「學習單批改」的標準答案
        if submission_type == '學習單批改' and learnsheets and worksheet_category:
            print(f"Attempting to load standard answers for {grade_level} {worksheet_category} {learnsheets}")
            standard_answers_data = get_standard_answer_for_lesson_from_gcs(
                GCS_PROMPT_BUCKET_NAME,
                "ai_english_file/", # 你的 GCS 檔案路徑
                grade_level,
                learnsheets,
                worksheet_category
            )
            if standard_answers_data:
                standard_answers_json_str = json.dumps(standard_answers_data, ensure_ascii=False, indent=2)
                print(f"Loaded specific '學習單' standard answers (first 200 chars): {standard_answers_json_str[:200]}...")
            else:
                print("Failed to load specific '學習單' standard answers. Gemini will rely on search_tool for all answers.")
        
        # 邏輯 2：【新增】處理「讀寫習作評分」的標準答案
        elif submission_type == '讀寫習作評分' and bookrange:
            print(f"Attempting to load standard answers for {grade_level} {bookrange}")
            standard_answers_data = get_standard_answer_for_reading_writing_from_gcs(
                GCS_PROMPT_BUCKET_NAME,
                "ai_english_file/", # 你的 GCS 檔案路徑
                grade_level,
                bookrange # 使用 bookrange 作為 key
            )
            if standard_answers_data:
                standard_answers_json_str = json.dumps(standard_answers_data, ensure_ascii=False, indent=2)
                print(f"Loaded specific '讀寫習作' standard answers (first 200 chars): {standard_answers_json_str[:200]}...")
            else:
                print("Failed to load specific '讀寫習作' standard answers. Gemini will rely on search_tool for all answers.")


        # --- 1. 根據 submissionType 確定要加載的 Prompt 文件名 ---
        prompt_file_map = {
            "段落寫作評閱": "段落寫作評閱.txt",
            "測驗寫作評改": "測驗寫作評改.txt",
            "學習單批改": "學習單批改.txt",
            "讀寫習作評分": "讀寫習作評分.txt"
        }
        prompt_file_name = prompt_file_map.get(submission_type)
        if not prompt_file_name:
            return jsonify({"error": f"Unsupported submission type: {submission_type}"}), 400

        # --- 2. 從 GCS 加載對應的 Prompt 模板 ---
        prompt_folder_path = "ai_english_prompt/"
        full_prompt_path_in_bucket = f"{prompt_folder_path}{prompt_file_name}"

        base_prompt_text = get_prompt_from_gcs(GCS_PROMPT_BUCKET_NAME, full_prompt_path_in_bucket)
        if not base_prompt_text:
            return jsonify({"error": f"Failed to load prompt template for {submission_type} from gs://{GCS_PROMPT_BUCKET_NAME}/{full_prompt_path_in_bucket}"}), 500

        # --- 3. 準備 JSON 格式範例 (從 mock data 生成) ---
        # 為了演示，我們還是需要 mock data 來定義 JSON 結構
        # 您之後可以根據需要調整這些 mock data
        mock_paragraph_data_for_structure = {
        "submissionType": "段落寫作評閱",
        "error_analysis": [
            {
            "original_sentence": "With my heart beating rapidly in excitement, I tried to look past the sea olf people and see through the large glass windows of the department store.",
            "error_type": "拼寫錯誤",
            "error_content": "oIf 應為 of，departiment 應為 department",
            "suggestion": "With my heart beating rapidly in excitement, I tried to look past the sea of people and see through the large glass windows of the department store. (olf 應為 of, department store 通常是一個詞組，但此處 department 單獨出現可能指部門，如果指百貨公司則應為 department store)"
            },
            {
            "original_sentence": "Every person waiting outside had the same goal as mine to take advantage of the huge sales the shop was offering.",
            "error_type": "文法錯誤 (比較結構)",
            "error_content": "mine 後面應加上 is 或 was，以完成比較。",
            "suggestion": "Every person waiting outside had the same goal as mine: to take advantage of the huge sales the shop was offering. (在 mine 後面加上冒號或 is/was 來完成比較結構會更清晰)"
            },
            {
            "original_sentence": "However, many other customers had beaten me to the task.",
            "error_type": "用字遣詞 (表達不自然)",
            "error_content": "beaten me to the task 略顯不自然，可替換為更常見的表達方式。",
            "suggestion": "However, many other customers had arrived earlier / gotten there before me. ('beaten me to the task' 略顯不自然，可替換為更常見的表達方式)"
            },
            {
            "original_sentence": "Therefore, I stood slightly farther away from the enterance than I had planned,but that did not put out my ambition to purchase as many items as possible.",
            "error_type": "拼寫錯誤，標點符號",
            "error_content": "enterance 應為 entrance，but 前面的逗號應改為分號或句號",
            "suggestion": "Therefore, I stood slightly farther away from the entrance than I had planned; but that did not diminish my ambition to purchase as many items as possible."
            },
            {
            "original_sentence": "The constant chatter around me became impatient as time trickled by.",
            "error_type": "用字遣詞",
            "error_content": "chatter 本身不會感到 impatient，應是人感到 impatient。",
            "suggestion": "I became impatient with the constant chatter around me as time trickled by."
            }
        ],
        "rubric_evaluation": {
            "structure_performance": [
            {
                "item": "Task Fulfillment and Purpose",
                "score": 8,
                "comment": "很好地完成了任務，描述了一次購物的經歷，並表達了情感的轉變。主題明確。"
            },
            {
                "item": "Topic Sentence and Main Idea",
                "score": 7,
                "comment": "段落中有多個主題句，但主旨明確，圍繞著購物經歷和情感轉變展開。"
            },
            {
                "item": "Supporting Sentences and Argument Development",
                "score": 7,
                "comment": "細節描述豐富，但部分細節可以更精煉，使論述更集中。"
            },
            {
                "item": "Cohesion and Coherence",
                "score": 7,
                "comment": "整體連貫性不錯，但部分句子之間的銜接可以更自然。"
            },
            {
                "item": "Concluding Sentence and Closure",
                "score": 8,
                "comment": "結尾總結了整件事情，並點明了主題，有很好的收尾。"
            }
            ],
            "content_language": [
            {
                "item": "Depth of Analysis and Critical Thinking",
                "score": 7,
                "comment": "對情感的轉變有一定程度的分析，但可以更深入地挖掘內心感受。"
            },
            {
                "item": "Grammar and Sentence Structure",
                "score": 6,
                "comment": "文法基礎尚可，但存在一些錯誤，需要加強練習。"
            },
            {
                "item": "Vocabulary and Word Choice",
                "score": 7,
                "comment": "詞彙使用恰當，但可以嘗試使用更多樣化的詞彙。"
            },
            {
                "item": "Spelling, Punctuation, and Mechanics",
                "score": 6,
                "comment": "拼寫和標點符號方面存在一些錯誤，需要仔細檢查。"
            },
            {
                "item": "Persuasive Effectiveness and Audience Awareness",
                "score": 7,
                "comment": "故事具有一定的感染力，能引起讀者的共鳴。"
            }
            ]
        },
        "overall_assessment": {
            "total_score": "68/100",
            "suggested_grade": "C+",
            "grade_basis": "依據七年級標準評量。",
            "general_comment": "整體而言，作文內容生動有趣，但文法和拼寫方面仍需加強。繼續努力，注意細節，相信你會寫得更好！"
        },
        "model_paragraph": "With my heart beating rapidly in excitement, I tried to look past the sea of people and see through the large glass windows of the department store. Every person waiting outside had the same goal as mine: to take advantage of the huge sales the shop was offering. I had arrived early in the morning, hoping to be close to the doors. However, many other customers had arrived even earlier. Therefore, I stood slightly farther away from the entrance than I had planned, but that did not diminish my ambition to purchase as many items as possible. I became impatient with the constant chatter around me as time trickled by. Suddenly, the glass doors burst open. I watched as men and women in front of me flooded into the store. All around me, people pushed each other, eager to get in. We were like sardines in a box as we crammed through the narrow doors. Being too preoccupied to notice my surroundings, I tripped on the edge of the carpet. To my disappointment, I found myself sprawled on the floor, watching as people grabbed goods off the shelves. My ankle was sprained, and it was as though all my waiting had gone to waste. Even worse, no one even stopped to help me up. Limping around the store, I realized I couldn't get to the discounted items fast enough. Although I had arrived earlier than most, my carelessness had resulted in a disadvantage. I saw a number of products snatched up by quicker hands, and people watched as other people filled up carts and baskets. Consequently, my former excitement faded away, replaced by regret. How I wish I had not come! Having given up hope, I slowly made my way to the exit, my hands empty and my wallet full. Stepping out the glass doors, I noticed in the corner of my eye several people holding out signs. Curious, I went to check it out. It was a charity for stray dogs and they had brought puppies with them. I couldn't resist the urge to caress the canines' heads. Wagging their tails enthusiastically, they licked my palms. I giggled, all my disappointment dissolved like salt in water. After playing with them for a while, I pulled out my purse and donated all the money I had planned to spend. At the end of the day, I did go home with my purse empty. However, instead of products, I had a cute puppy in my hands. What a wonderful day it had been.",
        "teacher_summary_feedback": "你的作文內容很有趣，描述了一次難忘的購物經歷。故事的敘述流暢，情感表達也比較自然。不過，在文法和拼寫方面還有進步的空間。多加練習，注意細節，相信你會寫得更好！"
        }
        mock_quiz_data_for_structure = {
        "submissionType": "測驗寫作評改",
        "error_analysis_table": [
            {
            "original_sentence": "It was the anniversary of the mall where a mutitude of discounts took place.",
            "error_type": "拼寫錯誤 / 用字選擇",
            "problem_description": "單字 'mutitude' 拼寫錯誤，應為 'multitude'。同時，'took place' 用於描述折扣的發生略顯生硬，'were offered' 更自然。",
            "suggestion": "It was the anniversary of the mall where a multitude of discounts were offered."
            },
            {
            "original_sentence": "Some people waited patiently in line and killed their time by being phubbers, whereas others couldn't stand the taxing process of waiting in line and gave up.",
            "error_type": "用字選擇 (非正式/俚語)",
            "problem_description": "'Phubbers' 是較新的非正式詞彙，不一定所有讀者都理解，建議在測驗寫作中使用更通俗、正式的表達，例如 'using their phones' 或 'distracted by their phones'。'taxing process' 表達準確。",
            "suggestion": "Some people waited patiently in line and killed their time by using their phones, whereas others couldn't stand the taxing process of waiting in line and gave up."
            },
            {
            "original_sentence": "To make matters worse, some impatient customers even lost their temper and tried to cut in lines, causing disputes and leaving the mall in chaos.",
            "error_type": "固定用法",
            "problem_description": "應為'cut in line'。",
            "suggestion": "To make matters worse, some impatient customers even lost their temper and tried to cut in line, causing disputes and leaving the mall in chaos."
            },
            {
            "original_sentence": "Others either surrendered or got cut off after the mall closed.",
            "error_type": "用詞選擇",
            "problem_description": "'surrendered'在此情境下稍正式，可用'gave up'。",
            "suggestion": "Others either gave up or got cut off after the mall closed."
            }
        ],
        "summary_feedback_for_student": 
        {
            "summary_feedback":"你的作文整體結構完整，敘事流暢，能夠生動地描寫場景和人物心理。詞彙使用豐富，展現了不錯的英文基礎。不過，在拼寫和用詞的準確性上還有進步空間。注意檢查拼寫錯誤，並選擇更貼切、自然的詞彙，可以讓你的作文更上一層樓。",
            "total_score_display": "92 / 100",
            "suggested_grade_display": "A-",
            "grade_basis_display": "根據國中三年級寫作標準"
        },
        "revised_demonstration": 
        {
            "original_with_errors_highlighted": "<strong>Anxiously</strong> waiting, legions of people stood in front of the gate. It was the anniversary of the mall where a <strong>mutitude</strong> of discounts took place. When it was about eight AM, a <strong>staff</strong> of the mall approached the door. No sooner did he open the door than the crowd dashed in. They entered every store, took what they wanted to <strong>bry</strong>, and <strong>literally</strong> went on a shopping spree. Thousands of purchases were made and everyone thought that they could shop to their hearts' <strong>contert</strong>. However, the story unfolded in the opposite way. As more and more people got into the mall, not only were the stores packed, but the line waiting in front of the cashier stretched for more than ten meters. The smiles on people's face and their <strong>electricfied</strong> mood <strong>withered</strong> as time went by. The time spent on <strong>shoppirg</strong> was actually less than the time spent on waiting. Given this frustrating condition, the crowd had different reactions. Some people waited patiently in line and killed their time by being <strong>phubbers</strong>, whereas others couldn't stand the <strong>taxing</strong> process of waiting in line and gave up. To make matters worse, some impatient customers even lost their temper and tried to cut in lines, causing disputes and leaving the mall in chaos. At the end of the day, only one third of the customers made it to pay for what they had taken. Others either <strong>surrendered</strong> or got cut off after the mall closed. The next day, this incident was reported by the news, and people started to reflect. Eventually, most of them reached the same conclusion that we should no longer blindly follow the crowd and get fooled by the marketing strategies of the mall. After all, no one wants to wait in line for hours and wind up wasting their time.",
            "suggested_revision": "Anxiously waiting, legions of people stood in front of the gate. It was the anniversary of the mall where a large number of discounts were offered. When it was about eight AM, an employee of the mall approached the door. No sooner did he open the door than the crowd dashed in. They entered every store, took what they wanted to buy, and went on a shopping spree. Thousands of purchases were made and everyone thought that they could shop to their hearts' content. However, the story unfolded in the opposite way. As more and more people got into the mall, not only were the stores packed, but the line waiting in front of the cashier stretched for more than ten meters. The smiles on people's faces and their electrified mood faded as time went by. The time spent on shopping was actually less than the time spent on waiting. Given this frustrating condition, the crowd had different reactions. Some people waited patiently in line and killed their time by using their phones, whereas others couldn't stand the difficult process of waiting in line and gave up. To make matters worse, some impatient customers even lost their temper and tried to cut in lines, causing disputes and leaving the mall in chaos. At the end of the day, only one third of the customers made it to pay for what they had taken. Others either gave up or got cut off after the mall closed. The next day, this incident was reported by the news, and people started to reflect. Eventually, most of them reached the same conclusion that we should no longer blindly follow the crowd and get fooled by the marketing strategies of the mall. After all, no one wants to wait in line for hours and wind up wasting their time."
        },
        "positive_learning_feedback": "你的寫作展現了很強的敘事能力和豐富的詞彙量，能夠清楚地表達想法，讓讀者感受到你的思考與情感。即使過程中出現了一些小錯誤，也完全不影響整體的表現。請不要因此氣餒，因為每一次寫作的練習，都是一次難能可貴的學習與成長機會。透過不斷地修正與嘗試，你會更了解自己的風格，也會漸漸掌握如何讓語言更具感染力。繼續保持你對寫作的熱情與好奇心，相信你會在這條路上越走越穩，越寫越好，未來也有機會創作出更多令人印象深刻的作品！"
        }
        mock_learning_sheet_structure = {
        "submissionType": "學習單批改",
        "title": "📋 學習單批改結果",
        "sections": [
            {
            "section_title": "[考卷上的大標題(粗體)]",
            "questions_feedback": [
                {
                "question_number": "1",
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]",
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[標準答案]中對應題號的正確答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Lesson 1/Pre-listening Questions/1:Yes, there are two sports teams in my school. They are the soccer team and the basketball team.)]"
                },
                {
                "question_number": "2",
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]",
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[標準答案]中對應題號的正確答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Lesson 1/Pre-listening Questions/2:Yes, I play sports in my free time.)]"
                }
            ],
            "section_summary": "[根據學生在此部分的表現生成總結]"
            },
            {
            "section_title": "[考卷上的大標題(粗體)]",
            "questions_feedback": [
                {
                "question_number": "1",
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]",
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[標準答案]中對應題號的正確答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Lesson 1/While-listening Notes/1:Do you practice basketball after school every day)]"
                }
            ],
            "section_summary": "[根據學生在此部分的表現生成總結]"
            },
            {
            "section_title": "[考卷上的大標題(粗體)]",
            "questions_feedback": [
                {
                "question_number": "1",
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]",
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[標準答案]中對應題號的正確答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Lesson 1/Dialogue Mind Map/1:basketball]"
                }
            ],
            "section_summary": "[根據學生在此部分的表現生成總結，並依照III.的配分計分]"
            },
            {
            "section_title": "[考卷上的大標題(粗體)]",
            "questions_feedback": [ 
                {
                "question_number": "1", 
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]", 
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[標準答案]中對應題號的正確答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Lesson 1/Post-listening Questions and Answers/1:They worry about their grades at school.)]"
                }
            ],
            "section_summary": "[根據學生在此部分的表現生成總結]"
            }
        ],
        "overall_score_summary_title": "✅ 總分統計與等第建議",
        "score_breakdown_table": [
            {
            "section": "[考卷上的大標題(粗體)]",
            "max_score": "[根據考卷上的配分]",
            "obtained_score": "[計算此部分得分]"
            },
            {
            "section": "[考卷上的大標題(粗體)]",
            "max_score": "[根據考卷上的配分]",
            "obtained_score": "[計算此部分得分]"
            },
            {
            "section": "[考卷上的大標題(粗體)]",
            "max_score": "[根據考卷上的配分]",
            "obtained_score": "[計算此部分得分]"
            },
            {
            "section": "[考卷上的大標題(粗體)]",
            "max_score": "[根據考卷上的配分]",
            "obtained_score": "[計算此部分得分]"
            }
        ],
        "final_total_score_text": "總分：100 學生分數：[學生得分]",
        "final_suggested_grade_title": "🔺等第建議",
        "final_suggested_grade_text": "[根據總分生成建議等第與說明]",
        "overall_feedback_title": "📚 總結性回饋建議（可複製給學生）",
        "overall_feedback": "[針對學生考卷的作答整體表現生成正面總結性回饋]"
        }

        # 新增：讀寫習作評分的 JSON 結構範例
        mock_reading_writing_structure = {
        "submissionType": "讀寫習作評分",
        "title": "📘讀寫習作批改結果",
        "sections": [
            {
            "section_title": "I. [考卷上的大標題與配分]",
            "questions_feedback": [
                {
                "question_number": "1",
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]",
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[學生年級]習作標準答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Book 5/Lesson 2/I Read and Write/1:interests)]"
                },
                {
                "question_number": "2",
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]",
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[學生年級]習作標準答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Book 5/Lesson 2/I Read and Write/2:reason)]"
                }
            ],
            "section_summary": "[根據學生在此部分的表現生成總結，並依照I.的配分計分]"
            },
            {
            "section_title": "II. [考卷上的大標題與配分]",
            "questions_feedback": [
                {
                "question_number": "1",
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]",
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[學生年級]習作標準答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Book 5/Lesson 2/II Look and Fill In/1:tiring)]"
                }
            ],
            "section_summary": "[根據學生在此部分的表現生成總結，並依照II.的配分計分]"
            },
            {
            "section_title": "III. [考卷上的大標題與配分]",
            "questions_feedback": [
                {
                "question_number": "1",
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]",
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[學生年級]習作標準答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Book 5/Lesson 2/III Read and Write/1:James thought (that) Linda would like the gift.]"
                }
            ],
            "section_summary": "[根據學生在此部分的表現生成總結，並依照III.的配分計分]"
            },
            {
            "section_title": "IV.[考卷上的大標題與配分]",
            "questions_feedback": [ 
                {
                "question_number": "1", 
                "student_answer": "[學生實際的答案]",
                "is_correct": "[✅/❌]", 
                "comment": "[根據學生答案正確或錯誤生成內容]",
                "correct_answer": "[[學生年級]習作標準答案]",
                "answer_source_query":"[標準答案實際出處(search_tool(query=''))]",
                "answer_source_content":"[標準答案實際的內容(格式範例:Book 5/Lesson 2/IV Fill In/1:reasons / choice)]"
                }
            ],
            "section_summary": "[根據學生在此部分的表現生成總結，並依照IV.的配分計分]"
            }
        ],
        "overall_score_summary_title": "✅ 總分統計與等第建議",
        "score_breakdown_table": [
            {
            "section": "I. Vocabulary & Grammar",
            "max_score": "[根據考卷上的配分]",
            "obtained_score": "[計算此部分得分]"
            },
            {
            "section": "II. Cloze Test",
            "max_score": "[根據考卷上的配分]",
            "obtained_score": "[計算此部分得分]"
            },
            {
            "section": "III. Reading Comprehension",
            "max_score": "[根據考卷上的配分]",
            "obtained_score": "[計算此部分得分]"
            },
            {
            "section": "IV. Write",
            "max_score": "[根據考卷上的配分]",
            "obtained_score": "[計算此部分得分]"
            }
        ],
        "final_total_score_text": "總分：100 學生分數：[學生得分]",
        "final_suggested_grade_title": "🔺等第建議",
        "final_suggested_grade_text": "[根據總分生成建議等第與說明]",
        "overall_feedback_title": "📚 總結性回饋建議（可複製給學生）",
        "overall_feedback": "[針對學生考卷的作答整體表現生成正面總結性回饋]"
        }
        json_format_example_str = get_json_format_example(
            submission_type,
            mock_paragraph_data_for_structure,
            mock_quiz_data_for_structure,
            mock_learning_sheet_structure,
            mock_reading_writing_structure
        )

        # --- 4. 填充 Prompt 模板 ---
        # 假設您的 GCS Prompt 文件使用 Python 的 .format() 風格佔位符
        # 例如： "學生年級：{grade_level}\n作文內容：\n{essay_content}\nJSON範例：\n{json_example}"
        try:
            final_prompt = base_prompt_text.format(
                Book = bookrange if submission_type == '讀寫習作評分' else "",
                learnsheet = learnsheets if submission_type == '學習單批改' else "",
                grade_level=grade_level,
                submission_type=submission_type,
                essay_content=essay_content,
                standard_answer_if_any=processed_standard_answer if submission_type == '測驗寫作評改' else "",
                scoring_instructions_if_any=scoring_instructions if submission_type == '測驗寫作評改' else "",
                json_format_example_str=json_format_example_str,
                current_lesson_standard_answers_json=standard_answers_json_str
            )
        except KeyError as ke:
            print(f"Error formatting prompt: Missing key {ke} in prompt template or provided variables.")
            return jsonify({"error": f"Prompt template formatting error: Missing key {ke}"}), 500

        print(f"Final prompt (first 300 chars, excluding JSON example): {final_prompt.split('JSON 輸出格式範例：')}...")

        # 將 final_prompt 字符串作為第一個文本 Part 加入到 contents_for_gemini 列表的開頭
        contents_for_gemini.insert(0, Part.from_text(final_prompt))

        print(f"Total parts to send to Gemini: {len(contents_for_gemini)}")

        # --- 5. 調用 Gemini 模型 ---
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.5,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        # 設置安全設定，避免因安全問題被阻擋
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # 在呼叫 gemini_model.generate_content(...) 之前
        print(f"DEBUG: Size of final_prompt in memory: {get_size(final_prompt):.2f} MB")
        print(f"DEBUG: Size of contents_for_gemini list in memory: {get_size(contents_for_gemini):.2f} MB")

        print("Calling Gemini model with non-streaming mode...")
        
        # 【修改點 1】將呼叫 Gemini 的程式碼包在 try-except 中，以捕獲 API 可能的錯誤
        try:
            response = gemini_model.generate_content(
                contents_for_gemini,
                generation_config=generation_config,
                tools=tools_list,
                safety_settings=safety_settings  # 【新增】加入安全設定
            )
            print("Gemini model responded.")
        except Exception as api_error:
            print(f"Error calling Gemini API: {api_error}")
            traceback.print_exc()
            # 嘗試打印 response 的部分內容以供除錯
            error_details = str(api_error)
            return jsonify({"error": "AI model API call failed.", "details_for_log": error_details}), 500


        try:
            # 步驟 1: 檢查是否有候選答案
            if not response.candidates:
                finish_reason = getattr(response, 'prompt_feedback', 'No prompt_feedback attribute')
                if hasattr(finish_reason, 'block_reason'):
                    finish_reason = f"Blocked due to: {finish_reason.block_reason}"
                print(f"Error: Gemini response is empty or blocked. Finish reason: {finish_reason}")
                return jsonify({"error": "AI model did not return a valid response (it might have been blocked).", "details_for_log": str(finish_reason)}), 500

            # 步驟 2: 手動拼接所有 text parts
            # 這是解決 "Multiple content parts are not supported" 的關鍵
            full_response_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
            
            print(f"Combined raw AI response text (first 300 chars): {full_response_text[:300]}...")

            # 步驟 3: 從拼接後的完整字串中提取 JSON 區塊
            # 這個邏輯可以處理 JSON 前後有其他文字的情況
            json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", full_response_text, re.DOTALL)
            
            if json_block_match:
                ai_response_text = json_block_match.group(1)
                print("Successfully extracted JSON block using regex.")
            else:
                # 如果正則表達式找不到，作為備用方案，我們假設整個文本就是 JSON
                # 這可以處理模型直接返回純 JSON 的情況
                print("Warning: Could not find ```json``` block. Attempting to parse the whole text.")
                ai_response_text = full_response_text

            # 步驟 4: 解析提取出的 JSON 字串
            ai_result = json.loads(ai_response_text)

            # 可選的健全性檢查
            if 'submissionType' not in ai_result or ai_result.get('submissionType') != submission_type:
                print(f"Warning: AI returned submissionType ('{ai_result.get('submissionType')}') differs from request ('{submission_type}'). Correcting.")
                ai_result['submissionType'] = submission_type
            
            print("Successfully parsed AI JSON response.")
            return jsonify(ai_result)

        except json.JSONDecodeError as je:
            print(f"AI response JSON decode error: {je}")
            # 當解析失敗時，打印拼接後的完整文本，而不是可能不完整的 ai_response_text
            print(f"Problematic AI response text: {full_response_text}")
            return jsonify({"error": "AI response format error (cannot parse JSON).", "details_for_log": full_response_text[:500]}), 500
        
        except Exception as e:
            print(f"Error processing Gemini response: {e}")
            traceback.print_exc()
            error_detail = str(response) if len(str(response)) < 500 else str(response)[:500] + "..."
            return jsonify({"error": "Failed to process AI response.", "details_for_log": error_detail}), 500

    except Exception as e:
        print(f"Overall error in /api/grade: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error during grading.", "details": str(e)}), 500
# --- 應用程式啟動 ---
if __name__ == '__main__':
    # 從環境變數獲取 PORT，允許 Cloud Run 等平台設置
    port = int(os.environ.get("PORT", 5000))
    # 生產環境中 debug 應為 False
    app.run(host='0.0.0.0', port=port, debug=os.environ.get("FLASK_DEBUG", "True").lower() == "true")
