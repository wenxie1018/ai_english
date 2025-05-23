import io
import json
import os
import traceback 

from flask import Flask, request, jsonify
from flask_cors import CORS # type: ignore
from google.cloud import vision
from google.cloud import storage 

import vertexai # type: ignore
from vertexai.generative_models import GenerativeModel, Part, Tool, grounding, HarmCategory, HarmBlockThreshold # type: ignore

# --- 初始化 Flask 應用 ---
app = Flask(__name__)
CORS(app) # 開發階段允許所有來源，生產環境應配置具體來源

# --- 配置 ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "knsh-ai")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "global")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.0-flash")

DATASTORE_ID = os.environ.get("DATASTORE_ID", "ai-english-tutor_1747191591294")
DATASTORE_COLLECTION_LOCATION = "global"
DATASTORE_RESOURCE_NAME = f"projects/{GCP_PROJECT_ID}/locations/{DATASTORE_COLLECTION_LOCATION}/collections/default_collection/dataStores/{DATASTORE_ID}"

# GCS Bucket 用於存儲 Prompt 文件
GCS_PROMPT_BUCKET_NAME = os.environ.get("GCS_PROMPT_BUCKET_NAME", "ai_english_tutor")

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
def perform_ocr(image_file_storage):
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
    print(f"Request Content-Type: {request.content_type}") # <--- 添加這行來打印 Content-Type

    try:
        submission_type = None
        grade_level = None
        text_input = None
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
            text_input = data.get('text') # JSON 中通常不直接傳文件
            if submission_type == '測驗寫作評改':
                standard_answer_text = data.get('standardAnswerText', '')
                # JSON 通常不直接傳文件，所以 standard_answer_image_file 通常會是 None
                scoring_instructions = data.get('scoringInstructions', '')
        else:
            print(f"Warning: Received request with unhandled Content-Type: {request.content_type}")
            # 可以選擇在這裡返回錯誤，或者嘗試從 request.form 獲取（如果可能是 x-www-form-urlencoded）
            submission_type = request.form.get('submissionType') # 最後嘗試
            grade_level = request.form.get('gradeLevel')


        print(f"Parsed Submission Type: {submission_type}, Grade Level: {grade_level}")

        # --- 後續的驗證和處理邏輯 ---
        if not submission_type or not grade_level: # 現在這個判斷更關鍵
            return jsonify({"error": "Missing submissionType or gradeLevel after parsing request"}), 400


        essay_content = ""
        source_files_for_ocr = []
        if submission_type == '段落寫作評閱' or submission_type == '測驗寫作評改':
            if text_input:
                essay_content = text_input
            elif essay_image_files:
                source_files_for_ocr.extend(essay_image_files)
            else: # 前端應該已經做了驗證，但後端也做一次
                return jsonify({"error": f"For {submission_type}, text input or an essay image is required."}), 400
        elif submission_type == '學習單批改':
            if learning_sheet_files:
                source_files_for_ocr.extend(learning_sheet_files)
            else:
                return jsonify({"error": "Learning sheet file is required for '學習單批改'."}), 400
        elif submission_type == '讀寫習作評分':
            if reading_writing_files:
                source_files_for_ocr.extend(reading_writing_files)
            else:
                return jsonify({"error": "Reading/writing worksheet file is required for '讀寫習作評分'."}), 400
        else: # 處理未知的 submission_type (理論上 prompt_file_map 會先攔截)
            return jsonify({"error": f"Unsupported submission type for content processing: {submission_type}"}), 400

        # 如果需要 OCR，現在需要處理多個文件
        if source_files_for_ocr:
            ocr_results = []
            for file_storage in source_files_for_ocr:
                print(f"Performing OCR for: {file_storage.filename}")
                # 重置文件指針，如果文件之前被讀取過 (例如 request.files.getlist 可能已經讀取過一次)
                # 雖然 storage_client.read() 應該能處理，但以防萬一
                file_storage.seek(0)
                ocr_text = perform_ocr(file_storage)
                if "OCR_ERROR:" in ocr_text:
                    # 決定如何處理單個文件 OCR 失敗：是中止所有，還是忽略這個文件？
                    # 這裡選擇記錄錯誤並繼續，但最終 essay_content 可能會缺少一部分
                    print(f"OCR failed for {file_storage.filename}: {ocr_text}. Skipping this file.")
                    # return jsonify({"error": f"OCR failed for {file_storage.filename}: {ocr_text}"}), 400 # 或者直接報錯
                elif ocr_text.strip():
                    ocr_results.append(ocr_text)
            
                if not ocr_results: # 如果所有 OCR 都失敗或返回空
                 return jsonify({"error": "OCR failed or returned empty for all provided images."}), 400
            
                # 將所有 OCR 結果合併，通常按順序用換行符分隔
                # 你可能需要根據圖片的順序來決定如何合併，或者讓 Gemini 模型自己處理多段文本
                essay_content = "\n\n".join(ocr_results) # 使用雙換行符分隔不同圖片的內容

        if not essay_content.strip():
            return jsonify({"error": "Essay content is empty after processing input"}), 400
        print(f"Essay content (first 100 chars): {essay_content[:100]}...")
        
        processed_standard_answer = ""
        if submission_type == '測驗寫作評改':
            if standard_answer_text:
                processed_standard_answer = standard_answer_text
            elif standard_answer_image_files: # 檢查列表是否有內容
                print("Performing OCR for standard answer images...")
                ocr_standard_answer_results = []
                for file_storage in standard_answer_image_files:
                    file_storage.seek(0) # 重置指針
                    ocr_text = perform_ocr(file_storage)
                    if "OCR_ERROR:" in ocr_text:
                        print(f"OCR for standard answer image {file_storage.filename} failed or returned empty: {ocr_text}")
                        # 決定如何處理，這裡選擇忽略錯誤的
                    elif ocr_text.strip():
                        ocr_standard_answer_results.append(ocr_text)
                
                if ocr_standard_answer_results:
                    processed_standard_answer = "\n\n".join(ocr_standard_answer_results)
                else:
                    # 如果所有標準答案圖片 OCR 都失敗，processed_standard_answer 會是空
                    print("OCR for all standard answer images failed or returned empty.")
                    # 你可以選擇在這裡報錯，或者允許沒有標準答案的情況
            print(f"Processed Standard Answer (first 100 chars): {processed_standard_answer[:100]}...")

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
        "questions_feedback": [
            {
            "question_number": "I-1a",
            "student_answer": "He go to school.",
            "is_correct": "❌",
            "comment": "主詞動詞一致性錯誤，應為 goes。",
            "correct_answer": "He goes to school."
            },
            {
            "question_number": "I-1b",
            "student_answer": "She is a good student.",
            "is_correct": "✅",
            "comment": "回答正確。",
            "correct_answer": "She is a good student."
            },
            {
            "question_number": "II-1 (看圖回答)",
            "student_answer": "The cat is on the table.",
            "is_correct": "可接受答案",
            "comment": "語法正確，與參考答案 'The cat sits on the table.' 意思相近。",
            "correct_answer": "The cat sits on the table."
            },
            {
            "question_number": "III-1 (造句)",
            "student_answer": "I like apple.",
            "is_correct": "❌",
            "comment": "名詞 'apple' 作為可數名詞單數時，前面通常需要冠詞，或使用複數形式。且 'apple' 用字未在建議範圍內。",
            "correct_answer": "I like an apple. / I like apples. (建議使用教材範圍內水果，如 I like bananas.)"
            }
        ],
        "score_summary_title": "📊 總分統計",
        "score_details": [
            {
            "description": "正確及可接受：3 題 × 25 分 (估計)",
            "points": "75 分"
            },
            {
            "description": "錯誤：1 題 × 25 分 (估計)",
            "points": "-25 分"
            }
        ],
        "total_score_text": "總分：75.0 / 100 分 (估計)",
        "suggested_grade_title": "🎓 等第建議",
        "suggested_grade": "B",
        "feedback_summary_title": "📌 回饋建議",
        "feedback_summary": "本次學習單作答情況尚可。在選擇題和部分簡答題上表現不錯，能正確運用所學句型。但在造句和部分文法細節（如主詞動詞一致性、冠詞使用）上仍有進步空間。請特別注意參考答案中的正確用法，並多加練習。詞彙方面，請盡量使用教材建議範圍內的單字。繼續努力！"
        }

        # 新增：讀寫習作評分的 JSON 結構範例
        mock_reading_writing_structure = {
        "submissionType": "讀寫習作評分",
        "title": "📘讀寫習作批改結果",
        "sections": [
            {
            "section_title": "I. Vocabulary & Grammar (選擇題)",
            "questions_feedback": [
                {
                "question_number": "1",
                "student_answer": "B) is",
                "is_correct": "✅",
                "comment": "回答正確。",
                "correct_answer": "B) is"
                },
                {
                "question_number": "2",
                "student_answer": "A) go",
                "is_correct": "❌",
                "comment": "主詞為第三人稱單數 He，動詞應為 goes。",
                "correct_answer": "C) goes"
                }
            ],
            "section_summary": "共 10 題，8 題正確，預估得分 16 分 (每題2分估計)"
            },
            {
            "section_title": "II. Cloze Test (克漏字)",
            "questions_feedback": [
                {
                "question_number": "1",
                "student_answer": "playing",
                "is_correct": "✅",
                "comment": "回答正確。",
                "correct_answer": "playing"
                }
            ],
            "section_summary": "共 5 題，4 題正確，預估得分 8 分 (每題2分估計)"
            },
            {
            "section_title": "III. Reading Comprehension (閱讀測驗)",
            "questions_feedback": [
                {
                "question_number": "1",
                "student_answer": "A) Because he was sick.",
                "is_correct": "✅",
                "comment": "回答正確。",
                "correct_answer": "A) Because he was sick."
                }
            ],
            "section_summary": "共 5 題，5 題正確，預估得分 15 分 (每題3分估計)"
            },
            {
            "section_title": "IV. Write (引導式寫作)",
            "questions_feedback": [ 
                {
                "question_number": "IV", 
                "student_answer": "My favorite season is summer. I like summer because I can go swimming. The weather is hot, but I feel happy. I also like to eat ice cream in summer. Summer is a wonderful season.",
                "is_correct": "可接受答案", 
                "comment": "內容表達基本清晰，文法大致正確。句子結構可以更豐富些。使用了 'wonderful'，詞彙尚可。符合題目要求。預估符合程度：良好 (約占此題滿分的75-85%)",
                "correct_answer": "（此為開放性寫作題，無唯一標準答案。可參考範文：My favorite season is summer. I enjoy it because the long, sunny days allow me to go swimming at the beach. Although the weather can be very hot, I always feel energetic and happy. Eating delicious ice cream is another great pleasure of summer. For me, summer is truly a fantastic season full of joy.）"
                }
            ],
            "section_summary": "引導式寫作：內容表達良好，文法尚可，預估得分 12 分 (滿分15分估計)"
            }
        ],
        "overall_score_summary_title": "✅ 總分統計與等第建議",
        "score_breakdown_table": [
            {
            "section": "I. Vocabulary & Grammar",
            "max_score": "25 分 (估計)",
            "obtained_score": "16 分"
            },
            {
            "section": "II. Cloze Test",
            "max_score": "25 分 (估計)",
            "obtained_score": "8 分"
            },
            {
            "section": "III. Reading Comprehension",
            "max_score": "25 分 (估計)",
            "obtained_score": "15 分"
            },
            {
            "section": "IV. Write",
            "max_score": "25 分 (估計)",
            "obtained_score": "12 分"
            }
        ],
        "final_total_score_text": "總分 100 (估計滿分) 得分 51",
        "final_suggested_grade_title": "🔺等第建議",
        "final_suggested_grade_text": "B (表現良好，多數題目回答正確，寫作部分有發展空間【參考檔案規則：總分百分比 80-89% 為 B】)",
        "overall_feedback_title": "📚 總結性回饋建議（可複製給學生）",
        "overall_feedback": "本次讀寫習作整體表現良好。你在選擇題和閱讀測驗部分展現了不錯的理解和判斷能力，大部分題目都能正確作答。克漏字部分也掌握得不錯。引導式寫作方面，內容表達基本清晰，但可以嘗試運用更多元的句型和詞彙來豐富表達，並注意文法細節的準確性。請繼續加強寫作練習，並仔細對照參考答案中的錯誤點進行訂正，相信會有更大的進步！"
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
                grade_level=grade_level,
                submission_type=submission_type, # Prompt 模板中可能需要
                essay_content=essay_content,
                standard_answer_if_any=processed_standard_answer if submission_type == '測驗寫作評改' else "",
                scoring_instructions_if_any=scoring_instructions if submission_type == '測驗寫作評改' else "",
                json_format_example_str=json_format_example_str
            )
        except KeyError as ke:
            print(f"Error formatting prompt: Missing key {ke} in prompt template or provided variables.")
            return jsonify({"error": f"Prompt template formatting error: Missing key {ke}"}), 500

        print(f"Final prompt (first 300 chars, excluding JSON example): {final_prompt.split('JSON 輸出格式範例：')[0][:300]}...")

        # --- 5. 調用 Gemini 模型 ---
        generation_config = {
            "temperature": 0.7,
            "top_p": 1,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        print("Calling Gemini model...")
        response = gemini_model.generate_content(
            final_prompt,
            generation_config=generation_config,
            tools=tools_list
        )
        print("Gemini model responded.")

        # --- 6. 處理 Gemini 的回應 ---
        if not response.candidates or not response.candidates[0].content.parts:
            error_detail = str(response) if len(str(response)) < 500 else str(response)[:500] + "..."
            print(f"Error: Gemini response empty or not as expected. Details: {error_detail}")
            return jsonify({"error": "AI model did not return a valid response.", "details_for_log": error_detail}), 500

        ai_response_text = response.text
        print(f"Raw AI response text (first 300 chars): {ai_response_text[:300]}...")

        # 清理常見的 Markdown 包裝
        if ai_response_text.strip().startswith("```json"):
            ai_response_text = ai_response_text.strip()[7:-3].strip()
        elif ai_response_text.strip().startswith("```"):
            ai_response_text = ai_response_text.strip()[3:-3].strip()

        try:
            ai_result = json.loads(ai_response_text)
            if 'submissionType' not in ai_result or ai_result.get('submissionType') != submission_type:
                print(f"Warning: AI returned submissionType ('{ai_result.get('submissionType')}') differs from request ('{submission_type}'). Correcting.")
                ai_result['submissionType'] = submission_type
            print("Successfully parsed AI JSON response.")
            return jsonify(ai_result)
        except json.JSONDecodeError as je:
            print(f"AI response JSON decode error: {je}")
            print(f"Problematic AI response text: {ai_response_text}")
            return jsonify({"error": "AI response format error (cannot parse JSON).", "details_for_log": ai_response_text[:500]}), 500

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