import io
import json
import os
import re
import traceback
from typing import List, Optional, Dict, Any, Union

# --- 1. FastAPI 和 Pydantic 相關導入 ---
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Google Cloud 和 Vertex AI 相關導入 ---
from google.cloud import vision
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Tool, grounding, HarmCategory, HarmBlockThreshold

# --- 環境變數加載 ---
from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# 2. PYDANTIC 模型 (保持不變)
# ==============================================================================
# ... (此處省略所有 Pydantic 模型定義，與前一版本相同) ...
# --- "段落寫作評閱" 的模型 ---
class ErrorAnalysisItem(BaseModel):
    original_sentence: str
    error_type: str
    error_content: str
    suggestion: str

class RubricItem(BaseModel):
    item: str
    score: int
    comment: str

class RubricEvaluation(BaseModel):
    structure_performance: List[RubricItem]
    content_language: List[RubricItem]

class OverallAssessment(BaseModel):
    total_score: str
    suggested_grade: str
    grade_basis: str
    general_comment: str

class ParagraphResponse(BaseModel):
    submissionType: str = Field(..., example="段落寫作評閱")
    error_analysis: List[ErrorAnalysisItem]
    rubric_evaluation: RubricEvaluation
    overall_assessment: OverallAssessment
    model_paragraph: str
    teacher_summary_feedback: str

# --- "測驗寫作評改" 的模型 ---
class ErrorAnalysisTableItem(BaseModel):
    original_sentence: str
    error_type: str
    problem_description: str
    suggestion: str

class SummaryFeedback(BaseModel):
    summary_feedback: str
    total_score_display: str
    suggested_grade_display: str
    grade_basis_display: str
    
class RevisedDemonstration(BaseModel):
    original_with_errors_highlighted: str
    suggested_revision: str

class QuizResponse(BaseModel):
    submissionType: str = Field(..., example="測驗寫作評改")
    error_analysis_table: List[ErrorAnalysisTableItem]
    summary_feedback_for_student: SummaryFeedback
    revised_demonstration: RevisedDemonstration
    positive_learning_feedback: str

# --- "學習單批改" & "讀寫習作評分" 的通用模型 ---
class QuestionFeedback(BaseModel):
    question_number: str
    student_answer: str
    is_correct: str
    comment: str
    correct_answer: str
    answer_source_query: str
    answer_source_content: str

class SectionFeedback(BaseModel):
    section_title: str
    questions_feedback: List[QuestionFeedback]
    section_summary: str

class ScoreBreakdownItem(BaseModel):
    section: str
    # 【最終修正】允許 max_score 是字串、整數或浮點數
    max_score: Union[str, int, float]
    # 【最終修正】允許 obtained_score 是字串、整數或浮點數
    obtained_score: Union[str, int, float]

class WorksheetResponse(BaseModel):
    submissionType: str = Field(..., example="學習單批改")
    title: str
    sections: List[SectionFeedback]
    overall_score_summary_title: str
    score_breakdown_table: List[ScoreBreakdownItem]
    final_total_score_text: str
    final_suggested_grade_title: str
    final_suggested_grade_text: str
    # 【修正】將這兩個欄位標記為可選的 (Optional)
    # 這樣即使 AI 沒有返回它們，驗證也能通過
    overall_feedback_title: Optional[str] = None
    overall_feedback: Optional[str] = None

# 使用 Union 來定義路由可能返回的多種類型，增強類型提示
ApiResponse = Union[ParagraphResponse, QuizResponse, WorksheetResponse]

# ==============================================================================
# 3. FastAPI 應用初始化與配置 (保持不變)
# ==============================================================================
# ... (此處省略 FastAPI app 和 Google Cloud client 的初始化，與前一版本相同) ...
# --- 初始化 FastAPI 應用 ---
app = FastAPI(
    title="AI 英文家教 API",
    description="使用 Vertex AI Gemini 對多種類型的英文寫作作業進行評分。",
    version="1.0.0"
)

# --- 配置 CORS (跨來源資源共用) 中介軟體 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有 HTTP 方法
    allow_headers=["*"],  # 允許所有請求標頭
)


# --- 配置 ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1") # 範例: us-central1
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME")
DATASTORE_ID = os.environ.get("DATASTORE_ID")
GCS_PROMPT_BUCKET_NAME = os.environ.get("GCS_PROMPT_BUCKET_NAME")

# 檢查必要的環境變數是否已設定
if not all([GCP_PROJECT_ID, GEMINI_MODEL_NAME, DATASTORE_ID, GCS_PROMPT_BUCKET_NAME]):
    raise ValueError("一個或多個必要的環境變數未設定 (GCP_PROJECT_ID, GEMINI_MODEL_NAME, DATASTORE_ID, GCS_PROMPT_BUCKET_NAME)")

DATASTORE_COLLECTION_LOCATION = "global"
DATASTORE_RESOURCE_NAME = f"projects/{GCP_PROJECT_ID}/locations/{DATASTORE_COLLECTION_LOCATION}/collections/default_collection/dataStores/{DATASTORE_ID}"


# --- 在應用程式啟動時初始化 Google Cloud 客戶端 ---
try:
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    vision_client = vision.ImageAnnotatorClient()
    storage_client = storage.Client(project=GCP_PROJECT_ID)

    gemini_model = GenerativeModel(GEMINI_MODEL_NAME)
    search_tool = Tool.from_retrieval(
        grounding.Retrieval(grounding.VertexAISearch(datastore=DATASTORE_RESOURCE_NAME))
    )
    tools_list = [search_tool]
    print("Vertex AI 和 Google Cloud 客戶端初始化成功。")

except Exception as e:
    print(f"嚴重錯誤: 初始化 Google Cloud 客戶端失敗: {e}")
    traceback.print_exc()
    # 在實際應用中，如果客戶端初始化失敗，您可能希望程式退出
    
# --- 用於生成 JSON 結構的模擬數據 (保留自原始碼，用於生成 Prompt) ---
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
# ==============================================================================
# 4. 異步輔助函數 (增加了 print 語句)
# ==============================================================================

async def perform_ocr(image_file: UploadFile) -> str:
    if not image_file or not image_file.filename:
        return "OCR_ERROR: 未提供圖片檔案。"
    try:
        print(f"  [OCR] 正在處理檔案: {image_file.filename} (大小: {image_file.size} bytes)")
        content = await image_file.read()
        image = vision.Image(content=content)
        response = vision_client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(f"Vision API 錯誤: {response.error.message}")
        
        ocr_text = response.text_annotations[0].description if response.text_annotations else ""
        print(f"  [OCR] 完成。識別出 {len(ocr_text)} 個字符。")
        return ocr_text
        
    except Exception as e:
        print(f"  !!! ERROR in perform_ocr !!!")
        traceback.print_exc() # 打印詳細的錯誤堆疊
        return f"OCR_ERROR: {str(e)}"

async def get_gcs_blob_text(bucket_name: str, file_path: str) -> Optional[str]:
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        print(f"  [GCS] 正在讀取: gs://{bucket_name}/{file_path}")
        if not blob.exists():
            print(f"  !!! ERROR in get_gcs_blob_text: 檔案不存在。")
            return None
        
        text_content = blob.download_as_text()
        print(f"  [GCS] 成功讀取 {len(text_content)} 個字符。")
        return text_content

    except Exception as e:
        print(f"  !!! ERROR in get_gcs_blob_text !!!")
        traceback.print_exc()
        return None
    
async def get_standard_answer_from_gcs(
    bucket_name: str,
    base_path: str,
    grade_level: str,
    category_key: str,
    answer_map: Dict[str, str],
    lookup_key: str
) -> Optional[Dict[str, Any]]:
    """從 GCS 獲取指定主題的標準答案 JSON。"""
    file_key = f"{grade_level}{category_key}"
    target_filename = answer_map.get(file_key)
    if not target_filename:
        print(f"錯誤: 找不到對應的標準答案檔案，鍵值為 '{file_key}'。")
        return None
    
    full_path = f"{base_path}{target_filename}"
    json_content = await get_gcs_blob_text(bucket_name, full_path)
    
    if not json_content:
        return None
        
    try:
        all_data = json.loads(json_content)
        lesson_data = all_data.get(lookup_key)
        
        if not lesson_data:
            print(f"警告: 在檔案 {target_filename} 中找不到鍵 '{lookup_key}'。")
            return None
        
        print(f"成功為 {lookup_key} 載入標準答案。")
        return lesson_data
    except json.JSONDecodeError as e:
        print(f"解析 GCS 的 JSON 檔案時出錯 ({full_path}): {e}")
        return None
    
def get_json_format_example(submission_type: str) -> str:
    """根據提交類型返回對應的 JSON 格式範例字串。"""
    examples = {
        '測驗寫作評改': mock_quiz_data_for_structure,
        '段落寫作評閱': mock_paragraph_data_for_structure,
        '學習單批改': mock_learning_sheet_structure,
        '讀寫習作評分': mock_reading_writing_structure,
    }
    # 如果類型未知，默認為段落寫作
    mock_data = examples.get(submission_type, mock_paragraph_data_for_structure)
    return json.dumps(mock_data, ensure_ascii=False, indent=2)
        
# ... (其他輔助函數 get_standard_answer_from_gcs 和 get_json_format_example 保持不變) ...

# ==============================================================================
# 5. 主要 API 路由 (增加了 print 語句)
# ==============================================================================

@app.post("/api/grade", response_model=ApiResponse, tags=["評分"])
async def grade_writing(
    # --- 路由參數保持不變 ---
    submissionType: str = Form(..., description="提交類型"),
    gradeLevel: str = Form(..., description="學生年級"),
    text: Optional[str] = Form(None, description="純文字輸入的作業內容"),
    bookrange: Optional[str] = Form(None, description="冊次"),
    learnsheets: Optional[str] = Form(None, description="學習單主題"),
    worksheetCategory: Optional[str] = Form(None, description="學習單類別"),
    standardAnswerText: Optional[str] = Form("", description="純文字的標準答案"),
    scoringInstructions: Optional[str] = Form("", description="評分說明"),
    essayImage: List[UploadFile] = File([], description="作文圖片檔案"),
    learningSheetFile: List[UploadFile] = File([], description="學習單圖片檔案"),
    readingWritingFile: List[UploadFile] = File([], description="讀寫習作圖片檔案"),
    standardAnswerImage: List[UploadFile] = File([], description="標準答案圖片檔案")
):
    print("\n" + "="*80)
    print("||" + " " * 28 + "開始處理新的評分請求" + " " * 28 + "||")
    print("="*80 + "\n")

    try:
        # --- 階段 1: 打印收到的請求參數 ---
        print("--- [1. 接收到的請求參數] ---")
        print(f"  - submissionType: {submissionType}")
        print(f"  - gradeLevel: {gradeLevel}")
        print(f"  - text: {'(有內容)' if text else '(無)'}")
        print(f"  - bookrange: {bookrange or '(無)'}")
        print(f"  - learnsheets: {learnsheets or '(無)'}")
        print(f"  - worksheetCategory: {worksheetCategory or '(無)'}")
        print(f"  - essayImage: {len(essayImage)} 個檔案")
        print(f"  - learningSheetFile: {len(learningSheetFile)} 個檔案")
        print(f"  - readingWritingFile: {len(readingWritingFile)} 個檔案")
        print(f"  - standardAnswerImage: {len(standardAnswerImage)} 個檔案\n")
        
        contents_for_gemini: List[Part] = []
        essay_content = ""
        
        student_files = []
        if submissionType in ['段落寫作評閱', '測驗寫作評改']: student_files = essayImage
        elif submissionType == '學習單批改': student_files = learningSheetFile
        elif submissionType == '讀寫習作評分': student_files = readingWritingFile
        
        # --- 階段 2: 處理學生作業內容 ---
        print("--- [2. 處理學生作業內容 (OCR 或文字)] ---")
        if text:
            essay_content = text
            print("  - 使用了純文字輸入。")
        elif student_files:
            print("  - 正在處理上傳的圖片檔案...")
            ocr_results = []
            contents_for_gemini.append(Part.from_text("以下是學生提交的原始作業圖片，供您參考其版面和手寫內容："))
            for file in student_files:
                await file.seek(0)
                ocr_text = await perform_ocr(file)
                if "OCR_ERROR:" not in ocr_text and ocr_text.strip():
                    ocr_results.append(ocr_text)
                await file.seek(0)
                image_data = await file.read()
                contents_for_gemini.append(Part.from_data(data=image_data, mime_type=file.content_type))
            if not ocr_results: raise HTTPException(status_code=400, detail="所有圖片的 OCR 均失敗，且未提供純文字輸入。")
            essay_content = "\n\n".join(ocr_results)
        else:
            raise HTTPException(status_code=400, detail=f"對於 '{submissionType}'，必須提供純文字輸入或圖片檔案。")
        
        print(f"\n  [作業內容預覽 (前 300 字)]:\n---\n{essay_content[:300]}\n---\n")

        # --- 階段 3: 處理標準答案 (若有) ---
        print("--- [3. 處理標準答案] ---")
        processed_standard_answer = ""
        if submissionType == '測驗寫作評改':
            if standardAnswerText:
                processed_standard_answer = standardAnswerText
                print("  - 使用了純文字輸入的標準答案。")
            elif standardAnswerImage:
                print("  - 正在處理上傳的標準答案圖片...")
                # ... (此處省略 OCR 邏輯，與上面類似) ...
            else:
                print("  - 無標準答案提供。")
        else:
            print("  - 此提交類型無需標準答案。")
        
        if processed_standard_answer:
            print(f"\n  [標準答案預覽 (前 200 字)]:\n---\n{processed_standard_answer[:200]}\n---\n")

        # --- 階段 4: 從 GCS 獲取結構化答案 (若有) ---
        print("--- [4. 從 GCS 獲取結構化答案] ---")
        standard_answers_json_str = ""

        # 【修正】將這段邏輯加回來
        if submissionType == '學習單批改' and learnsheets and worksheetCategory:
            print(f"  - 條件滿足，嘗試為「學習單批改」載入答案...")
            answer_map = { 
                "七年級全英提問學習單參考答案":"全英提問學習單參考答案(01_1下).txt", 
                "八年級全英提問學習單參考答案":"全英提問學習單參考答案(01_2下).txt", 
                "九年級全英提問學習單參考答案":"全英提問學習單參考答案(01_3下).txt", 
                "七年級差異化學習單參考答案":"差異化學習單參考答案(01_1下).txt", 
                "八年級差異化學習單參考答案":"差異化學習單參考答案(01_2下).txt", 
                "九年級差異化學習單參考答案":"差異化學習單參考答案(01_3下).txt" 
            }
            # 之前這裡缺少了 get_standard_answer_from_gcs 這個輔助函數的實現
            # 我們假設這個函數存在，如果不存在需要加回來
            standard_answers_data = await get_standard_answer_from_gcs(
                GCS_PROMPT_BUCKET_NAME, 
                "ai_english_file/", 
                gradeLevel, 
                worksheetCategory, 
                answer_map, 
                learnsheets
            )
            if standard_answers_data:
                standard_answers_json_str = json.dumps(standard_answers_data, ensure_ascii=False, indent=2)

        elif submissionType == '讀寫習作評分' and bookrange:
            print(f"  - 條件滿足，嘗試為「讀寫習作評分」載入答案...")
            answer_map = { 
                "七年級讀寫習作參考答案": "113_1習作標準答案.txt", 
                "八年級讀寫習作參考答案": "113_2習作標準答案.txt", 
                "九年級讀寫習作參考答案": "113_3習作標準答案.txt" 
            }
            standard_answers_data = await get_standard_answer_from_gcs(
                GCS_PROMPT_BUCKET_NAME, 
                "ai_english_file/", 
                gradeLevel, 
                "讀寫習作參考答案", 
                answer_map, 
                bookrange
            )
            if standard_answers_data:
                standard_answers_json_str = json.dumps(standard_answers_data, ensure_ascii=False, indent=2)

        # 檢查結果
        if standard_answers_json_str:
            print(f"\n  [GCS 答案預覽 (前 200 字)]:\n---\n{standard_answers_json_str[:200]}\n---\n")
        else:
            print("  - 無需或未找到 GCS 結構化答案。\n")

        # --- 階段 5: 準備並打印最終 Prompt ---
        print("--- [5. 準備最終 Prompt] ---")
        prompt_map = { "段落寫作評閱": "段落寫作評閱.txt", "測驗寫作評改": "測驗寫作評改.txt", "學習單批改": "學習單批改.txt", "讀寫習作評分": "讀寫習作評分.txt" }
        prompt_file = prompt_map.get(submissionType)
        if not prompt_file: raise HTTPException(status_code=400, detail=f"不支持的提交類型: {submissionType}")
        prompt_path = f"ai_english_prompt/{prompt_file}"
        base_prompt_text = await get_gcs_blob_text(GCS_PROMPT_BUCKET_NAME, prompt_path)
        if not base_prompt_text: raise HTTPException(status_code=500, detail="從 GCS 載入 Prompt 模板失敗。")
        
        final_prompt_text = base_prompt_text.format(
            Book=bookrange or "",
            learnsheet=learnsheets or "",
            grade_level=gradeLevel,
            submission_type=submissionType,
            essay_content=essay_content,
            standard_answer_if_any=processed_standard_answer,
            scoring_instructions_if_any=scoringInstructions or "",
            json_format_example_str=get_json_format_example(submissionType),
            current_lesson_standard_answers_json=standard_answers_json_str
        )
        contents_for_gemini.insert(0, Part.from_text(final_prompt_text))
        
        # 打印最終的 Prompt (不含冗長的 JSON 範例)
        prompt_preview = final_prompt_text.split("JSON 輸出格式範例：")[0]
        print("\n  [最終 Prompt 預覽 (發送給 Gemini 的內容)]:")
        print("-" * 50)
        print(prompt_preview)
        print("-" * 50 + "\n")

        # --- 階段 6: 呼叫 Gemini API ---
        print("--- [6. 呼叫 Gemini API] ---")
        generation_config = { "temperature": 0.1, "top_p": 0.5, "max_output_tokens": 8192, "response_mime_type": "application/json" }
        safety_settings = { category: HarmBlockThreshold.BLOCK_NONE for category in HarmCategory }
        
        print("  - 正在發送請求...")
        response = gemini_model.generate_content(
            contents_for_gemini,
            generation_config=generation_config,
            tools=tools_list,
            safety_settings=safety_settings
        )
        print("  - 已收到 Gemini 回應。\n")

        # --- 階段 7: 處理並返回結果 ---
        print("--- [7. 處理 API 回應] ---")

        # 檢查是否有任何候選回應
        if not response.candidates:
            # 獲取被攔截的原因
            reason = "未知原因"
            if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                reason = response.prompt_feedback.block_reason
            
            # 記錄詳細日誌並返回具體的錯誤訊息給前端
            error_detail = f"AI 模型未返回任何回應，可能已被安全設定阻擋。原因: {reason}"
            print(f"!!! ERROR: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)

        # 獲取回應文字
        response_text = "".join(part.text for part in response.candidates[0].content.parts)
        print(f"  [Gemini 原始回應預覽 (前 500 字)]:\n---\n{response_text[:500]}\n---\n")

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        cleaned_text = json_match.group(1) if json_match else response_text.strip()

        # ======================= 【關鍵修正】 =======================
        # 在執行 json.loads 之前，檢查 cleaned_text 是否為空或無效
        if not cleaned_text or not cleaned_text.strip().startswith(('{', '[')):
            error_detail = "AI 模型返回了空的或無效的 JSON 內容。"
            # 在伺服器日誌中記錄完整的原始回應，方便排錯
            print(f"!!! ERROR: {error_detail}")
            print(f"--- Gemini 原始無效回應 ---")
            print(response_text)
            print(f"--------------------------")
            # 返回一個更友善的錯誤給前端
            raise HTTPException(status_code=500, detail=f"{error_detail} 請檢查輸入內容或稍後再試。")
        # ==========================================================

        try:
            ai_json = json.loads(cleaned_text)
            print("  - JSON 解析成功，準備返回結果。")
            # ... (後續的返回邏輯)

        except json.JSONDecodeError as e:
            # 處理雖然不是空，但格式錯誤的 JSON
            error_detail = f"AI 模型返回的 JSON 格式錯誤: {e}"
            print(f"!!! ERROR: {error_detail}")
            print(f"--- Gemini 格式錯誤的 JSON ---")
            print(cleaned_text)
            print(f"------------------------------")
            raise HTTPException(status_code=500, detail=error_detail)


        print("\n" + "="*80)
        print("||" + " " * 31 + "請求處理完成" + " " * 31 + "||")
        print("="*80 + "\n")

        if submissionType == '段落寫作評閱': return ParagraphResponse.model_validate(ai_json)
        elif submissionType == '測驗寫作評改': return QuizResponse.model_validate(ai_json)
        elif submissionType in ['學習單批改', '讀寫習作評分']: return WorksheetResponse.model_validate(ai_json)
        else: return ai_json

    except Exception as e:
        # 統一的錯誤處理，打印 traceback 並返回 HTTP 錯誤
        print("\n" + "!"*80)
        print("!!!" + " " * 31 + "請求處理時發生錯誤" + " " * 31 + "!!!")
        print("!"*80 + "\n")
        traceback.print_exc()
        # 如果是 HTTPException，就重新拋出它，否則包裝成一個通用的 500 錯誤
        if isinstance(e, HTTPException):
            raise e
        else:
            raise HTTPException(status_code=500, detail=f"伺服器內部發生未知錯誤: {str(e)}")

# ==============================================================================
# 6. 伺服器運行說明 (保持不變)
# ==============================================================================
# ... (此處省略運行說明) ...