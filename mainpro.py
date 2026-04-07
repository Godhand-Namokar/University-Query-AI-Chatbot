import os
import logging
import pandas as pd
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters,
    ContextTypes, ConversationHandler
)
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import pipeline

# === Load environment variables ===
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")
USER_DB = os.path.join(BASE_DIR, "Databaseai.xlsx.xlsx")
TIMETABLE_XLSX = os.path.join(BASE_DIR, "BTech_Sem2_Timetable.xlsx")
HOLIDAY_XLSX = os.path.join(BASE_DIR, "MITVPU_Holidays_Important_Dates.xlsx")
TIMETABLE_FAQ = os.path.join(BASE_DIR, os.getenv("TIMETABLE_FAQ", "Timetable_FAQs.txt"))

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

PHONE = 1
df_tt_global = None # Global DataFrame for timetable

# === Phone number normalization helper ===
def normalize_phone_number(phone: str) -> str:
    """Normalize phone numbers to digits only, removing +, spaces, dashes, and parentheses."""
    return re.sub(r"[^\d]", "", phone)

# === Data loading and preparation ===
def load_data():
    global df_tt_global
    try:
        df_tt_global = pd.read_excel(TIMETABLE_XLSX)
        logger.info(f"Successfully loaded timetable from {TIMETABLE_XLSX}")
    except FileNotFoundError:
        logger.error(f"Timetable file not found at {TIMETABLE_XLSX}. Please ensure the file exists.")
        df_tt_global = pd.DataFrame({"Time": [], "Monday": [], "Tuesday": [], "Wednesday": [], "Thursday": [], "Friday": []})

    df_users = pd.read_excel(USER_DB)
    df_users.columns = df_users.columns.str.strip()   # Remove whitespace from column names
    df_holidays = pd.read_excel(HOLIDAY_XLSX, header=None)
    faq_content = []
    if os.path.exists(TIMETABLE_FAQ):
        with open(TIMETABLE_FAQ, 'r', encoding='utf-8') as f:
            faq_content = [line.strip() for line in f.readlines() if line.strip()]
    else:
        logger.warning(f"FAQ file not found at {TIMETABLE_FAQ}")
    return df_tt_global, df_users, df_holidays, faq_content

def transform_timetable(df_tt):
    long_format = []
    if df_tt is not None and not df_tt.empty:
        for _, row in df_tt.iterrows():
            time_slot = row['Time']
            for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                subject = row.get(day)
                if pd.notna(subject):
                    long_format.append(f"On {day}, at {time_slot}, {subject} is scheduled.")
    return long_format

def parse_holidays(df_holidays):
    df = df_holidays.dropna(how="all")
    events = []
    for _, row in df.iterrows():
        date_str = str(row[0])
        desc = str(row[1]) if pd.notna(row[1]) else ""
        events.append((date_str, desc))
    return events

def check_today_event(events):
    today = datetime.now().strftime("%b %d").lower()
    for date_str, desc in events:
        if today in date_str.lower():
            return desc
    return None

def prepare_documents(df_tt, faq_content):
    timetable_sentences = transform_timetable(df_tt)
    timetable_sentences += [f"FAQ: {entry}" for entry in faq_content]
    document = Document(page_content="\n".join(timetable_sentences))
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents([document])

def timetable_to_markdown_table(df_timetable):
    if df_timetable is None or df_timetable.empty:
        return "Timetable data is not available."
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    header_columns = ["Time"] + [day for day in days if day in df_timetable.columns]
    lines = []
    lines.append("| " + " | ".join(header_columns) + " |")
    lines.append("|" + "|".join(["---"] * len(header_columns)) + "|")
    for _, row in df_timetable.iterrows():
        line_parts = []
        for col in header_columns:
            cell_value = row.get(col, "")
            cell_value_str = str(cell_value) if pd.notna(cell_value) else ""
            line_parts.append(cell_value_str)
        lines.append("| " + " | ".join(line_parts) + " |")
    return "\n".join(lines)

# === LangChain and data initialization ===
qa_chain = None
parsed_events = []

try:
    df_tt_global, df_users_global, df_holidays, faq_content = load_data()
    doc_chunks = prepare_documents(df_tt_global, faq_content)
    parsed_events = parse_holidays(df_holidays)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)

    qa_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        device="cpu"
    )
    llm = HuggingFacePipeline(pipeline=qa_model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
except Exception as e:
    logger.error(f"Initialization error: {e}")

# === Robust user verification with debug output ===
def verify_phone(phone: str) -> dict | None:
    try:
        normalized_input = normalize_phone_number(phone)
        print(f"\n--- Phone Verification Debug ---")
        print(f"User phone (original): {phone}")
        print(f"User phone (normalized): {normalized_input}")
        print("DataFrame columns:", df_users_global.columns.tolist())
        print("First 5 rows of DataFrame:")
        print(df_users_global.head())
        db_phones = df_users_global['phone_number'].astype(str).apply(normalize_phone_number)
        print("Database phones (normalized):", db_phones.tolist())
        matched = df_users_global[
            (db_phones == normalized_input) |
            (db_phones.str[-10:] == normalized_input[-10:])
        ]
        print("Matched rows:")
        print(matched)
        print("--- End Debug ---\n")
        return matched.iloc[0].to_dict() if not matched.empty else None
    except Exception as e:
        logger.error(f"Phone verification error: {e}")
        return None

# === Telegram Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today_event = check_today_event(parsed_events)
    message = "📚 Welcome! Please verify your phone number to continue."
    if today_event:
        message = f"⚠️ Today: {today_event}\n\n" + message

    await update.message.reply_text(
        message,
        reply_markup=ReplyKeyboardMarkup(
            [[KeyboardButton(text="📱 Share Phone Number", request_contact=True)]],
            one_time_keyboard=True,
            resize_keyboard=True,
        ),
    )
    return PHONE

async def handle_phone(update: Update, context: ContextTypes.DEFAULT_TYPE):
    contact = update.message.contact
    if not contact or not contact.phone_number:
        await update.message.reply_text("❌ Please share a valid contact.")
        return ConversationHandler.END

    user_info = verify_phone(contact.phone_number)
    if not user_info:
        await update.message.reply_text("❌ Unauthorized number. Access denied.", reply_markup=ReplyKeyboardRemove())
        return ConversationHandler.END

    context.user_data.update({
        "role": user_info.get('role', 'user'),
        "name": user_info.get('name', 'User'),
        "phone": contact.phone_number
    })
    await update.message.reply_text(
        f"✅ Verified as {user_info.get('role', 'user').capitalize()}. Ask your questions.",
        reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global df_tt_global

    if "role" not in context.user_data:
        await update.message.reply_text("❌ Verify first using /start")
        return

    query = update.message.text.strip().lower()
    if not query:
        await update.message.reply_text("❌ Please enter a valid question.")
        return

    timetable_keywords = ["show timetable", "display timetable", "full timetable", "weekly timetable", "view timetable"]
    day_keywords = ["monday", "tuesday", "wednesday", "thursday", "friday"]

    if any(keyword in query for keyword in timetable_keywords):
        if df_tt_global is not None and not df_tt_global.empty:
            table_str = timetable_to_markdown_table(df_tt_global)
            await update.message.reply_text(
                f"📅 **Weekly Timetable:**\n\n<pre>\n{table_str}\n</pre>",
                parse_mode=ParseMode.HTML
            )
        else:
            await update.message.reply_text("🕒 Timetable data is currently unavailable. Please try again later.")
        return
    elif any(day in query for day in day_keywords):
        requested_day = ""
        for day in day_keywords:
            if day in query:
                requested_day = day.capitalize()
                break

        if df_tt_global is not None and not df_tt_global.empty and requested_day in df_tt_global.columns:
            day_schedule = df_tt_global[['Time', requested_day]].dropna(subset=[requested_day])
            if not day_schedule.empty:
                table_str = timetable_to_markdown_table(day_schedule)
                await update.message.reply_text(
                    f"🗓️ **{requested_day} Schedule:**\n\n<pre>\n{table_str}\n</pre>",
                    parse_mode=ParseMode.HTML
                )
            else:
                await update.message.reply_text(f"No classes scheduled for {requested_day}.")
        elif df_tt_global is None or df_tt_global.empty:
            await update.message.reply_text("🕒 Timetable data is currently unavailable. Please try again later.")
        else:
            await update.message.reply_text(f"❌ Could not find schedule for {requested_day}.")
        return

    try:
        result = qa_chain({"query": query})
        await update.message.reply_text(f"📖 Answer:\n{result.get('result', 'No answer found.')}")
    except Exception as e:
        logger.error(f"QA error: {e}")
        await update.message.reply_text("❌ Error processing your question.")

# === Main ===
def main():
    try:
        application = Application.builder().token(BOT_TOKEN).build()

        conv_handler = ConversationHandler(
            entry_points=[CommandHandler("start", start)],
            states={PHONE: [MessageHandler(filters.CONTACT, handle_phone)]},
            fallbacks=[],
        )

        application.add_handler(conv_handler)
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

        logger.info("Starting bot...")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Fatal error: {e}")

if __name__ == "__main__":
    main()


# To run the bot, ensure you have the required environment variables set in a .env file.
    # Uncomment the following line to run the bot
