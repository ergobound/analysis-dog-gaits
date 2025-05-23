from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    error
)
from telegram.ext import (
    CallbackQueryHandler,
    ConversationHandler,
    InvalidCallbackData,
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters,
)
from tocluster import *

from functools import wraps
import json
import logging, sys
import html
import traceback
from telegram.constants import ParseMode
import datetime
import os
# logger = logging.getLogger(__name__)
# from dotenv import load_dotenv
# load_dotenv()

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

from warnings import filterwarnings
from telegram.warnings import PTBUserWarning
filterwarnings(action="ignore", message=r".*CallbackQueryHandler", category=PTBUserWarning)

# os.environ["TOKEN"]

from constants import *

START, STARTCHAT, CANCEL, WAIT_VIDEO, WAIT_PROMPT, WAIT_DOGINFO = range(0, 6)

os.makedirs("sessions", exist_ok=True)

# Обработчик ошибок
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)
    update_str = update.to_dict() if isinstance(update, Update) else str(update)
    message = (
        "An error occurred while processing 'update': \n"
        f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
        "</pre>\n\n"
        f"<pre>context.chat_data = {html.escape(str(context.chat_data))}</pre>\n\n"
        f"<pre>context.user_data = {html.escape(str(context.user_data))}</pre>\n\n"
        f"<pre>context.user_data = {html.escape(str(context.bot_data))}</pre>\n\n"
        f"<pre>{html.escape(tb_string)}</pre>"
    )

    await context.bot.send_message(
        chat_id=DEVELOPER_CHAT_ID, text=message, parse_mode=ParseMode.HTML
    )

# Функция доступа. Вход разрешен только ADMINS
def restricted(func):
    @wraps(func)
    async def wrapped(update: Update, context, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ADMINS:
            text = "Sorry, access to the bot has been denied."
            await update.effective_message.reply_text(text=text)
            logging.info(f"Unauthorized access is denied for {user_id}.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

@restricted # включение обработки доступа для start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Текст:
    text = "An AI assistant based on a multimodal visual-linguistic model capable of analyzing a dog's gait and detecting musculoskeletal diseases."
    # Кнопка. 
    keyboard = [[InlineKeyboardButton(text="Start", callback_data=str(STARTCHAT))]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query: # при нажатии на кнопку back to start: 
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text=text, reply_markup=reply_markup)
    else: # при вводе команды /start:
        await update.message.reply_text(text=text, reply_markup=reply_markup)

async def start_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = "Upload a video of the dog."
    keyboard = [[InlineKeyboardButton(text="Cancel", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.answer()
    await update.effective_message.edit_text(text=text, reply_markup=reply_markup)
    return WAIT_VIDEO

async def prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # user_data это всегда папка в которую мы сохраняем данные пользователя id
    user_data = context.user_data
    # проверяем gif или video
    file = update.message.animation or update.message.video
    # получаем id пользователя который отправил видео или gif
    user_id = update.effective_user.id
    ###
    keyboard = [[InlineKeyboardButton(text="Cancel", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    # получаем время now 
    timenow = datetime.datetime.now()
    timenow = datetime.datetime.strftime(timenow, "%y%m%d-%H%m%S")
    ###
    user_data["session"] = timenow
    # создать папку sessions/user_data["session"] если ее нет
    os.makedirs("sessions/" + user_data["session"], exist_ok=True)
    # получаем файл который был отправлен
    try: 
        getfile = await file.get_file()
    except error.BadRequest as err:
        if "File is too big" in err:
            text = "The file is too big, try another one."
        else:
            text = "Problems with load video."
        await update.effective_message.reply_text(
                        text=text,
                        reply_markup=reply_markup)
        return WAIT_VIDEO
    # Сохраняем файл
    file_name = file.file_name
    file_path = rf'sessions/{user_data["session"]}/{file_name}'
    await getfile.download_to_drive(file_path)
    # Сохраняем файл в папку пользователя
    user_data['video_path'] = file_path
    #
    text = "Send Prompt."
    await update.effective_message.reply_text(text=text, reply_markup=reply_markup)
    return WAIT_PROMPT

async def dog_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message.text
    context.user_data["prompt"] = message     # Сохраняем промпт в папку пользователя
    text = "Send a description of the dog, such as:\nBreed, age, observation of disease progression"
    keyboard = [[InlineKeyboardButton(text="Cancel", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_message.reply_text(text=text, reply_markup=reply_markup)
    return WAIT_DOGINFO

async def final(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_data = context.user_data
    # Сохранение введенных данных в перменные prompt, text, video_path, session, data_path, user_data['user_id']
    prompt = user_data["prompt"]
    text = user_data["text"] = update.effective_message.text
    video_path = user_data["video_path"]
    session = user_data["session"]
    data_path = user_data["data_path"] = f"sessions/{session}/data.json"
    user_data['user_id'] = update.effective_user.id
    # Сохраняем данные user_data в файл data.json
    with open(data_path, 'w', encoding='utf-8') as file:
        data = json.dumps(user_data, ensure_ascii=False)
        file.write(data)
    # Отправляем сообщение пользователю:
    text = f"It may take 5 to 30 minutes for the results of request #{session} to arrive" 
    await update.effective_message.reply_text(text=text)
    # Запускаем моментально функция waiting_process
    context.job_queue.run_once(waiting_process, 0, user_id=update.effective_user.id)
    # Выход из цепочки ConversationHandelr:
    return ConversationHandler.END

async def waiting_process(context: ContextTypes.DEFAULT_TYPE) -> None:
    print('waiting process')
    user_id = context.user_data['user_id']
    session = context.user_data['session']
    # Запускаем функцию process, которая формирует задачу и отправляет ее в кластер
    result = await process(context.user_data)

    text = f"Response to request #{session}:\n{result}"
    if result:
        await context.bot.send_message(chat_id=user_id, text=text)
    else:
        text = f"An error occurred on request #{session}!"
        await context.bot.send_message(chat_id=user_id, text=text)

async def handle_invalid_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    text = "Sorry, the button is out of date 😕 Please re-enter the /start command"
    await update.effective_message.edit_text(
        text=text
    )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    keyboard = [[InlineKeyboardButton(text="Back to start", callback_data=str(START))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text('Cancel.', reply_markup=reply_markup)
    return ConversationHandler.END

def main() -> None:
    # Build app
    app = ApplicationBuilder().token(TOKEN).arbitrary_callback_data(True).build()
    conv_chat = ConversationHandler(
        # при нажатии на кнопку start мы входим в цепочку ConversationHandler
        entry_points=[CallbackQueryHandler(start_chat, pattern=f"^{STARTCHAT}$")],
        states={
            WAIT_VIDEO: [MessageHandler(filters.VIDEO | filters.ANIMATION, prompt)],
            WAIT_PROMPT: [MessageHandler(filters.TEXT & ~filters.COMMAND, dog_info)],
            WAIT_DOGINFO: [MessageHandler(filters.TEXT & ~filters.COMMAND, final)],
        },
        # fallbacks - выйти из цепочки:
        fallbacks=[CallbackQueryHandler(cancel, pattern=f"^{CANCEL}$"),
                   CommandHandler('start', start)]
    )
    # Регистрация функции error_handler, которая запускается при ошибке
    app.add_error_handler(error_handler)
    # Регистрируем conv_chat (ConversationHandler) - та самая цепочка
    app.add_handler(conv_chat)
    # message command /start
    app.add_handler(CommandHandler('start', start))
    # Регистрируем кнопку Back to start
    app.add_handler(CallbackQueryHandler(start, pattern=f"^{START}$"))
    # Если после перезапуска нажать на кнопку, то выйдет ошибка, за это отвечает следующее:
    app.add_handler(
        CallbackQueryHandler(handle_invalid_button, pattern=InvalidCallbackData)
    )
    # Запуск бота
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()