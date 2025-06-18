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

START, STARTCHAT, CANCEL, WAIT_VIDEO, WAIT_AGE, WAIT_BREAD, WAIT_NOTE = range(0, 7)
PROMPT = "Ты умный помощник."
MESSAGE = """Опишите походку собаки по видеозаписи, обращая внимание на:
1. Симметрия движений конечностей.
2. Нагрузка на передние/задние лапы.
3. Признаки хромоты или атрофии мышц.
4. Характеристики шага (длина, ритм, положение хвоста).
Найдите проблемы с опорно-двигательным аппаратом собаки.

Попробуй найти название заболевания. Если ты не уверен в определении диагноза, то не называй точный диагноз."""

os.makedirs("sessions", exist_ok=True)

# Обработчик ошибок
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    tb_list = traceback.format_exception(None, context.error, context.error.__traceback__)
    tb_string = "".join(tb_list)
    update_str = update.to_dict() if isinstance(update, Update) else str(update)
    message = (
        "Произошла ошибка при обработке 'update': \n"
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
            text = "Извините, доступ к боту закрыт."
            await update.effective_message.reply_text(text=text)
            logging.info(f"Несанкционированный доступ запрещен для {user_id}.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

@restricted # включение обработки доступа для start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # text = "ИИ-помощник на основе мультимодальной визуально-лингвистической модели, способный анализировать походку собаки и выявлять заболевания опорно-двигательного аппарата."
    text = "Привет! Я виртуальный ветеринарный помощник. Давайте вместе позаботимся о вашей собаке и попробуем определить есть ли у нее заболевания опорно-двигательного аппарата."
    keyboard = [[InlineKeyboardButton(text="Старт", callback_data=str(STARTCHAT))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    video = 'only-walk.mp4'
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text=text, reply_markup=reply_markup)
    else:
        await update.message.reply_video(video=video,
                                         caption=text,
                                         reply_markup=reply_markup)

async def start_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = """Снимите короткое видео походки вашей собаки и загрузите следующим сообщением.
Чтобы обеспечить точность анализа, следуйте следующим рекомендациям:
1. Угол камеры:
    Записывайте видео с бокового ракурса, чтобы в кадр постоянно попадало все тело собаки (от головы до хвоста, все лапы должны быть видны).
2. Траектория движения:
    Постарайтесь, чтобы собака прошлась по прямой линии и по ровной поверхности (например, по тротуару или коридору).
    Избегайте поворотов, препятствий или неровной поверхности.
3. Расстояние и кадрирование:
    Стойте на таком расстоянии, чтобы вся собака четко помещалась в кадр (избегайте крупных планов).
    Держите камеру ровно.
4. Освещение:
    Снимайте при дневном свете или при хорошем освещении, чтобы движения и конечности собаки были хорошо видны.
5. Продолжительность:
    Не снимайте длинные видео, вполне хватит 5-20 секунд."""
    keyboard = [[InlineKeyboardButton(text="Отмена", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.answer()
    await update.effective_user.send_message(text=text, reply_markup=reply_markup)
    return WAIT_VIDEO

async def age(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_data = context.user_data
    file = update.message.animation or update.message.video
    user_id = update.effective_user.id
    keyboard = [[InlineKeyboardButton(text="Отмена", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    timenow = datetime.datetime.now()
    timenow = datetime.datetime.strftime(timenow, "%y%m%d-%H%M%S")
    user_data["session"] = timenow # название сессии
    # создать папку sessions/session если ее нет
    os.makedirs("sessions/" + user_data["session"], exist_ok=True)
    try: 
        getfile = await file.get_file()
    except error.BadRequest as err:
        if "File is too big" in err:
            text = "Файл слишком большой, попробуйте другой."
        else:
            text = "Проблемы с загрузкой видео, измените видео и попробуйте снова."
        await update.effective_message.reply_text(
                        text=text,
                        reply_markup=reply_markup)
        return WAIT_VIDEO
    file_name = file.file_name
    file_path = rf'sessions/{user_data["session"]}/{file_name}'
    await getfile.download_to_drive(file_path)
    # Сохраняем файл в папку пользователя
    user_data['video_path'] = file_path
    text = "Сколько лет вашей собаке?\nЕсли вы затрудняетесь с ответом, напишите - неизвестно."
    await update.effective_message.reply_text(text=text, reply_markup=reply_markup)
    return WAIT_AGE

async def bread(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message.text
    context.user_data["age"] = message
    text = "Какая порода у вашей собаки? (Если смешанная, опишите основные черты, размер и т.п.)\nЕсли вы затрудняетесь с ответом, напишите - неизвестно."
    keyboard = [[InlineKeyboardButton(text="Отмена", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_message.reply_text(text=text, reply_markup=reply_markup)
    return WAIT_BREAD

async def note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message.text
    context.user_data["bread"] = message
    text = "Замечали ли вы какие-либо признаки дискомфорта - хромоту, скованность, усталость или избегание одной ноги? Если да, не могли бы вы рассказать об этом подробнее? (Какая нога, когда это началось, как это влияет на активность и т.п.)\nЕсли вы затрудняетесь с ответом, напишите - неизвестно."
    keyboard = [[InlineKeyboardButton(text="Отмена", callback_data=str(CANCEL))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_message.reply_text(text=text, reply_markup=reply_markup)
    return WAIT_NOTE

async def final(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_data = context.user_data
    user_data["prompt"] = PROMPT # теперь промпт стабильно всегда одинаков и един
    user_data["text"] = f"""Информация о собаке на видео:
Возраст: {user_data.get("age")}
Порода или иные черты: {user_data.get("bread")}
Наблюдения за собакой: {update.effective_message.text}

{MESSAGE}
"""
    session = user_data["session"]
    data_path = user_data["data_path"] = f"sessions/{session}/data.json"
    user_data['user_id'] = update.effective_user.id
    # Сохраняем данные user_data в файл data.json
    # (в дальнейшем data.json данные считываются в analysis.py)
    with open(data_path, 'w', encoding='utf-8') as file:
        data = json.dumps(user_data, ensure_ascii=False)
        file.write(data)
    text = f"Получение результатов запроса #{session} может занять от 5 до 30 минут." 
    await update.effective_message.reply_text(text=text)
    # Моментально запускаем функция waiting_process
    context.job_queue.run_once(waiting_process, 0, user_id=update.effective_user.id)
    return ConversationHandler.END

async def waiting_process(context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = context.user_data['user_id']
    session = context.user_data['session']
    # Запускаем функцию process, которая формирует задачу и отправляет ее в кластер
    result = await process(context.user_data)
    # в переменной result хранится конечный результат анализа, если не было никаких ошибок
    text = f"""Ответ на запрос #{session}:

{result}

⚠️ Пожалуйста, обратите внимание: я являюсь виртуальным помощником, а не лицензированным ветеринаром.
Предоставленная мной информация носит исключительно информационный характер и не заменяет профессиональной ветеринарной диагностики или лечения.
Если ваша собака испытывает боль или симптомы ухудшаются, пожалуйста, как можно скорее обратитесь к лицензированному ветеринару."""
    if result:
        await context.bot.send_message(chat_id=user_id, text=text)
    else:
        text = f"Произошла ошибка при запросе #{session}!"
        await context.bot.send_message(chat_id=user_id, text=text)

async def handle_invalid_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.callback_query.answer()
    event = update.effective_message.to_dict()
    photo = event.get("photo", None)
    video = event.get("video", None)
    text = "Извините, кнопка устарела 😕\nПожалуйста, введите повторно команду /start"
    if photo or video:
        await update.effective_user.send_message(text=text)
    else:
        await update.effective_message.edit_text(text=text)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    keyboard = [[InlineKeyboardButton(text="Вернуться в начало", callback_data=str(START))]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.answer()
    await query.edit_message_text('Отмена.', reply_markup=reply_markup)
    return ConversationHandler.END

def main() -> None:
    # Build app
    app = ApplicationBuilder().token(TOKEN).arbitrary_callback_data(True).build()
    conv_chat = ConversationHandler(
        entry_points=[CallbackQueryHandler(start_chat, pattern=f"^{STARTCHAT}$")],
        states={
            WAIT_VIDEO: [MessageHandler(filters.VIDEO | filters.ANIMATION, age)],
            WAIT_AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, bread)],
            WAIT_BREAD: [MessageHandler(filters.TEXT & ~filters.COMMAND, note)],
            WAIT_NOTE: [MessageHandler(filters.TEXT & ~filters.COMMAND, final)],
        },
        fallbacks=[CallbackQueryHandler(cancel, pattern=f"^{CANCEL}$"),
                   CommandHandler('start', start)]
    )
    app.add_error_handler(error_handler)
    app.add_handler(conv_chat)
    app.add_handler(CommandHandler('start', start))
    app.add_handler(CallbackQueryHandler(start, pattern=f"^{START}$"))
    app.add_handler(
        CallbackQueryHandler(handle_invalid_button, pattern=InvalidCallbackData)
    )
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()