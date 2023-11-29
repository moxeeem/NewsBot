import asyncio
import logging
import os
import io

import cloudpickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    CallbackQuery, FSInputFile, InlineKeyboardButton, InlineKeyboardMarkup,
    InputFile, ReplyKeyboardRemove
)
from aiogram.utils import markdown
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from wordcloud import WordCloud

from config import BOT_TOKEN
from keyboards import (
    create_pagination_keyboard, first_choice, visualizations_choice,
    back_button, change_button, menu_button
)
from parser import get_page
from text_prep import text_prep


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
RND_STATE = 42


class Form(StatesGroup):
    """
    A class to define the states of the form
    """
    df = State()
    pipe = State()
    X = State()
    predictions = State()
    page = State()
    message_chunks = State()


@dp.message(Command("help"))
async def get_help(message: types.Message) -> None:
    """
    Handles the 'help' command from the user.

    Parameters:
        message (types.Message): The message object from the user

    Returns:
        None
    """
    # Send a response to the user
    await message.answer(text="I am a bot that can provide you with news "
                              "analytics. \n\nStart working with me by "
                              "writing the /start command =)")


@dp.message(CommandStart())
async def handle_start(message: types.Message) -> None:
    """
    Handle the 'start' command from the user.

    Parameters:
        message (types.Message): The message object from the user

    Returns:
        None
    """
    # Send a response to the user
    await message.answer(
        text=f"Hello, "
             f"{markdown.hbold(message.from_user.full_name)}!"
             f"\nEnter the date in {markdown.hbold('yyyy/mm/dd')} format:",
        parse_mode=ParseMode.HTML
    )


@dp.message(lambda message: message.text.startswith('20'))
async def process_date(message: types.Message, state: FSMContext) -> None:
    """
    Process the user's input date and perform operations accordingly.

    Parameters:
        message (types.Message): The message object from the user
        state (FSMContext): The current state

    Returns:
        None
    """
    await message.reply("Looking for news...")

    try:
        date = message.text
        df = get_page(date)
    except:
        await message.answer("Make sure you enter the date correctly")

    if df.empty:
        await message.reply("There is no news on the date listed")
        return

    # Update the state with the dataframe
    await state.update_data(df=df)

    # Preprocessing
    df['content_clean'] = df.content.apply(text_prep)
    X = df['content_clean'].str.split()
    await state.update_data(X=X)

    # Load the classifier pipeline
    with open('svc_mv.pkl', 'rb') as model_file:
        pipe = cloudpickle.load(model_file)

    # Update the state with the pipeline
    await state.set_state(Form.pipe)
    await state.update_data(pipe=pipe)

    # Make predictions using the pipeline
    predictions = pipe.predict_proba(X)
    await state.update_data(predictions=predictions)

    await message.answer("Choose what you want:", reply_markup=first_choice)


@dp.message(F.text.lower() == "back")
async def to_start(message: types.Message, state: FSMContext) -> None:
    """
    Handler function for when the user sends the message "back".
    Clears the current state and goes back to the start.

    Parameters:
        message (types.Message): The message object from the user
        state (FSMContext): The current state

    Returns:
        None
    """
    await message.reply("Going back...", reply_markup=ReplyKeyboardRemove())

    # Clear the current state
    await state.clear()

    await handle_start(message)


@dp.message(F.text.lower() == "predictions")
async def predictions(message: types.Message, state: FSMContext) -> None:
    """
    Handler function for the "predictions" command.

    Parameters:
        message (types.Message): The message object from the user
        state (FSMContext): The current state

    Returns:
        None
    """

    await message.reply("Making predictions...", reply_markup=menu_button)

    await process_beginning_command(message, state)


@dp.message(Command(commands='beginning'))
async def process_beginning_command(message: types.Message,
                                    state: FSMContext) -> None:
    """
    Handler function for the "beginning" command. Runs the pagination with
    predictions

    Parameters:
        message (types.Message): The message object from the user
        state (FSMContext): The current state

    Returns:
        None
    """

    # Get data from state
    data = await state.get_data()
    pipe = data['pipe']
    df = data['df']
    predictions = data['predictions']

    # Set initial page state
    page = 0
    await state.set_state(Form.page)
    await state.update_data(page=page)

    # Generate message chunks
    message_chunks = []
    for index, (title, probas, url) in enumerate(zip(df['title'], predictions,
                                                     df['url'])):
        chunk = f"\n{index + 1}. {title}\n"
        chunk += f"URL: {url}\n\n"
        chunk += "Predictions:\n"
        top_predictions = sorted(zip(pipe.classes_, probas),
                                 key=lambda x: x[1], reverse=True)[:3]
        for topic, proba in top_predictions:
            chunk += f"- {topic}: {proba*100:.2f}%\n"
        chunk += "\n"
        message_chunks.append(chunk)

    # Update state with message chunks
    await state.set_state(Form.message_chunks)
    await state.update_data(message_chunks=message_chunks)

    # Get text for the current page
    text = message_chunks[page]

    # Send message with pagination keyboard
    await message.answer(
        text=text,
        reply_markup=create_pagination_keyboard(
            f'{page+1}/{len(message_chunks)+1}',
            'backward',
            'forward'
        )
    )


@dp.callback_query(F.data == 'forward')
async def process_forward_press(callback: CallbackQuery,
                                state: FSMContext) -> None:
    """
    This handler is triggered when the user presses the "forward" inline button
    during interaction with the pagination keyboard

    Parameters:
        callback (CallbackQuery): The callback query object
        state (FSMContext): The current state

    Returns:
        None
    """

    # Get data from state
    data = await state.get_data()
    page = data['page']
    message_chunks = data['message_chunks']

    # Check if there is a next page
    if page < len(message_chunks):
        page += 1
        await state.update_data(page=page)
        text = message_chunks[page]

        # Edit the message with the new text and pagination keyboard
        await callback.message.edit_text(
            text=text,
            reply_markup=create_pagination_keyboard(
                f'{page + 1}/{len(message_chunks) + 1}',
                'backward',
                'forward'
            )
        )
    # Answer the callback query
    await callback.answer()


@dp.callback_query(F.data == 'backward')
async def process_backward_press(callback: CallbackQuery,
                                 state: FSMContext) -> None:
    """
    This handler is triggered when the user presses the "backward" inline
    button during the interaction with the pagination keyboard

    Parameters:
        callback (CallbackQuery): The callback query object
        state (FSMContext): The current state

    Returns:
        None
    """

    # Get the current page and message chunks from the state data
    data = await state.get_data()
    page = data['page']
    message_chunks = data['message_chunks']

    # If the current page is greater than 0, decrement the page number
    if page > 0:
        page -= 1
        await state.update_data(page=page)
        text = message_chunks[page]
        await callback.message.edit_text(
            text=text,
            reply_markup=create_pagination_keyboard(
                f'{page + 1}/{len(message_chunks) + 1}',
                'backward',
                'forward'
            )
        )

    # Answer the callback query
    await callback.answer()


@dp.message(F.text.lower() == "to menu")
async def to_menu(message: types.Message, state: FSMContext) -> None:
    """
    Handler for the "to menu" command
    Calls the process_date function and removes the custom keyboard

    Parameters:
        message (types.Message): The incoming message
        state (FSMContext): The current state

    Returns:
        None
    """

    await message.reply("Going back...", reply_markup=ReplyKeyboardRemove())
    await message.answer("Choose what you want:", reply_markup=first_choice)


@dp.message(F.text.lower() == "change selection")
async def to_menu(message: types.Message, state: FSMContext) -> None:
    """
    Handler function for the "change selection" command
    Allows to select the another visualization

    Parameters:
        message (types.Message): The Telegram message object.
        state (FSMContext): The FSM context object.

    Returns:
        None
    """

    await message.reply("Going back...", reply_markup=ReplyKeyboardRemove())
    await visualizations(message)


@dp.message(F.text.lower() == "visualizations")
async def visualizations(message: types.Message) -> None:
    """
    Responds to the message with visualizations options

    Parameters:
        message (types.Message): The Telegram message object

    Returns:
        None
    """

    await message.reply("You have chose visualizations",
                        reply_markup=ReplyKeyboardRemove())

    # Send a message with the available visualization options
    await message.answer("Choose what you want:",
                         reply_markup=visualizations_choice)


@dp.message(F.text.lower() == "pie chart")
async def piechart(message: types.Message, state: FSMContext) -> None:
    """
    Handler function for the "pie chart" command.
    Generates and sends a pie chart based on the predictions data

    Parameters:
        message (types.Message): The Telegram message object
        state (FSMContext): The current state

    Returns:
        None
    """

    # Get data from state
    data = await state.get_data()
    predictions = data['predictions']
    pipe = data['pipe']

    await message.reply("You have chose pie chart",
                        reply_markup=ReplyKeyboardRemove())

    # Prepare data for the pie chart
    labels = pipe.classes_
    values = predictions.sum(axis=0)  # Sum along rows (across classes)

    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Predictions')
    plt.axis('equal')

    # Save and send the pie chart
    plt.savefig("pie.png")
    photo = FSInputFile("pie.png")
    await bot.send_photo(chat_id=message.chat.id, photo=photo,
                         reply_markup=change_button)


@dp.message(F.text.lower() == "word cloud")
async def wordcloud(message: types.Message, state: FSMContext) -> None:
    """
    Handler function for the "word cloud" command.
    Generates and sends a word cloud based on the predictions data

    Parameters:
        message (types.Message): The Telegram message object
        state (FSMContext): The current state

    Returns:
        None
    """

    await message.reply("You have chose wordcloud",
                        reply_markup=ReplyKeyboardRemove())

    data = await state.get_data()
    X = data['X']

    text = ' '.join(' '.join(doc) for doc in X)

    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(text)
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    plt.savefig("cloud.png")
    photo = FSInputFile("cloud.png")
    await bot.send_photo(chat_id=message.chat.id, photo=photo,
                         reply_markup=change_button)


@dp.message(F.text.lower() == "news on your topic")
async def news_on_topic(message: types.Message, state: FSMContext) -> None:
    """
    Handle the command for starting the process of news on your topic

    Parameters:
        message: The incoming message object
        state: The current state

    Returns:
        None
    """

    await message.answer("You have chose news on your topic. "
                         "Please enter a list of topics separated by commas",
                         reply_markup=ReplyKeyboardRemove())


@dp.message(F.text)
async def process_topics(message: types.Message, state: FSMContext) -> None:
    """
    Process the user's input topics and performs the news which is similar to
    user's input topics with the help of cosine similarity of user's topics and
    news texts

    Parameters:
        message: The incoming message object
        state: The current state

    Returns:
        None
    """

    # Split user-provided topics
    topics = message.text.split(',')

    # Get data from state
    data = await state.get_data()
    df = data['df']  # DataFrame with all news data
    news_texts = data['X']  # Preprocessed news texts (text_prep and split)

    # Original not preprocessed titles of preprocessed news texts
    news_texts_titles = df['title']

    topics = pd.DataFrame(topics, columns=['topic'])
    topics['topic_clean'] = topics['topic'].apply(text_prep)

    # Train Word2Vec model
    model = Word2Vec(sentences=news_texts,
                     vector_size=300,
                     min_count=5,
                     window=5,
                     seed=RND_STATE)

    # Calculate vectors for user-provided topics with
    # Mean Embedding Vectorization
    user_topic_vectors = np.array([
        np.mean([model.wv.get_vector(w) for w in topic.split() if w in
                 model.wv.key_to_index] or [np.zeros(model.vector_size)],
                axis=0) for topic in topics['topic_clean']])

    # Calculate vectors for news texts with Mean Embedding Vectorization
    news_text_vectors = np.array([
        np.mean([model.wv.get_vector(w) for w in text if w in
                 model.wv.key_to_index] or [np.zeros(model.vector_size)],
                axis=0) for text in news_texts])

    # Cosine similarity between user topic vectors and news text vectors
    similarities = cosine_similarity(user_topic_vectors, news_text_vectors)

    # Generate response text
    response_text = ""
    for i, topic in enumerate(topics['topic']):
        topic_similarities = similarities[i]

        # Get indices of best matching news articles
        best_match_indices = np.argsort(topic_similarities)[-3:][::-1]

        # Add topic and best matching news articles to response text
        response_text += f"\n{topic.strip()}:\n"

        for j, index in enumerate(best_match_indices):
            similarity_percentage = topic_similarities[index] * 100
            news_title = news_texts_titles.iloc[index]
            news_url = df['url'].iloc[index]
            response_text += f"\t{j + 1}. {news_title}\n"
            response_text += f"\t\tSimilarity: {similarity_percentage:.2f}%\n"
            response_text += f"\t\tURL: {news_url}\n\n"

        # Send response text to user
        await message.reply(response_text, reply_markup=menu_button)
        response_text = ""


async def main():
    """
    Starts the polling of the bot
    """

    logging.basicConfig(level=logging.DEBUG)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
