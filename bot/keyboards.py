from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    KeyboardButtonPollType
)
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from aiogram.filters.callback_data import CallbackData


first_choice = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Predictions"),
            KeyboardButton(text="Visualizations")
        ],
        [
            KeyboardButton(text="News on your topic"),
            KeyboardButton(text="Back")
        ]
    ],
    resize_keyboard=True,
    one_time_keyboard=True
)

back_button = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Back")
        ]
    ],
    resize_keyboard=True,
    one_time_keyboard=True
)

menu_button = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="To menu")
        ]
    ],
    resize_keyboard=True,
    one_time_keyboard=True
)

change_button = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Change selection"),
            KeyboardButton(text="Back")
        ]
    ],
    resize_keyboard=True,
    one_time_keyboard=True
)


LEXICON = {'backward': '<<',
           'forward': '>>'}


def create_pagination_keyboard(*buttons: str) -> InlineKeyboardMarkup:
    kb_builder = InlineKeyboardBuilder()

    kb_builder.row(*[InlineKeyboardButton(
        text=LEXICON[button] if button in LEXICON else button,
        callback_data=button) for button in buttons]
    )

    return kb_builder.as_markup()


visualizations_choice = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Pie chart"),
            KeyboardButton(text="Word cloud"),
            KeyboardButton(text="To menu")
        ]
    ],
    resize_keyboard=True,
    one_time_keyboard=True
)
