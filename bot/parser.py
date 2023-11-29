from bs4 import BeautifulSoup
import re
import requests
from loguru import logger
import threading
import pandas as pd

from fake_useragent import UserAgent
from pandas import DataFrame

topic_mapping = {
    'Особое мнение': 'Общество',
    'Технологии': 'Общество',
    'Строительство': 'Город',
    'Недвижимость': 'Город',
    'ЖКХ': 'Город',
    'Авто': 'Город',
    'Финляндия': 'Политика',
    'Власть': 'Политика',
    'Открытое письмо': 'Политика',
    'Бизнес': 'Финансы',
    'Работа': 'Финансы',
    'Новости компаний': 'Финансы',
    'Бизнес-трибуна': 'Финансы',
    'Доктор Питер': 'Здоровье',
    'Туризм': 'Здоровье',
    'Спорт': 'Здоровье',
    'Афиша Plus': 'Образ жизни',
    'Доброе дело': 'Образ жизни'
}

LOGGING = 'logs.log'
FORMAT = "{time} {level} {message}"


def get_page(p) -> DataFrame:
    """A function to retrieve data from a web page."""

    logger.add(LOGGING)
    logger.add(LOGGING, format=FORMAT)

    # Generating the url for the request
    url = f'https://www.fontanka.ru/{p}/all.html'

    # Receiving a response from the server
    response = requests.get(url, headers={'User-Agent': UserAgent().chrome})

    # Convert the response to a BeautifulSoup object
    tree = BeautifulSoup(response.content, 'html.parser')

    # Find all the news for the day (tag - a string of 5 characters,
    # for example: KXak5)
    news = tree.find_all('li', {'class': re.compile(r'^\w{5}$')})

    info = []

    logger.debug('Parsing of pages started')

    # Going around every news item
    for post in news:
        event = threading.Event()
        # Finding the news title
        try:
            title = post.select_one('li div div a').text
        except Exception as e:
            logger.warning(
                f"Failed to get the news title. Publication date: {p}. "
                f"Exception: {e}")
            pass

        # Finding the news topic
        try:
            topic = post.find('a').get('title')
        except Exception as e:
            logger.warning(
                f"Failed to get the news topic. Publication date: {p}. "
                f"Exception: {e}")
            pass

        # Finding the news link
        try:
            link = post.select_one('li div div a').get('href')
        except Exception as e:
            logger.warning(
                f"Failed to get the news link. Publication date: {p}. "
                f"Exception: {e}")
            pass

        # Finding the news time of publication
        try:
            time = post.select_one('li div time span').text
        except Exception as e:
            logger.warning(
                f"Failed to get the news publication time. "
                f"Publication date: {p}. Exception: {e}")
            pass

        # Finding the news number of comments
        try:
            comm = post.select_one('li div div a span').text
        except Exception as e:
            comm = 0
            logger.warning(
                f"Failed to get the news comments number. "
                f"Publication date: {p}. Exception: {e}")
            pass

        # Creating the url with our link
        if "https://doctorpiter.ru" in link:
            urli = link
        elif "https://www.fontanka.ru/longreads/" in link:
            urli = link
        elif "https://www.fontanka.ru" in link:
            urli = link
        else:
            urli = "https://www.fontanka.ru" + link

        # Going inside each news item we found
        response_inner = requests.get(urli, headers={'User-Agent': UserAgent().firefox})
        tree_inner = BeautifulSoup(response_inner.content, 'html.parser')

        # If this is affiliate news from doctorpiter.ru website
        if "https://doctorpiter.ru" in urli:
            views = 9999  # This site does not have a view counter

            try:
                content = tree_inner.find('section',
                                          {'class': "ds-article-content"}).text
            except Exception as e:
                logger.warning(
                    f"Failed to get the content from doctorpiter. "
                    f"Publication date: {p}. Exception: {e}")
                pass

        else:
            # Such pages are promotional and have no clear markup
            if "https://www.fontanka.ru/longreads/" in urli:
                logger.warning(f"It's the longread. Publication date: {p}.")
                pass

            else:
                # If this is news from fontanka.ru
                if "https://www.fontanka.ru" in urli:
                    try:
                        content = tree_inner.select_one(
                            'body div div div div div section div article div section div').text
                    except Exception as e:
                        logger.warning(
                            f"Failed to get the content from fontanka. "
                            f"Publication date: {p}. Exception: {e}")
                        pass

                    try:
                        views = tree_inner.find('div', {
                            'class': re.compile(r'^[A]\dg[a-z]$')}).text
                    except Exception as e:
                        logger.warning(
                            f"Failed to get the viewers number from fontanka. "
                            f"Publication date: {p}. Exception: {e}")
                        pass

                else:
                    logger.error(
                        f"Unknown source of the news post. "
                        f"Publication date: {p}.")

        # Class synthesis
        mapped_topic = topic_mapping.get(topic, topic)

        logger.debug('Parsing of page finished')

        # Creating a table row with the received data
        row = {
            'date': p,
            'title': title,
            'topic': mapped_topic,
            'url': urli,
            'time': time,
            'comm_num': comm,
            'views': views,
            'content': content
        }

        if row['date'] != 'date' and \
                row['title'] != 'title' and \
                row['topic'] != 'topic' and \
                row['url'] != 'url' and \
                row['time'] != 'time' and \
                row['comm_num'] != 'comm_num' and \
                row['views'] != 'views' and \
                row['content'] != 'content':
            info.append(row)

    info = pd.DataFrame(info)
    info = info.drop_duplicates()

    return info
