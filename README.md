# 📰 News Analytics Bot
[![readme.jpg](https://anopic.ag/YUuttmGnZB1PeYfLv4El5IZqA2oWDrn4aMkVwrnF.jpg)](https://anopic.ag/YUuttmGnZB1PeYfLv4El5IZqA2oWDrn4aMkVwrnF.jpg)

## Project Description
In this project we are implementing a telegram bot [@news_class_bot](https://t.me/news_class_bot) that provides analytics of a news resource using NLP.

fontanka.ru is used as a news resource

## Table of Contents

- [📰 News Analytics Bot](#-news-analytics-bot)
  - [Project Description](#project-description)
  - [Table of Contents](#table-of-contents)
  - [Files](#files)
  - [Dataset](#dataset)
  - [Parsing](#parsing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Classification problem](#classification-problem)
  - [Bot Functionality](#bot-functionality)
  - [Deployment](#deployment)
  - [Include Credits](#include-credits)
    - [Author](#author)
  - [License](#license)

## Files
- [fontanka_parsing.ipynb](https://github.com/moxeeem/NewsBot/blob/main/fontanka_parsing.ipynb) : Jupyter Notebook with Exploratory Data Analysis and parsing
- [classification.ipynb](https://github.com/moxeeem/NewsBot/blob/main/classification.ipynb) : Jupyter Notebook with ML pipelines
- [bot/svc_mv.pkl](https://github.com/moxeeem/NewsBot/blob/main/bot/xgboost_mv.pkl) : Cloudpickle file with SVM classifier and Word2Vec
- [bot/parser.ipynb](https://github.com/moxeeem/NewsBot/blob/main/bot/parser.ipynb) : An improved parser 
- [bot/text_prep.py](https://github.com/moxeeem/NewsBot/blob/main/bot/text_prep.py) : Text preprocessing function
- [bot/requirements.txt](https://github.com/moxeeem/NewsBot/blob/main/bot/requirements.txt) : Project's requirements
- [bot/keyboards.py](https://github.com/moxeeem/NewsBot/blob/main/bot/keyboards.py) : Bot's keyboards
- [bot/main.py](https://github.com/moxeeem/NewsBot/blob/main/bot/main.py) : A file with bot realisation



## Dataset
The dataset used to build the models was created by parsing news articles from the website fontanka.ru. The news is divided into topics that make up the classes. The classes are absolutely balanced.

The dataset uses news posts mainly from 2023, but also contains records from 2018-2022. In total it contains 26719 records.

**Target Variable**
- `topic` (categorical, string) : the topic of the news post

**All features**

| Feature             | Description                          | Type     |
|---------------------|--------------------------------------|----------|
| `date`              | The date of the news post            | Datetime |
| `title`             | The title of the news post           |  String  |
| `topic`             | The topic of the news post           |  String  |
| `url`               | URL of the news post                 |  String  |
| `time`              | The time of the news post            | Datetime |
| `comm_num`          | Number of comments                   | Integer  |
| `author`            | Author of the news post              |  String  |
| `views`             | Number of views                      | Integer  |
| `content`           | Content of the news post             |  String  |
| `year`              | The year of the news post            | Datetime |
| `month`             | The month of the news post           | Datetime |
| `weekday`           | The weekday of the news post         | Datetime |
| `log_views`         | Log of the number of views           |  Float   |
| `len_title`         | Length of the title                  | Integer  |
| `len_content`       | Length of the content                | Integer  |
| `lifetime`          | Lifetime of the news post            |  Float   |
| `views_by_minutes`  | Views per minute                     |  Float   |
| `log_comm`          | Log of the number of comments        |  Float   |


## Parsing
During the initial data collection, a large number of problems were identified when using the parser from `fontanka_parsing.ipynb`.
Therefore, it was decided to rewrite the parser, extending its functionality and improving its logic. The corrected parser is `parser.ipynb`.

To implement the parser we used regular expressions, as well as the lack of direct reference to html tags (tag names on the
fontanka.ru site are often changed).

Logging with the help of loguru library was also implemented.

## Exploratory Data Analysis
In this project, we analyze our data and perform EDA to understand its main characteristics before building our model. We found that articles are evenly distributed across topics, with most news from 2023. The publication dates show peaks in August and September, with fewer articles in winter. Weekdays have more news compared to weekends.

We also examined the average lengths of titles and articles. Keywords were also explored. Keywords are a key consideration, as articles from different topics share dominant keywords, affecting the model.

The number of views is important but varies with article age. We introduced the average growth rate of views but found it doesn't follow a lognormal distribution. The number of comments also doesn't have a lognormal distribution and resembles an exponential one.

## Classification problem

In this project we did text preprocessing using the Natasha library. More specifically, we did the following steps:

- lowering
- tokenize
- lemmatize
- remove symbols
- remove stop-words

Also we've trained Word2Vec for our news data and got an adequate result.

The project uses SVM with MeanEmbeddingVectorizer to classify texts. Accuracy of such a model is 0.77 (we also explained why we rely on this metric).

Despite the best performance of the XGBoost with MeanEmbeddingVectorizer model (accuracy = 0.78), we decided not to sacrifice memory and speed and chose SVM.
    
If this classifier makes a mistake, it will most likely confuse the class `Общество` with `Город` or `Политика`. This is not a big deal, because these topics are quite related.


## Bot Functionality
The Telegram bot offers the user three main functionalities:

- **Predictions**
  The bot uses a previously trained classifier to predict the relationship of a particular day's news to a particular date
- **Visualizations**
  The bot offers the ability to plot pie charts and word clouds for news for a particular day.
- **News by topic**
  The bot allows you to determine the most relevant news for a particular day. This is done by comparing user topic vectors and news vectors by cosine measure.


## Deployment
The final part of the project was to move it to a VPS. This was successfully implemented and now the bot is available here - [@news_class_bot](https://t.me/news_class_bot).


## Include Credits

### Author
- Maxim Ivanov - [GitHub](https://github.com/moxeeem), [Telegram](https://t.me/fwznn_ql1d_8)

This project was completed as part of the ["Основы нейронных сетей и NLP"](https://stepik.org/course/180984) course offered by [AI Education](https://stepik.org/users/628121134).

## License
This project is licensed under the MIT license. For more information, see the [LICENSE](/LICENSE) file.
