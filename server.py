# Launch with

# gunicorn -D --threads 4 -b 0.0.0.0:5000
# --access-logfile server.log --timeout 60
# server:app glove.6B.300d.txt bbc


from flask import Flask, render_template
from doc2vec import *
import sys

app = Flask(__name__)

# initialization
i = sys.argv.index('server:app')
glove_filename = sys.argv[i + 1]
articles_dirname = sys.argv[i + 2]
n = 5

gloves = load_glove(glove_filename)
articles_list = load_articles(articles_dirname, gloves)


@app.route("/")
def articles():
    """Show a list of article titles"""
    disp_articles = [('/article/' + a[0], a[1]) for a in articles_list]
    return render_template('articles.html', articles=disp_articles)


@app.route("/article/<topic>/<filename>")
def article(topic, filename):
    """
    Show an article with relative path filename.
    Assumes the BBC structure of topic/filename.txt
    so our URLs follow that.
    """

    for article in articles_list:
        if article[0] == topic + '/' + filename:
            reco_articles = recommended(article, articles_list, n)
            reco_articles = [('/article/' + a[1][0], a[1][1]) for a in reco_articles]
            return render_template('article.html',
                                   title=article[1],
                                   body=''.join(article[2]),
                                   articles=reco_articles)

    return "Article not found !!!"
