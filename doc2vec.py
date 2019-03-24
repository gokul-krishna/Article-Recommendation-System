import os
import re
import sys
import string
import codecs
import numpy as np

table = str.maketrans({key: ' ' for key in string.digits + string.punctuation + '\t\n\r'})
# match = re.compile(string.punctuation + '0-9\\r\\t\\n]')

# From scikit learn that got words from:
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def load_glove(filename):
    """
    Read all lines from the indicated file and return
    a dictionary mapping word:vector where vectors are
    of numpy `array` type.

    GloVe file lines are of the form:

    the 0.418 0.24968 -0.41242 0.1217 ...

    So split each line on spaces into a list;
    the first element is the word and the remaining
    elements represent factor components.
    The length of the vector should not matter;
    read vectors of any length.
    """
    glove = {}
    with open(filename, 'r') as read_file:
        for line in read_file:
            key = line.split(' ')[0]
            vector = np.array(line.split(' ')[1:], dtype=np.float)
            glove[key] = vector

    return glove


def filelist(root):
    """Return a fully-qualified list of filenames
       under root directory"""
    files = []

    for f in os.listdir(root):
        if os.path.isfile(root + '/' + f):
            if f.endswith('.txt'):
                files.append(root + '/' + f)
        else:
            files += filelist(root + '/' + f)

    return files


def get_text(filename):
    """
    Load and return the text of a text file,
    assuming latin-1 encoding as that
    is what the BBC corpus uses.
    Use codecs.open() function not open().
    """
    f = codecs.open(filename, encoding='latin-1', mode='r')
    s = f.read()
    f.close()
    return s


def words(text):
    """
    Given a string, return a list of words normalized as
    follows. Split the string to make words first by
    using regex compile() function and string.punctuation
    + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long.
    Lowercase all words
    Remove English stop words
    """
    text = text.lower()
    text = text.translate(table).strip()
    terms = text.split(' ')
    terms = [t for t in terms if len(t) > 2]
    terms = [t for t in terms if t not in ENGLISH_STOP_WORDS]

    return terms


def load_articles(articles_dirname, gloves):
    """
    Load all .txt files under articles_dirname and
    return a table (list of lists/tuples)
    where each record is a list of:

      [filename, title, article-text-minus-title,
       wordvec-centroid-for-article-text]

    We use gloves parameter to compute the word
    vectors and centroid.

    The filename is stripped of the prefix of the
    articles_dirname pulled in as script parameter
    sys.argv[2]. E.g., filename will be "business/223.txt"
    """
    article_list = []
    file_names = filelist(articles_dirname)

    for fname in file_names:
        article = get_text(fname)
        filename = '/'.join(fname.rsplit('/', 2)[1:])
        title = article.split('\n')[0]
        article_body = article.split('\n')[1:]
        centroid = doc2vec(article, gloves)
        article_list += [(filename, title, article_body, centroid)]

    return article_list


def doc2vec(text, gloves):
    """
    Return the word vector centroid for the text.
    Sum the word vectors for each word and
    then divide by the number of words.
    Ignore words not in gloves.
    """
    count_words = 0
    final_vector = 0
    terms = words(text)

    for t in terms:
        if t in gloves.keys():
            count_words += 1
            final_vector += gloves[t]

    return final_vector / count_words


def distances(article, articles):
    """
    Compute the euclidean distance from article to
    every other article and return a list of
    (distance, a) tuples for all a in articles.
    The article is one of the elements (tuple)
    from the articles list.
    """
    distance_list = []
    x1 = article[3]

    for a in articles:
        dist = np.linalg.norm(x1 - a[3])
        distance_list += [(dist, a)]

    return distance_list


def recommended(article, articles, n):
    """
    Return a list of the n articles (records with
    filename, title, etc...) closest to article's
    word vector centroid. The article is one of the elements
    (tuple) from the articles list.
    """

    distance_list = distances(article, articles)
    distance_list.sort(key=lambda x: x[0])

    return distance_list[1:n + 1]


if __name__ == '__main__':
    glove_filename = sys.argv[1]
    articles_dirname = sys.argv[2]

    gloves = load_glove(glove_filename)
    articles = load_articles(articles_dirname, gloves)

    print(gloves['dog'])
