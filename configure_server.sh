mkdir anaconda
cd anaconda
wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh
# PREFIX=/home/ubuntu/anaconda3
source ~/.bashrc
pip install numpy gunicorn Flask tweepy vaderSentiment colour
sudo yum install git
sudo apt-get install git
git clone https://github.com/USF-MSDS692/recommender-miragegokul.git
cd recommender-miragegokul/
mkdir data
cd data
wget https://s3-us-west-1.amazonaws.com/msan692/bbc.zip
wget https://s3-us-west-1.amazonaws.com/msan692/glove.6B.300d.txt.zip
unzip bbc.zip
unzip glove.6B.300d.txt.zip
cd ..

gunicorn -b 0.0.0.0:5000 server:app /home/ec2-user/recommender-miragegokul/data/glove.6B.300d.txt /home/ec2-user/recommender-miragegokul/data/bbc