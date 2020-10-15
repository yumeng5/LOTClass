
green=`tput setaf 2`
reset=`tput sgr0`

echo ${green}===Downloading AG News Data...===${reset}
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/a/illinois.edu/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/a/illinois.edu/uc?export=download&id=1zszTJudS8RMgTQxURkt1w2MhswNGA6Oa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zszTJudS8RMgTQxURkt1w2MhswNGA6Oa" -O agnews.zip && rm -rf /tmp/cookies.txt
echo ${green}===Unzipping AG News Data...===${reset}
unzip agnews.zip && rm agnews.zip
