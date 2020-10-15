
green=`tput setaf 2`
reset=`tput sgr0`

echo ${green}===Downloading Amazon Data...===${reset}
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/a/illinois.edu/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/a/illinois.edu/uc?export=download&id=1pRt5mPuuVbi-ZXD8QZzw_7DlAnEg3X15' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pRt5mPuuVbi-ZXD8QZzw_7DlAnEg3X15" -O amazon.zip && rm -rf /tmp/cookies.txt
echo ${green}===Unzipping Amazon Data...===${reset}
unzip amazon.zip && rm amazon.zip
