
green=`tput setaf 2`
reset=`tput sgr0`

echo ${green}===Downloading DBPedia Data...===${reset}
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/a/illinois.edu/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/a/illinois.edu/uc?export=download&id=1nCQQAC6XwfnyKtzWlNElMtz4s12kxfe7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nCQQAC6XwfnyKtzWlNElMtz4s12kxfe7" -O dbpedia.zip && rm -rf /tmp/cookies.txt
echo ${green}===Unzipping DBPedia Data...===${reset}
unzip dbpedia.zip && rm dbpedia.zip
