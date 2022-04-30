import csv
from tqdm import tqdm

with open('./badwords.txt', 'r') as f1:
    str1 = f1.read()
    f1.close()

with open('./not_badwords.txt', 'r') as f1:
    str2 = f1.read()
    f1.close()

bad_str = str1.replace('\n', '/')
bad_str = bad_str.split('/')

good_str = str2.replace('\n', '/')
good_str = good_str.split('/')

f2 = open('words.csv', 'w', newline='')
wr = csv.writer(f2)

for word in tqdm(bad_str):
    wr.writerow([word, 0])
for word in tqdm(good_str):
    if word in bad_str:
        pass
    else:
        wr.writerow([word, 1])
f2.close()