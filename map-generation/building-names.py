#!/usr/bin/env python3
import pandas as pd
import json

def main():
    all_words = set()
    df = pd.read_csv('outputs/buildings.csv')
    #print(df['name'])
    for i in df['name']:
        words = i.split(' ')
        for word in words:
            word = word.replace('(', '')
            word = word.replace(')', '')
            all_words.add(word)
    print(all_words)
    d = {}
    d['words'] = list(all_words)
    with open('outputs/words.json', 'w') as f:
        json.dump(d, f)


if __name__ == "__main__":
    main()
