#! /usr/bin/python3
import urllib.request


def download():
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in categories:
        categories_url = c.replace('_', '%20')
        path = base + categories_url + '.npy'
        print(path)
        urllib.request.urlretrieve(path, 'data/' + c + '.npy')


f = open("categories.txt", "r")
# And for reading use
categories = f.readlines()
f.close()

categories = [c.replace('\n', '').replace(' ', '_') for c in categories]
print('Downloading ' + str(len(categories)) + ' categories')

download()
