import urllib.request
import os.path

def download():
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in categories:
        categories_url = c.replace('_', '%20')
        path = base + categories_url + '.npy'
        
        if not os.path.isfile('data/' +str(c) + '.npy'):
            print('Downloading : ' + str(path))
            urllib.request.urlretrieve(path, 'data/' + c + '.npy')
        else:
            print(str(path) + ' is already downloaded !')


f = open("./categories.txt", "r")
categories = f.readlines()
f.close()

categories = [c.replace('\n', '').replace(' ', '_') for c in categories]
print('Downloading ' + str(len(categories)) + ' categories')

if __name__ == '__main__':
	download()
