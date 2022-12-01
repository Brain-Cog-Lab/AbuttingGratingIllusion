'''
Human experiments
'''
from flask import Flask, request, url_for, redirect, render_template
import os
import random


class SilhouetteTest():
    def __init__(self):
        self.silhouette_labels = ['clock', 'keyboard', 'airplane', 'cat', 'truck', 'oven', 'dog', 'elephant', 'boat', 'car', 'bird', 'bear', 'chair', 'bottle', 'knife', 'bicycle']
        self.ag_silhouette_imgs = [f"{label}/{label}{i}.png" for i in range(1,11) for label in self.silhouette_labels]
        random.shuffle(self.ag_silhouette_imgs)
        self.N = 0
        self.results = []
        self.image_name = None

    def initialize(self):
        random.shuffle(self.ag_silhouette_imgs)
        self.N = 0
        self.results = []
        self.image_name = None

class MNISTTest():
    def __init__(self):
        self.mnist_labels = ['zero','one','two','three','four','five','six','seven','eight','nine']
        self.ag_mnist_imgs = [f"{label}/{i}.png" for i in range(0,10) for label in self.mnist_labels]
        random.shuffle(self.ag_mnist_imgs)
        self.N = 0
        self.results = []
        self.image_name = None
        
    def initialize(self):
        random.shuffle(self.ag_mnist_imgs)
        self.N = 0
        self.results = []
        self.image_name = None

class HighResMNISTTest():
    def __init__(self):
        self.mnist_labels = ['zero','one','two','three','four','five','six','seven','eight','nine']
        self.ag_mnist_imgs = [f"{label}/{i}.png" for i in range(0,10) for label in self.mnist_labels]
        random.shuffle(self.ag_mnist_imgs)
        self.N = 0
        self.results = []
        self.image_name = None
        
    def initialize(self):
        random.shuffle(self.ag_mnist_imgs)
        self.N = 0
        self.results = []
        self.image_name = None



app = Flask(__name__)


@app.route('/')
def index():
    silhouette_test.initialize()
    mnist_test.initialize()
    high_resolution_mnist_test.initialize()
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    if silhouette_test.results != []:
        correct = 0
        for image_name, choice in silhouette_test.results:
            if choice != None and image_name[0:3] == choice[0:3]:
                correct+=1
        result = correct/len(silhouette_test.results)
        print(silhouette_test.results)
        print(f'result:{result}')
    elif mnist_test.results != []:
        correct = 0
        for image_name, choice in mnist_test.results:
            if choice != None and image_name[0:3] == choice[0:3]:
                correct+=1
        result = correct/len(mnist_test.results)
        print(mnist_test.results)
        print(f'result:{result}')
    elif high_resolution_mnist_test.results != []:
        correct = 0
        for image_name, choice in high_resolution_mnist_test.results:
            if choice != None and image_name[0:3] == choice[0:3]:
                correct+=1
        result = correct/len(high_resolution_mnist_test.results)
        print(high_resolution_mnist_test.results)
        print(f'result:{result}')
    if request.method == 'POST':
        return redirect(url_for('index'))
    silhouette_test.initialize()
    mnist_test.initialize()
    high_resolution_mnist_test.initialize()
    return render_template('result.html', result = result)





@app.route('/mnist_test', methods=['GET', 'POST'])
def mnist_test():
    if request.method == 'POST':
        choice = request.form.get('choice')
        mnist_test.results.append((mnist_test.image_name, choice))
        print(mnist_test.results)
        mnist_test.N  += 1
        if mnist_test.N >= 100:
            return redirect(url_for('result'))
    
    mnist_test.image_name = mnist_test.ag_mnist_imgs[mnist_test.N]
    image_path =  url_for('static', filename=f'/mnist/ag_i4_ur/{mnist_test.image_name}')

    return render_template('mnist_test.html', N=mnist_test.N+1, image_path = image_path)


@app.route('/high_resolution_mnist_test', methods=['GET', 'POST'])
def high_resolution_mnist_test():
    if request.method == 'POST':
        choice = request.form.get('choice')
        high_resolution_mnist_test.results.append((high_resolution_mnist_test.image_name, choice))
        print(high_resolution_mnist_test.results)
        high_resolution_mnist_test.N  += 1
        if high_resolution_mnist_test.N >= 100:
            return redirect(url_for('result'))
    
    high_resolution_mnist_test.image_name = high_resolution_mnist_test.ag_mnist_imgs[high_resolution_mnist_test.N]
    image_path =  url_for('static', filename=f'/high-resolution-mnist/ag_i4_ur/{high_resolution_mnist_test.image_name}')

    return render_template('high_resolution_mnist_test.html', N=high_resolution_mnist_test.N+1, image_path = image_path)


@app.route('/sin_test', methods=['GET', 'POST'])
def sin_test():
    if request.method == 'POST':
        choice = request.form.get('choice')
        silhouette_test.results.append((silhouette_test.image_name, choice))
        print(silhouette_test.results)
        silhouette_test.N  += 1
        if silhouette_test.N >= 160:
            return redirect(url_for('result'))
    
    silhouette_test.image_name = silhouette_test.ag_silhouette_imgs[silhouette_test.N]
    image_path =  url_for('static', filename=f'/silhouettes/ag_i4_ur/{silhouette_test.image_name}')

    return render_template('sin_test.html', N=silhouette_test.N+1, image_path = image_path)

if __name__ == "__main__":
    
    silhouette_test = SilhouetteTest()
    mnist_test = MNISTTest()
    high_resolution_mnist_test = HighResMNISTTest()
    app.run(debug = True)
