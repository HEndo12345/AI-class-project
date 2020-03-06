from Neuron_and_Trainer import*
import itertools
with open('cat_data.pkl', 'rb') as f:
        dataor = pkl.load(f)
datatrain= dataor['train']
datatrain[1] = datatrain.pop('cat')
datatrain[0] = datatrain.pop('no_cat')
data = []
for value in datatrain:
    for b in datatrain[value]:
        flat = b.flatten()/255
        data.append((flat,value))
        
simlist = open('gd.txt','w')
simlist.write(str(data))
neuron = Neuron(dimension = 64*64*3, activation = sigmoid)
trainer = Trainer(data, neuron, qloss)
trainer.train(0.001, 1000)
with open('result4.pkl', 'wb') as f:
        pkl.dump(neuron, f)
