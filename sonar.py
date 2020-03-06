from Neuron_and_Trainer import*
with open('sonar_data.pkl', 'rb') as f:
        dataor = pkl.load(f)
dataor[1] = dataor.pop('m')
dataor[-1] = dataor.pop('r')
data = []
for value in dataor:
    for b in dataor[value][:round(len(dataor[value])*0.90)]:
        data.append((b,value))
random.shuffle(data)
neuron = Neuron(dimension = len(dataor[1][0]), activation = perceptron)
trainer = Trainer(data, neuron, ploss)
trainer.train(0.05, 200)
with open('result.pkl', 'wb') as f:
        pkl.dump(neuron, f)
