#Code Written By ourselves
#For drawing the leaning curves

import matplotlib.pyplot as plt
name = "./head_logs/copy_ntm_"
train_loss = []
with open(name+"train.txt") as fin:
    for line in fin:
        arr = line.replace("\n","").split(" ")
        train_loss.append(float(arr[0]))
train_index = [ i for i in range(len(train_loss))]

test_same_loss = []
with open(name+"test_same.txt") as fin:
    for line in fin:
        arr = line.replace("\n","").split(" ")
        test_same_loss.append(float(arr[0]))
test_same_index = [ i*200 for i in range(len(test_same_loss))]

test_different_40_loss = []
with open(name+"test_different_40.txt") as fin:
    for line in fin:
        arr = line.replace("\n","").split(" ")
        test_different_40_loss.append(float(arr[0])/2)
test_different_40_index = [ i*200 for i in range(len(test_different_40_loss))]

plt.plot(train_index,train_loss, label="train")
plt.plot(test_same_index,test_same_loss, label="test")
plt.plot(test_different_40_index,test_different_40_loss,label="test_40")

plt.xlabel("Times")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss.png")
plt.show()
