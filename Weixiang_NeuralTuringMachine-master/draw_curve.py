import matplotlib.pyplot as plt
name = "copy_ntm_"
train_loss = []
with open(name+"train.txt") as fin:
    for line in fin:
        arr = line.replace("\n","")
        train_loss.append(float(arr[0]))
train_index = [ i for i in range(len(train_loss))]

test_same_loss = []
with open(name+"test_same.txt") as fin:
    for line in fin:
        arr = line.replace("\n","")
        test_same_loss.append(float(arr[0]))
test_same_index = [ i*200 for i in range(len(test_same_loss))]

test_different_40_loss = []
with open(name+"test_different_40.txt") as fin:
    for line in fin:
        arr = line.replace("\n","")
        test_different_40_loss.append(float(arr[0]))
test_different_40_index = [ i*200 for i in range(len(test_different_40_loss))]

test_different_60_loss = []
with open(name+"test_different_60.txt") as fin:
    for line in fin:
        arr = line.replace("\n","")
        test_different_60_loss.append(float(arr[0]))
test_different_60_index = [ i*200 for i in range(len(test_different_60_loss))]

test_different_80_loss = []
with open(name+"test_different_80.txt") as fin:
    for line in fin:
        arr = line.replace("\n","")
        test_different_80_loss.append(float(arr[0]))
test_different_80_index = [ i*200 for i in range(len(test_different_80_loss))]

test_different_100_loss = []
with open(name+"test_different_100.txt") as fin:
    for line in fin:
        arr = line.replace("\n","")
        test_different_100_loss.append(float(arr[0]))
test_different_100_index = [ i*200 for i in range(len(test_different_100_loss))]

test_different_120_loss = []
with open(name+"test_different_120.txt") as fin:
    for line in fin:
        arr = line.replace("\n","")
        test_different_120_loss.append(float(arr[0]))
test_different_120_index = [ i*200 for i in range(len(test_different_120_loss))]

plt.plot(train_index,train_loss, label="train")
plt.plot(test_same_index,test_same_loss, label="test")
plt.plot(test_different_40_index,test_different_40_loss,label="test_40")
plt.plot(test_different_60_index,test_different_60_loss,label="test_60")
plt.plot(test_different_80_index,test_different_80_loss,label="test_80")
plt.plot(test_different_100_index,test_different_100_loss,label="test_100")
plt.plot(test_different_120_index,test_different_120_loss,label="test_120")
plt.xlabel("Times")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss.png")
plt.show()
