from CMC import CMC
import numpy as np
import cv2 as cv

def cmc_values(rank, num_of_frames):
    i = 0
    score = 0
    values = []
    for i in range(0, len(rank)):
        score = score + float(rank[i])/float(num_of_frames)
        values.append(score)
    return values

def plot_cmc(values, title, ranks):
    if ranks > len(values):
        rank = len(values)
    else:
        rank = ranks
    cmc_dict ={
        "at Rank 1" : values
    }
    cmc = CMC(cmc_dict)
    cmc.plot(title = title, rank = rank)

def getOriginalClass(label_batch):
    for i, item in enumerate(label_batch):
        if int(item) == 1:
            class_name = i
            break
    return class_name


#prende batchMatrix: dim: numero immagini di un batch, numero classi +1
def countTry(batchMatrix, num_classes):
    count = np.zeros(shape=num_classes)

    for i in range(len(batchMatrix)):
        trueClass= batchMatrix[i][num_classes]
        for j in range(num_classes):
            if trueClass == batchMatrix[i][j]:
                count[j] += 1
                break
    return count, len(batchMatrix)





#passare tutto il batch_preds e label_batch
def defineImageRank(batch_preds,label_batch):

    batchMatrix =[]
    numClasses = int(len(batch_preds[0]))
    rank = np.zeros(shape =(numClasses,3))

    for i in range(len(batch_preds)):
        #batchMatrix.append([])

        trueClass = getOriginalClass(label_batch[i])

        for j in range(0,numClasses):
            rank[j][0] = j
            rank[j][1]=batch_preds[i][j]
            rank[j][2]=trueClass
            
        rankO = rank[np.argsort(-rank[:, 1])]

        sortedClass =[]
        for j in range(numClasses):
            sortedClass.append(int(rankO[j][0]))

        sortedClass.append(int(rankO[0][2]))
        batchMatrix.insert(i, sortedClass)

    return batchMatrix





def probabilityToClass(batch_preds,label_batch,rgb_batch):
    #np.matrix("")
    batchClassPreds = np.zeros(shape =(len(batch_preds),len(batch_preds[0])))

    #print(("32 =="+ str(len(batch_preds))+ " 49 == "+ str(len(batch_preds[0]) )))
    for i in range (0,len(batch_preds)):   #dovrebbe essere 32
        #print("len(batch_preds)32---"+ str(len(batch_preds)))
        maxValue = 0
        probab= 0

        try:
            print("label_batch "+ str(label_batch))
        except:
            print("errore label_batch")

        for k, value in enumerate (batch_preds[i]) :
            #print("len(batch_preds[0]) 49---" + str(len(batch_preds[0])))
            print("len(label_batch["+str(i)+"]) lista classi ---" + str((label_batch[i]))+" classe immagine? "+ str((label_batch[i][k]) ))

            #stampa il numero della classe associata(h)
            h=0
            for j,kval in enumerate(label_batch[i]):
                if kval > 0:
                    h=j
                    print("TROVATO")
                    break

            cv.imshow("IMMAGINE classe " + str(h+1), rgb_batch[i])


            probab = probab + float(batch_preds[i][k])
            if value > maxValue:
                maxValue = value
                batchClassPreds[i][k-1] = 0
                batchClassPreds[i][k] = 1
            else:
                batchClassPreds[i][k] = 0


            print("classi predette\n")
            print(batchClassPreds[i][k])
            print("classi effettive\n")
            print(label_batch[i][k])


            cv.waitKey(0)

        #somma delle probabilità tra le classi
        #print(probab)

    return batchClassPreds