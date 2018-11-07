from numpy import *  
import numpy as np
import time  
import matplotlib.pyplot as plt 
import pandas as pd 
import os.path
  

def distanciaEuclideana(v1, v2):  
	return sqrt(sum(power(v2 - v1, 2)))  

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Inicializar centroides aleatorios  
def crearCentroides(dataSet, k):
    numData, dimension = dataSet.shape
    centroides = zeros((k, dimension))
    for i in range(k):
        index = int(random.uniform(0, numData))
        centroides[i, :] = dataSet[index, :]
        # print("centroides  index",centroides[i,:])
    return centroides  

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# Divide en k grupos 
def kmeans(dataSet, k):
    cont=0
    numData = dataSet.shape[0]
    catDist = mat(zeros((numData, 2)))
    clusterActualizado = True

    ## Inicializar centroides
    centroides = crearCentroides(dataSet, k)

    while clusterActualizado:
        clusterActualizado = False
        for i in range(numData):
            minDist  = 100
            minIndex = 0

            for j in range(k):
                distancia = distanciaEuclideana(centroides[j, :], dataSet[i, :])
                # print("centroides  ",centroides[j, :])
                # print("dataset ", dataSet[i, :])
                if distancia < minDist:
                    minDist  = distancia
                    minIndex = j
                # print("minD ",minDist)
                # print("minI ",minIndex)
            
            if catDist[i, 0] != minIndex:
                clusterActualizado = True
                catDist[i, :] = minIndex, minDist**2 

        # plotCluster(dataSet,k,centroides,catDist,cont)
        cont+=1   

        ## Actualizar ubicacion de centroides
        for j in range(k):
            puntosCluster = dataSet[nonzero(catDist[:, 0] == j)[0]] #Extraer las muestras correspondientes de la matriz dataSet.
            # print("pcluster ",nonzero(catDist[:, 0].A))
            centroides[j, :] = mean(puntosCluster, axis = 0)  # Promedio de los objetos del cluster
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

def plotCluster(dataSet, k, centroides, catDist, cont):
    numData ,dimension=dataSet.shape

    color = ['or', 'ob', 'og', 'ok', 'oc', 'om', 'oy', 'ow']
    for i in range(numData):
        colorIndex = int(catDist[i,0])
        plt.plot(dataSet[i,0], dataSet[i,1],color[colorIndex])

    color = ['Dr', 'Db', 'Dg', 'Dk', 'Dc', 'Dm', 'Dy', 'Dw']
    for i in range(k):
        # print(centroides[i,0])
        plt.plot(centroides[i, 0], centroides[i, 1], color[i], color[colorIndex])
    
    out_path = os.path.join('D:\imagenes\kmeans', str(cont) + '.png')
    plt.savefig(out_path)
    plt.show()
    plt.close()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

dataSet = []  

fileIn = open("dataNorm.txt") 
for line in fileIn.readlines(): 
    temp=[]
    lineArr = line.strip().split(' ')
    temp.append(float(lineArr[0]))
    temp.append(float(lineArr[1]))
    dataSet.append(temp)
fileIn.close()  

dataSet = mat(dataSet)  
kmeans(dataSet, 3)  

