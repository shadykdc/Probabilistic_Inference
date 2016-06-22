#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *

#
# Coursework 4
#

def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    mean_int=0.0
    for j in range(noVariables):
        mean_int=(numpy.mean(realData[:,j]))
        mean.append(mean_int)
    
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    
    mean=Mean(theData)
    N=len(realData[:,0])
    for i in xrange(noVariables):
        for j in xrange(noVariables):
            for k in xrange(N):
            covar[i][j]+=(realData[k,i]-mean[i])*(realData[k,j]-mean[j])
    covar/=N

    return covar


def CreateEigenfaceFiles(theBasis):
    for i in range(len(theBasis[:,0])):
        filename='PrincipalComponent' + str(i) + '.jpg'
        SaveEigenface(theBasis[i,:],filename)


def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    data_faceimage_normalized=array(theFaceImage)-array(theMean)
    
    for i in range(len(theBasis[:,0])):
        magnitudes.append(sum(theBasis[i,:]*data_faceimage_normalized))
    
    return array(magnitudes)



def CreatePartialReconstructions(aBasis, aMean, componentMags):
    current_image=aMean#We start with the mean
    
    for i in range(len(componentMags)):
        current_image+=componentMags[i]*aBasis[i,:]
        filename='Reconstruction_Step' + str(i) + '.jpg'
        SaveEigenface(current_image,filename)



def PrincipalComponents(theData):
    orthoPhi = []
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes
    mean=Mean(dataImages)
    U=dataImages-mean
    
    (Eigenvalues,EigenUUtranspose)=linalg.eig(dot(U,U.T))
    
    Eigenvectors=dot(U.T,EigenUUtranspose)

    for i in range(len(Eigenvectors[0,:])): #Normalisation, the basis is already orthogonal (excluding the vector with eigenvalue 0)
        Eigenvectors[:,i]/=sqrt(sum(Eigenvectors[:,i]*Eigenvectors[:,i]))
      
    for i in range(len(Eigenvectors[0,:])): #Sort
        for j in range(i,len(Eigenvectors[0,:])-1):
            if(Eigenvalues[j]<Eigenvalues[j+1]):
            temp=Eigenvalues[j]
            Eigenvalues[j]=Eigenvalues[j+1]
            Eigenvalues[j+1]=temp
	  
            temp2=Eigenvectors[:,j].copy()
            Eigenvectors[:,j]=Eigenvectors[:,j+1].copy()
            Eigenvectors[:,j+1]=temp2.copy()
    
    orthoPhi=Eigenvectors.T
    return array(orthoPhi)


# Main program for Coursework 4
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)

AppendString("results.txt","Coursework Four Results by Kathryn Shea & Pierre Thary")
AppendString("results.txt","") #blank line
AppendString("results.txt","The mean vector for Hepatitis C data set")
AppendList("results.txt",Mean(theData))
AppendString("results.txt","") #blank line
AppendString("results.txt","The covariance matrix for Hepatitis C data set")
AppendArray("results.txt",Covariance(theData))
AppendString("results.txt","") #blank line

#Tasks 4.3 to 4.5
#Put the basis into a matrix
theBasis1=ReadEigenfaceBasis()
#Put the mean into a list
theMean1=ReadOneImage('MeanImage.jpg')
#Convert the basis into actual images
#CreateEigenfaceFiles(theBasis1)
theFaceImage=ReadOneImage('c.pgm')
#Get the magnitudes of 'c.pgm' projected onto the basis
mag=ProjectFace(theBasis1, theMean1, theFaceImage)

AppendString("results.txt","The components magnitudes of 'c.pgm' projected on that data set (Task 4.4)")
AppendList("results.txt",mag)
AppendString("results.txt","") #blank line

#Reconstruction 1
#CreatePartialReconstructions(theBasis1, theMean1, mag)

#Task 4.6
#Read images from a to f
dataImages=array(ReadImages())
#Compute and save the mean into a jpeg file
SaveEigenface(Mean(dataImages),'MeanImageAtoF.jpg')
theMean2=ReadOneImage('MeanImageAtoF.jpg')

#Compute the PCA of these 6 images
pca_basis=PrincipalComponents(dataImages)
#Project 'c.pgm' onto this new basis
mag2=ProjectFace(pca_basis, theMean2, theFaceImage)


# Uncomment if you want to compute the images to reconstruct c.pgm iteratively 
CreateEigenfaceFiles(pca_basis)
CreatePartialReconstructions(pca_basis, theMean2, mag2)
