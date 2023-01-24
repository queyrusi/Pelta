import torch 
import DataManagerPytorch as DMP
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time
import copy
import random
from TransformerModels import VisionTransformer, CONFIGS

# Set to true if random attack should be performed instead of FGSM/SAGA
RANDOM_ADVERSARIAL = False
# Set to true if FGSM/SAGA should be performed.
ATTACK = True

#Main attack method, takes in a list of models and a clean data loader
#Returns a dataloader with the adverarial samples and corresponding clean labels
def SelfAttentionGradientAttack(device, epsMax, numSteps, modelListPlus,
                                coefficientArray, dataLoader, clipMin, clipMax):
    #Basic graident variable setup 
    xClean, yClean = DMP.DataLoaderToTensor(dataLoader)
    xAdv = xClean #Set the initial adversarial samples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this) 
    #Compute eps step
    epsStep = epsMax/numSteps
    dataLoaderCurrent = dataLoader

    os.makedirs("Pictures", exist_ok = True)

    for i in range(0, numSteps):
        xGradientCumulative = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
        if not RANDOM_ADVERSARIAL and ATTACK:
            for m in range(0, len(modelListPlus)):
                currentModelPlus = modelListPlus[m]
                #First get the gradient from the model 
                xGradientCurrent = FGSMNativeGradient(device, dataLoaderCurrent, currentModelPlus)
                #Resize the graident to be the correct size 
                xGradientCurrent = torch.nn.functional.interpolate(xGradientCurrent, size=(xShape[1], xShape[2]))
                #Add the current computed gradient to the result 
                if currentModelPlus.modelName.find("ViT")>=0: 
                    attmap = GetAttention(dataLoaderCurrent, currentModelPlus)
                    xGradientCumulative = xGradientCumulative + coefficientArray[m]*xGradientCurrent *attmap
                else:
                    xGradientCumulative = xGradientCumulative + coefficientArray[m]*xGradientCurrent
        #Compute the sign of the graident and create the adversarial example 
        elif RANDOM_ADVERSARIAL:
            how_many = numSamples*xShape[0]*xShape[1]*xShape[2]
            xGradientCumulative = torch.from_numpy(np.array([random.choice([-1, 1])\
             for k in range(how_many)]).reshape((numSamples, xShape[0], xShape[1], xShape[2])))
            
        if ATTACK:
            xAdv = xAdv + epsStep*xGradientCumulative.sign()
        #Do the clipping 
        xAdv = torch.clamp(xAdv, clipMin, clipMax)
        

        # Plot comparison
        comparisonArray = np.concatenate((xClean[[0],:],
                                          260 * (xClean[[0],:]-xAdv[[0],:]),
                                          xAdv[[0],:]))
                                          #     ^-- 1st picture
        #Convert the result to dataloader
        dataLoaderCurrent = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)
    return dataLoaderCurrent

def GetAttention(dLoader, modelPlus):
    numSamples = len(dLoader.dataset)
    attentionMaps = torch.zeros(numSamples, modelPlus.imgSizeH, modelPlus.imgSizeW,3)
    currentIndexer = 0
    model = modelPlus.model.to(modelPlus.device)
    for ii, (x, y) in enumerate(dLoader):
        x = x.to(modelPlus.device)
        y = y.to(modelPlus.device)
        bsize = x.size()[0]
        attentionMapBatch = get_attention_map(model, x, bsize)
        # for i in range(0, dLoader.batch_size):
        for i in range(0, bsize):
            attentionMaps[currentIndexer] = attentionMapBatch[i]
            currentIndexer = currentIndexer + 1 
    del model
    torch.cuda.empty_cache() 
    # change order
    attentionMaps = attentionMaps.permute(0,3,1,2)
    return attentionMaps

#Native (no attack library) implementation of the FGSM attack in Pytorch 
def FGSMNativeGradient(device, dataLoader, modelPlus):
    #Basic variable setup
    model = modelPlus.model
    model.eval() #Change model to evaluation mode for the attack 
    model.to(device)
    sizeCorrectedLoader = modelPlus.formatDataLoader(dataLoader)
    #Generate variables for storing the adversarial examples 
    numSamples = len(sizeCorrectedLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(sizeCorrectedLoader) #Get the shape of the input (there may be easier way to do this)
    xGradient = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through each sample 
    tracker = 0

    for xData, yData in sizeCorrectedLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        #Put the data from the batch onto the device 
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        xDataTemp.requires_grad = True
        output = model(xDataTemp)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        # TODO We try to visualize gradient wrt parameters
        VITT = False
        PELTA = False
        cost.backward()
        xDataTempGrad = xDataTemp.grad.data
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xGradient[advSampleIndex] = xDataTempGrad[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        #Not sure if we need this but do some memory clean up 
        del xDataTemp
        torch.cuda.empty_cache()
    #Memory management 
    del model
    torch.cuda.empty_cache() 
    return xGradient

#Do the computation from scratch to get the correctly identified overlapping examples  
#Note these samples will be the same size as the input size required by the 0th model 
def GetFirstCorrectlyOverlappingSamplesBalanced(device, sampleNum, numClasses, dataLoader, modelPlusList):
    numModels = len(modelPlusList)
    totalSampleNum = len(dataLoader.dataset)
    #First check if modelA needs resize
    xTestOrig, yTestOrig = DMP.DataLoaderToTensor(dataLoader)
    #We need to resize first 
    if modelPlusList[0].imgSizeH != xTestOrig.shape[2] or modelPlusList[0].imgSizeW != xTestOrig.shape[3]:
        xTestOrigResize = torch.zeros(xTestOrig.shape[0], 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
        rs = torchvision.transforms.Resize((modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)) #resize the samples for model A
        #Go through every sample 
        for i in range(0, xTestOrig.shape[0]):
            xTestOrigResize[i] = rs(xTestOrig[i]) #resize to match dimensions required by modelA
        #Make a new dataloader
    # #Get accuracy array for each model 
    accArrayCumulative = torch.zeros(totalSampleNum) #Create an array with one entry for ever sample in the dataset
    for i in range(0, numModels):
        accArray = modelPlusList[i].validateDA(dataLoader)
        accArrayCumulative = accArrayCumulative + accArray
    #Basic variable setup 
    samplePerClassCount = torch.zeros(numClasses) #keep track of samples per class
    maxRequireSamplesPerClass = 1
    xTest, yTest = DMP.DataLoaderToTensor(dataLoader) #Get all the data as tensors 
    # >>>yTest
    # tensor([3., 8., 8., 0. ... 4., 1., 9., 5.])
    #Memory for the solution 
    xClean = torch.zeros(sampleNum, 3, modelPlusList[0].imgSizeH, modelPlusList[0].imgSizeW)
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    #Go through all the samples
    # for i in range(0, totalSampleNum):
    i = 0
    while i < int(os.getenv('ATTACK_SAMPLE_POOL_SIZE')) and sampleIndexer < int(os.getenv('ATTACK_SAMPLES_NUM')):
        currentClass = int(yTest[i])
        if currentClass>=numClasses:
            pass
        if accArrayCumulative[i] == numModels:
            xClean[sampleIndexer] = xTest[i]
            yClean[sampleIndexer] = yTest[i]
            sampleIndexer = sampleIndexer +1 #update the indexer 
            samplePerClassCount[currentClass] = samplePerClassCount[currentClass] + 1 #Update the number of samples for this class
        i+=1

    #Check the over all number of samples as well
    if sampleIndexer != sampleNum:
        print("Not enough clean samples found.")
    #All error checking done, return the clean balanced loader 
    #Conver the solution into a dataloader
    cleanDataLoader = DMP.TensorToDataLoader(xClean, yClean, transforms = None, batchSize = modelPlusList[0].batchSize, randomizer = None)

    #Do one last check to make sure all samples identify the clean loader correctly 
    for i in range(0, numModels):
        cleanAcc = modelPlusList[i].validateD(cleanDataLoader)
        if cleanAcc[0] != 1.0:
            raise ValueError("The clean accuracy is not 1.0")
        else:
            print("Clean Acc "+ modelPlusList[i].modelName+":", cleanAcc[0])
    return cleanDataLoader

def get_attention_map(model, xbatch, batch_size, img_size=int(os.getenv("IMG_SIZE"))):
    attentionMaps = torch.zeros(batch_size,img_size, img_size,3)
    index = 0
    for i in range(0,batch_size):
        ximg = xbatch[i].cpu().numpy().reshape(1,3,img_size,img_size)
        ximg = torch.tensor(ximg).cuda()
        model.eval()
        res, att_mat = model.forward2(ximg)  
        # model should return attention_list for all attention_heads
        # each element in the list contains attention for each layer
       
        att_mat = torch.stack(att_mat).squeeze(1)
        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat.cpu().detach() + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
    
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), (img_size, img_size))[..., np.newaxis]
        mask = np.concatenate((mask,)*3, axis=-1)
        attentionMaps[index] = torch.from_numpy(mask)
        index = index + 1
    return attentionMaps

def visualize_imgs(imgs, labels=None, name="default", row=1, cols=3,):
    imgs = imgs.transpose(0, 2,3,1)
    # now we need to unnormalize our images. 
    fig = plt.figure(figsize=(20,5))
    for i in range(imgs.shape[0]):
        ax = fig.add_subplot(row, cols, i+1, xticks=[], yticks=[])
        ax.imshow(imgs[i])
        #ax.set_title(labels[i].item())
    plt.savefig(name)