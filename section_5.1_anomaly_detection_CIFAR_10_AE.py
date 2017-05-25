#http://qiita.com/shngt/items/fba14034f5c45845a16d
#https://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models.pdf
from six.moves import range
import numpy as np
import tflearn
from skimage import io
import skimage
from sklearn.metrics import average_precision_score,mean_squared_error,roc_auc_score
import tensorflow as tf
from tflearn.datasets import cifar10
from tflearn.layers.normalization import local_response_normalization
import matplotlib.pyplot as plt


# Global variables
nb_feature = 64


image_and_anamolies = {'image': 5,'anomalies1':3,'anomalies2':3,'imagecount': 500,'anomaliesCount': 250}
image_with_noise_injection = {'image': 5,'imagecount': 5000}





ROOT = "/Users/raghav/Documents/Uni/KDD-2017/sample_autoencoder/autoencode_softthreshold/cifar-10-batches-py"
basepath="/Users/raghav/Documents/Uni/ECML_2017/experiments/cifar_10/ae/results/inductive/"
mean_square_error_dict ={}

# load cifar-10 data
(X, Y), (testX, testY) = cifar10.load_data(ROOT)
testX = np.asarray(testX)
testY = np.asarray(testY)
side = X.shape[1]
channel = X.shape[3]
noise_factor = 0.1
mue = 0.1
d = 3072
lamda_in_cost = 0.01
N_to_costfunc = np.zeros((200,d ))
# Define the convoluted ae architecture
net = tflearn.input_data(shape=[None, 32, 32, 3])
net = tflearn.fully_connected(net, 256)
hidden_layer = tflearn.fully_connected(net, nb_feature)
net = tflearn.fully_connected(hidden_layer, 256)
decoder = tflearn.fully_connected(net, 32*32*3,activation='sigmoid')
net = tflearn.reshape(decoder, (-1, 32, 32, 3))

# net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
#                          loss='mean_square', metric=None)

mue = 0.1
net = tflearn.regression_RobustAutoencoder(net,mue,hidden_layer,decoder, optimizer='adam', learning_rate=0.001,
                         loss='rPCA_autoencoderLoss_FobsquareLoss', metric=None,name="vanilla_autoencoder")

model = tflearn.DNN(net, tensorboard_verbose=0)



def addNoise(original, noise_factor):
    noisy = original + np.random.normal(loc=0.0, scale=noise_factor, size=original.shape)
    return np.clip(noisy, 0., 1.)
def add_Salt_Pepper_Noise(original, noise_factor):
    #noisy = original + np.random.normal(loc=0.0, scale=noise_factor, size=original.shape)
    noisy = skimage.util.random_noise(original, mode='s&p',clip=False,amount=0.1)
    return np.clip(noisy, 0., 1.)
def prepare_cifar_data_with_anamolies(original,original_labels,image_and_anamolies):

    imagelabel = image_and_anamolies['image']
    imagecnt = image_and_anamolies['imagecount']

    idx = np.where(original_labels ==imagelabel)

    idx = idx[0][:imagecnt]


    images = original[idx]

    images_labels = original_labels[idx]

    anamoliescnt = image_and_anamolies['anomaliesCount']
    anamolieslabel1 = image_and_anamolies['anomalies1']

    anmolies_idx1 = np.where(original_labels ==anamolieslabel1)
    anmolies_idx1 = anmolies_idx1[0][:(anamoliescnt/2)]
    ana_images1 = original[anmolies_idx1]
    ana_images1_labels = original_labels[anmolies_idx1]

    anamolieslabel2 = image_and_anamolies['anomalies2']

    anmolies_idx2 = np.where(original_labels ==anamolieslabel2)
    anmolies_idx2 = anmolies_idx2[0][:(anamoliescnt/2)]
    ana_images2 = original[anmolies_idx2]
    ana_images2_labels = original_labels[anmolies_idx2]

    temp = np.concatenate((images, ana_images1), axis=0)
    data = np.concatenate((temp, ana_images2), axis=0)

    #labels for these images
    templabel = np.concatenate((images_labels, ana_images1_labels), axis=0)
    datalabels = np.concatenate((templabel, ana_images2_labels), axis=0)


    return [data,datalabels]

def prepare_cifar_data_with_noise_injection(original,original_labels,image_with_noise_injection):

    imagelabel = image_with_noise_injection['image']
    imagecnt = image_with_noise_injection['imagecount']

    idx = np.where(original_labels ==imagelabel)
    idx = idx[0][:imagecnt]

    images = original[idx]
    images_labels = original_labels[idx]

    data = images
    datalabels = images_labels

    return [data,datalabels]


def compute_mse(Xclean,Xdecoded,lamda):
    #print len(Xdecoded)
    Xclean = np.reshape(Xclean, (len(Xclean),3072))
    m,n =  Xclean.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded),(m,n))
    #print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded),3072))

    meanSq_error= mean_squared_error(Xclean, Xdecoded)
    mean_square_error_dict.update({lamda:meanSq_error})
    print("\n Mean square error Score ((Xclean, Xdecoded):")
    print(mean_square_error_dict.values())

    return mean_square_error_dict


# Function to compute softthresholding values
def soft_threshold(lamda,b):

    th = float(lamda)/2.0
    print ("(lamda,Threshold)",lamda,th)
    print("The type of b is ..., its len is ",type(b),b.shape,len(b[0]))

    if(lamda == 0):
        return b
    m,n = b.shape

    x = np.zeros((m,n))

    k = np.where(b > th)
    # print("(b > th)",k)
    #print("Number of elements -->(b > th) ",type(k))
    x[k] = b[k] - th

    k = np.where(np.absolute(b) <= th)
    # print("abs(b) <= th",k)
    # print("Number of elements -->abs(b) <= th ",len(k))
    x[k] = 0

    k = np.where(b < -th )
    # print("(b < -th )",k)
    # print("Number of elements -->(b < -th ) <= th",len(k))
    x[k] = b[k] + th
    x = x[:]

    return x

def compute_best_worst_rank(testX,Xdecoded):
     #print len(Xdecoded)

    testX = np.reshape(testX, (len(testX),3072))
    m,n =  testX.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded),(m,n))
    #print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded),3072))

    # Rank the images by reconstruction error
    anamolies_dict = {}
    for i in range(0,len(testX)):
        anamolies_dict.update({i:np.linalg.norm(testX[i] - Xdecoded[i])})

    # Sort the recont error to get the best and worst 10 images
    best_top10_anamolies_dict={}
    # Rank all the images rank them based on difference smallest  error
    best_sorted_keys = sorted(anamolies_dict, key=anamolies_dict.get, reverse=False)
    worst_top10_anamolies_dict={}
    worst_sorted_keys = sorted(anamolies_dict, key=anamolies_dict.get, reverse=True)


    # Picking the top 10 images that were not reconstructed properly or badly reconstructed
    counter_best = 0
    # Show the top 10 most badly reconstructed images
    for b in best_sorted_keys:
        if(counter_best <= 29):
            counter_best = counter_best + 1
            best_top10_anamolies_dict.update({b:anamolies_dict[b]})
    best_top10_keys = best_top10_anamolies_dict.keys()


    # Picking the top 10 images that were not reconstructed properly or badly reconstructed
    counter_worst = 0
    # Show the top 10 most badly reconstructed images
    for w in worst_sorted_keys:
        if(counter_worst <= 29):
            counter_worst = counter_worst + 1
            worst_top10_anamolies_dict.update({w:anamolies_dict[w]})
    worst_top10_keys = worst_top10_anamolies_dict.keys()

    return [best_top10_keys,worst_top10_keys]

def fit_auto_DAE(input,Xclean):


    input = np.reshape(input, (len(input),32,32,3))
    model.fit(input, Xclean, n_epoch=10,
          run_id="auto_encoder", batch_size=128)

    ae_output = model.predict(input)
    ae_output = np.reshape(ae_output, (len(ae_output),3072))

    return ae_output


def compute_softhreshold(XtruewithNoise,N,lamda,Xclean):
    XtruewithNoise = np.reshape(XtruewithNoise, (len(XtruewithNoise),3072))
    print "lamda passed ",lamda
    # inner loop for softthresholding
    for i in range(0, 1):
        train_input = XtruewithNoise - N
        XAuto = fit_auto_DAE(train_input,Xclean) # XAuto is the predictions on train set of autoencoder
        XAuto = np.asarray(XAuto)
        #print "XAuto:",type(XAuto),XAuto.shape
        softThresholdIn = XtruewithNoise - XAuto
        softThresholdIn = np.reshape(softThresholdIn, (len(softThresholdIn),3072))
        N = soft_threshold(lamda,softThresholdIn)
        print("Iteration NUmber is : ",i)
        print ("NUmber of non zero elements  for N,lamda",np.count_nonzero(N),lamda)
        print ( "The shape of N", N.shape)
        print ( "The minimum value of N ", np.amin(N))
        print ( "The max value of N", np.amax(N))


    return N


def visualise_anamolies_detected(testX,noisytestX,decoded,N,best_top10_keys,worst_top10_keys,lamda):

    N = np.reshape(N, (len(N),32,32,3))
    #Display the decoded Original, noisy, reconstructed images
    print "side:",side
    print "channel:",channel
    img = np.ndarray(shape=(side*4, side*10, channel))
    print "img shape:",img.shape

    for i in range(10):
        row = i // 10 * 3
        col = i % 10
        img[side*row:side*(row+1), side*col:side*(col+1), :] = testX[best_top10_keys[i]]
        img[side*(row+1):side*(row+2), side*col:side*(col+1), :] = noisytestX[best_top10_keys[i]]
        img[side*(row+2):side*(row+3), side*col:side*(col+1), :] = decoded[best_top10_keys[i]]
        img[side*(row+3):side*(row+4), side*col:side*(col+1), :] = N[best_top10_keys[i]]

    img *= 255
    img = img.astype(np.uint8)

    #Save the image decoded
    print("\nSaving results for best after being encoded and decoded: @")
    print(basepath+'/best/')
    io.imsave(basepath+'/best/'+str(lamda)+'salt_p_denoising_cae_decode.png', img)

    #Display the decoded Original, noisy, reconstructed images for worst
    img = np.ndarray(shape=(side*4, side*10, channel))
    for i in range(10):
        row = i // 10 * 3
        col = i % 10
        img[side*row:side*(row+1), side*col:side*(col+1), :] = testX[worst_top10_keys[i]]
        img[side*(row+1):side*(row+2), side*col:side*(col+1), :] = noisytestX[worst_top10_keys[i]]
        img[side*(row+2):side*(row+3), side*col:side*(col+1), :] = decoded[worst_top10_keys[i]]
        img[side*(row+3):side*(row+4), side*col:side*(col+1), :] = N[worst_top10_keys[i]]

    img *= 255
    img = img.astype(np.uint8)

    #Save the image decoded
    print("\nSaving results for worst after being encoded and decoded: @")
    print(basepath+'/worst/')
    io.imsave(basepath+'/worst/'+str(lamda)+'salt_p_denoising_cae_decode.png', img)


    return


def evalPred(predX, trueX, trueY):

    trueX = np.reshape(trueX, (len(trueX),3072))
    # m,n =  trueX.shape
    # predX = np.reshape(np.asarray(predX),(m,n))
    #print Xdecoded.shape
    predX = np.reshape(predX, (len(predX),3072))

    if predX.shape[1] > 1:
        recErr = ((predX - trueX) ** 2).sum(axis = 1)
    else:
        recErr = predX

    ap  = average_precision_score(trueY, recErr)
    auc = roc_auc_score(trueY, recErr)

    return (ap, auc)
# Prepare data with anamolies defines as per image_and_anamolies
# [X,Y]=prepare_cifar_data_with_noise_injection(X,Y,image_with_noise_injection)
# [testX,testY]=prepare_cifar_data_with_anamolies(testX,testY,image_and_anamolies)


# [X,Y]=prepare_cifar_data_with_anamolies(X,Y,image_and_anamolies)
# [testX,testY]=prepare_cifar_data_with_anamolies(testX,testY,image_and_anamolies)

[X,Y]=prepare_cifar_data_with_noise_injection(X,Y,image_with_noise_injection)
[testX,testY]=prepare_cifar_data_with_anamolies(testX,testY,image_and_anamolies)


# X= np.concatenate((XPos,XNeg),axis=0)
# Y=np.concatenate((YPos,YNeg),axis=0)

# Make the Dog samples as positive and cat samples as negative
testY[testY ==5] = 1
testY[testY ==3] =-1

Y[Y ==5]= 1
Y[Y ==3]= -1
Y = Y.ravel()

# Make the Dog samples as positive and cat samples as negative
testY[testY ==5] = 1
testY[testY ==3] = 0
# Prepare a noisy dataset
noisyX = add_Salt_Pepper_Noise(X, noise_factor)
#noisyX = addNoise(X, noise_factor)

# testX is used to predict the trained cae
testX = tflearn.data_utils.shuffle(testX)[0]
#noisyTestX = addNoise(testX, noise_factor)
noisytestX = add_Salt_Pepper_Noise(testX, noise_factor)




#define lamda set
lamda_set = [ 0.0,0.01,0.1,0.5,1.0, 10.0, 100.0]
#lamda_set = [0.0]
mue = 0.0
TRIALS= 7
ap = np.zeros((TRIALS,))
auc = np.zeros((TRIALS,))


# outer loop for lamda
for l in range(0,len(lamda_set)):
    # Learn the N using softthresholding technique
    N =  0
    lamda = lamda_set[l]
    N = compute_softhreshold(X,N,lamda,X)

    #Predict the conv_AE autoencoder output
    XTrue = np.reshape(X, (len(X),32,32,3))
    decoded = model.predict(XTrue)

    #compute MeanSqared error metric
    compute_mse(XTrue,decoded,lamda)

    # rank the best and worst reconstructed images
    [best_top10_keys,worst_top10_keys]=compute_best_worst_rank(XTrue,decoded)

    #Visualise the best and worst ( image, BG-image, FG-Image)
    XPred = np.reshape(np.asarray(decoded),(len(decoded),32,32,3))
    visualise_anamolies_detected(X,X,decoded,N,best_top10_keys,worst_top10_keys,lamda)

    XPred = decoded
    # print type(XPred),len(XPred)
    XTrue=X
    # print type(XTrue),len(XTrue),XTrue.shape
    YTrue = Y
  # YTrue  print "Ytrue============================"
  #   print type(YTrue),len(YTrue),YTrue.shape
  #   print

    (ap[l], auc[l]) = evalPred(XPred, XTrue, YTrue)



print('AUPRC = %1.4f +- %1.4f' % (np.mean(ap), np.std(ap)/np.sqrt(TRIALS)))
print('AUROC = %1.4f +- %1.4f' % (np.mean(auc), np.std(auc)/np.sqrt(TRIALS)))


# plotting the mean precision score
print("\n Saving the Mean square error Score ((Xclean, Xdecoded):")
fig1_mean_square_error=plt.figure(figsize=(8,5))
plt.xlabel("CAE-Denoiser")
plt.ylabel("Mean- Sq Error")
print("\n Mean square error Score ((Xclean, Xdecoded):")
print(mean_square_error_dict.values())

for k,v in mean_square_error_dict.iteritems():
    print k,v
# basic plot
data = mean_square_error_dict.values()
plt.boxplot(data)
fig1_mean_square_error.savefig(basepath+'_mean_square_error.png')


