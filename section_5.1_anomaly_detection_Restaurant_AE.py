#http://qiita.com/shngt/items/fba14034f5c45845a16d
#https://papers.nips.cc/paper/5023-generalized-denoising-auto-encoders-as-generative-models.pdf
from six.moves import range
import numpy as np
import tflearn
from skimage import io
import skimage
from sklearn.metrics import average_precision_score,mean_squared_error
import tensorflow as tf
from tflearn.datasets import cifar10
from tflearn.layers.normalization import local_response_normalization
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Global variables
nb_feature = 64
image_and_anamolies = {'image': 5,'anomalies1':3,'anomalies2':3,'imagecount': 10000,'anomaliesCount': 500}
ROOT = "/Users/raghav/Documents/Uni/KDD-2017/sample_autoencoder/autoencode_softthreshold/cifar-10-batches-py"
basepath="/Users/raghav/Documents/Uni/ECML_2017/experiments/restaurant/ae/results/"
mean_square_error_dict ={}


side1 = 120
side2 = 160
channel = 1
noise_factor = 0.0



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
def compute_mse(Xclean,Xdecoded,lamda):
    #print len(Xdecoded)
    Xclean = np.reshape(Xclean, (len(Xclean),19200))
    m,n =  Xclean.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded),(m,n))
    #print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded),19200))

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

    testX = np.reshape(testX, (len(testX),19200))
    m,n =  testX.shape
    Xdecoded = np.reshape(np.asarray(Xdecoded),(m,n))
    #print Xdecoded.shape
    Xdecoded = np.reshape(Xdecoded, (len(Xdecoded),19200))

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

# Function to train and predict autoencoder output
def fit_auto(input,testX):
    model.fit(input, input, n_epoch=10, validation_set=(testX,testX),
          run_id="vanilla_auto_encoder", batch_size=10)
    # Compute the predictions
    encode_decode = model.predict(testX)
    return encode_decode


def fit_auto_DAE(input,Xclean):

    input = np.reshape(input, (len(input),120,160))
    Xclean = np.reshape(Xclean, (len(Xclean),120,160))

    model.fit(input, Xclean, n_epoch=10,validation_set=0.1,
          run_id="auto_encoder", batch_size=10)

    ae_output = model.predict(input)
    ae_output = np.reshape(ae_output, (len(ae_output),19200))

    return ae_output

def compute_softhreshold(XtruewithNoise,N,lamda,Xclean):
    #XtruewithNoise = np.reshape(XtruewithNoise, (len(XtruewithNoise),19200))
    print "lamda passed ",lamda



    # inner loop for softthresholding
    for i in range(0, 10):

        #print "XtruewithNoise shape ",XtruewithNoise.shape
        #print "N-shape",N.shape
        XtruewithNoise = np.reshape(XtruewithNoise, (len(XtruewithNoise),19200))
        train_input = XtruewithNoise - N
        train_input = train_input.reshape([-1, side1, side2, 1])
        XAuto = fit_auto_DAE(train_input,Xclean) # XAuto is the predictions on train set of autoencoder
        XAuto = np.reshape(XAuto, (len(XAuto),19200))
        print "XAuto:",type(XAuto),XAuto.shape

        softThresholdIn = XtruewithNoise - XAuto
        softThresholdIn = np.reshape(softThresholdIn, (len(softThresholdIn),19200))
        N = soft_threshold(lamda,softThresholdIn)
        #N = N.reshape([-1, side1, side2, 1])
        print("Iteration NUmber is : ",i)
        print ("NUmber of non zero elements  for N,lamda",np.count_nonzero(N),lamda)
        print ( "The shape of N", N.shape)
        print ( "The minimum value of N ", np.amin(N))
        print ( "The max value of N", np.amax(N))


    return N


def visualise_anamolies_detected(testX,noisytestX,decoded,N,best_top10_keys,worst_top10_keys,lamda):


    #Display the decoded Original, noisy, reconstructed images

    img = np.ndarray(shape=(side1*3, side2*10))
    print "img shape:",img.shape

    for i in range(10):
        row = i // 10 * 3
        col = i % 10
        img[side1*row:side1*(row+1), side2*col:side2*(col+1)] = np.transpose(np.reshape(testX[best_top10_keys[i]].transpose(),(160, 120)))
        img[side1*(row+1):side1*(row+2), side2*col:side2*(col+1)] = np.transpose(np.reshape(np.asarray(decoded[best_top10_keys[i]]).transpose(),(160, 120)))
        img[side1*(row+2):side1*(row+3), side2*col:side2*(col+1)] = np.transpose(np.reshape(N[best_top10_keys[i]].transpose(),(160, 120)))


    img *= 255
    img = img.astype(np.uint8)

    #Save the image decoded
    print("\nSaving results for best after being encoded and decoded: @")
    print(basepath+'/best/')
    io.imsave(basepath+'/best/'+str(lamda)+'salt_p_denoising_dae_decode.png', img)

    #Display the decoded Original, noisy, reconstructed images for worst
    img = np.ndarray(shape=(side1*3, side2*10))
    for i in range(10):
        row = i // 10 * 3
        col = i % 10
        img[side1*row:side1*(row+1), side2*col:side2*(col+1)] = np.transpose(np.reshape(testX[worst_top10_keys[i]].transpose(),(160, 120)))
        img[side1*(row+1):side1*(row+2), side2*col:side2*(col+1)] = np.transpose(np.reshape(np.asarray(decoded[worst_top10_keys[i]]).transpose(),(160, 120)))
        img[side1*(row+2):side1*(row+3), side2*col:side2*(col+1)] = np.transpose(np.reshape(N[worst_top10_keys[i]].transpose(),(160, 120)))


    img *= 255
    img = img.astype(np.uint8)

    #Save the image decoded
    print("\nSaving results for worst after being encoded and decoded: @")
    print(basepath+'/worst/')
    io.imsave(basepath+'/worst/'+str(lamda)+'salt_p_denoising_dae_decode.png', img)


    return


def prepare_fgbg_restraurantData():

    mat_fg_bg_restaurant = loadmat('/Users/raghav/Documents/Uni/KDD-2017/sample_autoencoder/autoencode_softthreshold/DRMF_data/fgbg_restaurant200.mat')
    mat = mat_fg_bg_restaurant
    images = mat.values()

    imgs = mat['imgs']
    print "imgs shape:",imgs.shape

    return imgs



# Prepare data with anamolies defines as per image_and_anamolies
X_1= prepare_fgbg_restraurantData()
X = X_1.reshape([-1, side1, side2, 1])

# Prepare a noisy dataset currently
noise_factor = 0.0
#XnoisyX = add_Salt_Pepper_Noise(X, noise_factor)

side1 = 120
side2 = 160
channel1 = 200
d = 19200
print X.shape
mue = 0.1
N_to_costfunc = np.zeros((200,d ))
#print("Passing  the value of Nvar at...",N_var)
lamda_in_cost = 0.01

# # Define the convoluted ae architecture
# net = tflearn.input_data(shape=[None, d])
# #net = tflearn.fully_connected(net, 256)
# hidden_layer = tflearn.fully_connected(net, nb_feature)
# #net = tflearn.fully_connected(hidden_layer, 256)
# decoder = tflearn.fully_connected(hidden_layer, d, activation='sigmoid')


# Define the convoluted ae architecture another hidden layer
encode_1 = tflearn.input_data(shape=[None, side1,side2,1])
encode_2 = tflearn.fully_connected(encode_1, 256)
hidden_layer = tflearn.fully_connected(encode_2, nb_feature)
decode_1 = tflearn.fully_connected(hidden_layer, 256)
decoder_layer = tflearn.fully_connected(decode_1, d,activation="sigmoid")
decoder = tflearn.reshape(decoder_layer, (-1, side1, side2, 1))




# net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
#                          loss='mean_square', metric=None)

net = tflearn.regression_RobustAutoencoder(decoder,mue,hidden_layer,decoder_layer, optimizer='adam', learning_rate=0.001,
                         loss='rPCA_autoencoderLoss', metric=None,name="vanilla_autoencoder")
model = tflearn.DNN(net, tensorboard_verbose=0)




#define lamda set
lamda_set = [ 0.0,0.01,0.1,0.5,1.0, 10.0, 100.0]
mue = 0.1

# outer loop for lamda
for l in range(0,len(lamda_set)):
    # Learn the N using softthresholding technique
    N =  0
    lamda = lamda_set[l]
    N = compute_softhreshold(X,N,lamda,X)

    N = np.reshape(N, (len(N),19200))

    #Predict the conv_AE autoencoder output
    decoded = model.predict(X)

    #compute MeanSqared error metric
    decoded = np.reshape(decoded, (len(decoded),19200))
    compute_mse(X_1,decoded,lamda)

    # rank the best and worst reconstructed images
    [best_top10_keys,worst_top10_keys]=compute_best_worst_rank(X_1,decoded)

    #Visualise the best and worst ( image, BG-image, FG-Image)
    visualise_anamolies_detected(X_1,X_1,decoded,N,best_top10_keys,worst_top10_keys,lamda)



# plotting the mean precision score
print("\n Saving the Mean square error Score ((Xclean, Xdecoded):")
fig1_mean_square_error=plt.figure(figsize=(8,5))
plt.xlabel("CAE-Denoiser")
plt.ylabel("Mean- Sq Error")
print("\n Mean square error Score ((Xclean, Xdecoded):")
ascending_keys = sorted(mean_square_error_dict, key=mean_square_error_dict.get, reverse=False)
for item in ascending_keys:
    print " Ascending order lamda and mse printing...."
    print item, mean_square_error_dict[item]

for k,v in mean_square_error_dict.iteritems():
    print k,v

# basic plot
data = mean_square_error_dict.values()
plt.boxplot(data)
fig1_mean_square_error.savefig(basepath+'_mean_square_error.png')


