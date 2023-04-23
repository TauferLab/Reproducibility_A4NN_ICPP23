import os
import pandas as pd 
import numpy as np
import torch
import multiprocessing as mp
from torch.utils.data import Dataset
from torchvision import transforms
import ctypes
from utils import get_image_dim
#from skimage import import
from PIL import Image

# if regression == False, then getItem will only return the protein conformation for classification of conformation
# if regression == True, then getItem returns protein orientation angles for regression, as well as the protein conformations to inform the regression.
# TODO: setup regression of angles to only regress on all proteins with a given conformation. ideally, we will have a separate NN to predict angles for each conformation type.
# data should be separated into training data (stored in <folder_path>/images/trainset/) and testing data (stored in <folder_path>/images/testset/). We use an 80%/20% split with 80% of the data for each class put in the trainset and 20% of the data put in the testset.

class ProteinDiffDataset(Dataset):
    def __init__(self, folder_path, train=True, regression=False, scale_180=True): #, root_dir, transform=None):        
        self.regression = regression
        self.scale_180 = scale_180
        if self.scale_180:
            print("Removing psi symmetry by moving to [0, 180] from [0, 360]")

        if train == True:
            #trainset 1n0u has 31754 samples
            #trainset 1n0vc has 31754 samples
            self.data_path = os.path.join(folder_path,"images/trainset/")
        else:
            #testset 1n0u has 7938 samples
            #testset 1n0vc has 7938 samples
            self.data_path = os.path.join(folder_path,"images/testset/")

        #Get list of image paths
        self.image_list = self.__getDataFiles__(self.data_path)
        
        #Calculate length
        self.data_len = len(self.image_list)
        nb_samples = self.data_len 
        
        if train == True:
            print("num training samples: {}".format(nb_samples))
        else: 
            print("num testing samples: {}".format(nb_samples))

        # Open one image to get dimensions from
        data_shape, c = get_image_dim(self.data_path, EXT='tiff')
        w, h = data_shape

        shared_array_base = mp.Array(ctypes.c_float,nb_samples*c*h*w)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(nb_samples,c,h,w)
        self.shared_array = torch.from_numpy(shared_array)
        
        # Initializing an array to store a cached version of the protein conformations
        # We need this info for classification or regression -- when we do regression of orientation, we may wish to use info about protein conformation
        conformation_array_base = mp.Array(ctypes.c_long,nb_samples)
        self.conformation_array = np.ctypeslib.as_array(conformation_array_base.get_obj())
        
        if self.regression == True:
            # initializing an array to cache protein orientation angle values (don't need this if we aren't doing regression)
            labels_array_base = mp.Array(ctypes.c_float,nb_samples*3)
            labels_array = np.ctypeslib.as_array(labels_array_base.get_obj())
            self.labels_array = labels_array.reshape(nb_samples,3)

            # create a dataframe that stores orientation angle values; only need this if we're doing regression
            labels_path= os.path.join(folder_path, "angles_list.txt")
            self.data_labels = pd.read_csv(labels_path, sep=' ', header=0, index_col='id')
            #print(self.data_labels)
        # we've initialized caches, but they are empty. we should not yet use data from the cache in training
        self.use_cache = False

        #Transforms
        self.to_tensor = transforms.ToTensor()

    # this function is used to set a flag once the cache has been populated to prompt use of cache in the training. 
    # should populate the cache during first epoch of training and set use_cache to true with this function after the first epoch of training is complete
    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
        print("use cache set to: "+str(use_cache))

    def __name__(self):
        return "ProteinDiffraction"

    def __getitem__(self, index):
        if not self.use_cache:
            # Get image name
            single_image_path = self.image_list[index]
            #image_info will contain the pairs: (conf_label, id#.tiff)
            image_info = single_image_path.split("ef2_")[1].split("_ptm")
            conf_label, image_id = image_info[0], int(image_info[1].split(".")[0])

            #print(image_id)
            #print(conf_label)

            #open image
            im = Image.open(single_image_path) 
            im_array = np.array(im)
            c = len(im.getbands())
            
            #print("array = "+str(im_array))        
            
            # Resizing to include image channels. The PD images are greyscale, so there's just one channel (expected imarray size is now 1x127x127)
            # If in the future we have color images, that would be 3 image channels (r,g,b), so we'd need expand_dims(im_array, (0, 1, 2))
            im_array = np.expand_dims(im_array, axis=0) if c == 1 else np.expand_dims(im_array, axis=(0, 1, 2))
            
            # Any desired preprocessing operations may be applied here to the numpy array

            # Transform image numpy array to a tensor
            im_tensor = torch.from_numpy(im_array).float()

            # Saving the image tensor to the array we created to store cached data
            self.shared_array[index] = im_tensor
            

            # Next we cache the protein conformations and/or orientation angles in arrays  
            # If we are not doing regression, we only need to cache and return the conformation
            if self.regression == False:
                
                # We assign numbers to the conformation classes: 1n0u -> 1, and 1n0vc -> 2; these numbers are what we will cache
                if conf_label == "1n0u":
                    conformation = 0
                    #print("label = "+str(label))
                elif conf_label == "1n0vc":
                    conformation = 1
                    #print("label = "+str(label))
                else:
                    # throw error -- unknown label
                    raise ValueError("unknown conformation")

                # Caching conformation
                self.conformation_array[index] = conformation
                
                image_tensor=self.shared_array[index]
                conf = self.conformation_array[index]
                return image_tensor, conf

            # Otherwise, we are doing regression of protein angles
            # In this case, we need to cache and return the protein conformations and orientation angles
            else:

                # The image ids repeat for the diff conformations. i.e. Conformation 1 has images numbered 1 to x, and conformation 2 again has images numbered 1 to x. 
                # In order to get the correct orientation angles, we need to find the image whose id number and conformation type match the case we're considering.
                # We first select as candidates all images with the same image_id
                candidates = len(self.data_labels.at[image_id,"conformation"])

                # Will save index of the correct figure in currentFig. Init it with -1 (a num that's not a valid index); this way we can detect if an index is never assigned.
                currentFig = -1

                for j in [0,candidates-1]:
                    #print(self.data_labels.at[image_id,"conformation"])
                    conform_df = self.data_labels.at[image_id,"conformation"]
                    #print(conform_df.iat[0])
                    #if conf_label == str(self.data_labels.at[image_id,"conformation"][j]):
                    if conf_label == str(conform_df.iat[j]):
                        currentFig=j
            
                if conf_label == "1n0u":
                    conformation = 0
                    #print("label = "+str(label))
                elif conf_label == "1n0vc":
                    conformation = 1
                    #print("label = "+str(label))
                else:
                    # throw error -- unknown label
                    raise ValueError("unknown conformation")
                    #print("unknown label")
            
                self.conformation_array[index] = conformation
                # .at grabs an entry in the dataframe using the NAMES of row/column. .iat grabs an entry using the NUMBER (or index) of row/column
                phi = (self.data_labels.at[image_id,"phi"]).iat[currentFig]
                theta = (self.data_labels.at[image_id,"theta"]).iat[currentFig]
                psi = (self.data_labels.at[image_id,"psi"]).iat[currentFig]

                if self.scale_180:
                    if psi > 180:
                        psi = psi - 180

                #saving the angle values to the cache arrray 
                self.labels_array[index] = (phi,theta,psi)
       
                #print("phi = "+str(phi))
                #TODO phi values above are good. something is going wrong with the angle_phi calculation (ditto for theta & psi)
 
                image_tensor=self.shared_array[index]
                conf = self.conformation_array[index]
                angle_phi,angle_theta,angle_psi = self.labels_array[index]
                #print("angle_phi = "+str(angle_phi))
                #print("returning the following: "+str(image_tensor)+" "+str(conf)+" "+str(angle_phi)+" "+str(angle_theta)+" "+str(angle_psi)) 
                return image_tensor, conf, angle_phi, angle_theta, angle_psi
    
        else:
            image_tensor=self.shared_array[index]
            conf = self.conformation_array[index]

            if self.regression == True:
                angle_phi,angle_theta,angle_psi = self.labels_array[index]
                return image_tensor, conf, angle_phi, angle_theta, angle_psi
            
            return image_tensor, conf
            #TODO: fill in this stuff...

    #function to find number of images in dataset
    def __len__(self):
        return self.data_len

    # A helper function that creates a list of all tiff files in a given directory. (PD datafiles have tiff extensions)
    
    def __getDataFiles__(self, DIRECTORY, EXT='tiff'):
        datafiles=[]
        for item in os.scandir(DIRECTORY):
            if item.name.endswith("."+EXT) and not item.name.startswith("."):
                datafiles.append(item.path)
                
        return datafiles

    
if __name__=="__main__":
    pdData = ProteinDiffDataset('/lustre/xg-cis230020/sw/1e15/')
    
