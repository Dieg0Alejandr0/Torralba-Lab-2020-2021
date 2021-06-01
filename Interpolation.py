from tqdm import tqdm
import numpy as np
import PIL 
from PIL import Image
import imageio
import argparse
import random
import pickle

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

def interpolations(model, vectors, folder, steps=100):
    """
    Given a generator model, the beginning and end vector of our path, the 
    folder we wish to save our interpolations, and the number of steps in
    interpolations, we create the latent space interpolation between two
    latent space vectors representative of two images. 
    
    Input(s):
    *model(torch.nn.Module): The generator model whose latent space is of
    interest.
    
    *vectors(2-tuple of torch.tensors): Two latent space vectors which represent
    the endpoints of our desired interpolation.
    
    *folder(str): The directory where we wish to save the images that constitute
    our interpolation.
    
    *steps(int): The number of steps we take in our interpolation, i.e. 
    between the two torch.tensors in vectors.
    """
    
    (z_0, z_f), c = vectors, None #Interpolate on w 
    for i in range(steps+1):
        
        #Interpolation
        z_i = (1 - i/steps)*z_0 + (i/steps)*z_f
        
        #Use smooth latent-space mapping to create image
        w_i = model.mapping(z_i, c, truncation_psi=0.5, truncation_cutoff=8)
        image = model.synthesis(w_i, noise_mode='const', force_fp32=True)
        
        #We save the image
        directory = f'./interpolations/{folder}/{i:04d}.jpg'
        utils.save_image(image, directory, nrow=10, normalize=True, range=(-1, 1))
        
        
        
def make_gif(model, input_folder, output_folder, steps, name):
    """
    Given a generator model, the beginning and end vector of our path, the 
    folder we wish to save our interpolations, and the number of steps in
    interpolations, we create a GIF of the latent space interpolation 
    between two latent space vectors representative of two images. 
    
    Input(s):
    *model(torch.nn.Module): The generator model whose latent space is of
    interest.
    
    *input_folder(str): The directory consisting of images we wish to create
    a GIF out of.
    
    *output_folder(str): The directory where we wish to save our 
    constructed GIF.
    
    *steps(int): The number of steps we take in our interpolation, i.e. 
    between the two torch.tensors in vectors.
    
    *name(str): The desired name of the output GIF.
    """
    
    #The beginning and end points of our interpolations, and our lack of classes
    z_0, z_f, c = torch.randn([1, model.z_dim]).cuda(3), torch.randn([1, model.z_dim]).cuda(3), None

    #We create our interpolations in the folder "./interpolations/{data_folder}"
    interpolations(model, (z_0, z_f), input_folder, steps=steps)
    
    #We create our GIF from interpolations created in the line above
    images = []
    for i in range(steps+1):
        images.append(imageio.imread(f'./interpolations/{input_folder}/{i:04d}.jpg'))
    imageio.mimsave(f'./interpolations/GIF/{output_folder}/{name}.gif', images, duration=1/100000)
    
    
    
def create_gif_matrix(prefix, outdir, N):
    """
    Given the prefix of GIF names, a desired output
    directory, and desired GIF grid size, we output 
    a GIF which is a grid of previously made GIFs.
    
    Input(s):
    *prefix(str): The prefix of previously made GIFs
    
    *outdir(str): The directory we wish to save our 
    output GIF in.
    
    *N(int): The desired length of the grid in our
    output GIF.
    """
    
    #Collect all desired GIFs and find the number of frames of
    #the output GIF.
    gifs = [imageio.get_reader(f'{prefix}{i:03d}.gif') for i in range(N*N)]
    frames = min([gif.get_length() for gif in gifs])
    
    #We define the duration of our GIF and begin creating our 
    #desired gif.
    kargs = {'duration':1e-27}
    output = imageio.get_writer(f'{outdir}_total.gif')

    
    #We create our grid of GIFs
    for frame in range(frames):
        
        #We get the next frame of each GIF collected
        data = [gif.get_next_data() for gif in gifs]
        
        #We fill each entry in our grid
        for i in range(N):
            for j in range(N):
                
                #We format our row properly
                if (j == 0):
                    row = data[4*i]
                else:
                    row = np.hstack((row, data[4*i + j]))
            
            #Add row to our grid
            if (i == 0):
                new = row
            else:
                new = np.vstack((new, row))
        
        #Add frame of grid to grid GIF.
        output.append_data(new)
    
    #Properly close GIFs
    for gif in gifs:
        gif.close()
        
    output.close()
    
    
def image_to_vector(image):
    """
    Given an image opened with PIL, we output
    its array of values in a format we desire.
    
    Input(s):
    *image(PIL.Image.Image): An image opened by PIL
    """
    return np.matrix.flatten( np.array(image) )



def load_dataset(dataset, N, M):
    """
    Given the directory of a dataset, and the 
    dimensions of a matrix (N, M), we return a matrix
    of said dimensions which represents the values of
    the images in our given dataset.
    
    Input(s):
    *dataset(str): The directory of our dataset of 
    interest.
    
    *N(int): The number of rows in our output matrix.
    
    *M(int): The number of columns in our output matrix.
    
    Output(s):
    *data(np.array): The values that represent that images
    in our dataset of interest.
    """
    
    #We build matrix row by row
    data = np.zeros((N,M))
    for i in range(N):
        data[i,:] = image_to_vector(Image.open(f'{dataset}/Frame {i}.jpg').resize((32,32)))
    
    return data


def nearest_neighbor(frame, data):
    """
    Given a values of a frame and the values of 
    an entire dataset, we return the image in the
    dataset closest to the given frame.
    
    Input(s):
    *frame(np.array): The values of a frame we are
    interested in.
    
    *data(np.array): The values of the dataset we 
    wish to search.
    """
    
    #Row with smallest L2 norm of distance from frame is closest image
    distance = np.linalg.norm(data-frame, axis=1)
    
    return np.argmin(distance)


def frame_neighbors(gif, data, data_directory, directory, name):
    """
    Given a GIF, dataset, said dataset's directory, an output directory,
    and output name, we return a GIF whose frames are the closest
    neighbors (in the given dataset) to the corresponding frames in 
    the given GIF.
    
    Input(s):
    *gif: The GIF we want to find closest neighbors for.
    
    *data(np.array): The values of our dataset.
    
    *data_directory(str): The directory of our dataset.
    
    *directory(str): The directory we wish to save our output GIF in.
    
    *name(str): The name for our output GIF.
    """
    
    #We find the neighbors of each frame
    neighbors = []
    for frame in range(gif.n_frames):
        gif.seek(frame)
        frame_values = image_to_vector( gif.copy().convert('RGB').resize((32, 32)) )
        neighbors.append(nearest_neighbor(frame_values, data))
    
    #We load the neighbors found
    new_frames = []
    for neighbor in neighbors:
        new_frames.append(imageio.imread(f'{data_directory}/Frame {neighbor}.jpg'))
    
    #Save neighbors into gif
    imageio.mimsave(f'./interpolations/GIF/{directory}/{name}.gif', new_frames, duration=1/100000)
    
    
def get_frame(gif_name, frame):
    """
    Given the name of GIF and a particular frame,
    we return the tensor of said frame in given GIF,
    formatted for LPIPs calculation.
    
    *gif_name(str): The name of our GIF of interest.
    
    *frame(int): The frame of interest.
    """
    
    #Save frame as image
    gif = Image.open(gif_name)
    gif.seek(frame)
    gif = gif.convert('RGB')
    gif.save('interpolations/GIF/miscellaneous/miscellaneous.jpg')
    
    #Properly load frame for LPIPS
    return lpips.im2tensor(lpips.load_image('interpolations/GIF/miscellaneous/miscellaneous.jpg')) 


def interpolation_flips(model, device, steps, vectors, loss, offset=0):
    """
    Given a generator, device, a number of steps, we interpolate 
    on the number of steps given between two said vectors. 
    While we do this we store each step in the W latent space 
    of our generator and find any vectors that correspond to 
    visual flips by using a given loss function.
    
    Input(s):
    *model(torch.nn.Module): Our generator of interest.
    
    *steps(int): The number of steps in our interpolation.
    
    *vectors(int): The ends of our interpolation.
    
    *loss(torch.nn.Module): The loss function by which we 
    determine whether we've reached a visual flip.
    
    *offset(int): An integer that keeps of track of how
    many previous images we've looked at before when we've
    created larger matrices.
    
    Output(s):
    *images(torch.tensor): A matrix representing the values
    of our images.
    
    *flips(list): The rows of our images matrix (i.e. the
    particular images) that correponding to flips.
    """
    
    (z_0, z_f), c = vectors, None
    loss_fn = loss
    
    #We define our matrix of steps, and the rows of our flips
    images = torch.zeros(steps+1, 14*model.z_dim).cuda(device)
    flips = []
    for i in range(steps+1):
        
        #Interpolation
        z_i = (1 - i/steps)*z_0 + (i/steps)*z_f
        
        #Use smooth latent-space mapping to create image
        w_i = model.mapping(z_i, c, truncation_psi=0.5, truncation_cutoff=8)
        image = model.synthesis(w_i, noise_mode='const', force_fp32=True)
        
        #We save the image
        directory = f'./interpolations/GIF/miscellaneous/current.jpg'
        utils.save_image(image, directory, nrow=10, normalize=True, range=(-1, 1))
        
        #Calculate LPIPS distance from previous image and check
        #if current step is a flip.
        if i > 0:
            #We load our images in a proper format for LPIPs usage
            current = lpips.im2tensor(lpips.load_image('interpolations/GIF/miscellaneous/current.jpg')) 
            previous = lpips.im2tensor(lpips.load_image('interpolations/GIF/miscellaneous/previous.jpg')) 
            
            distance = loss_fn.forward(current, previous)
            if distance >= 0.15:
                flips.append(offset+i)
                
        #We save the image
        directory = f'./interpolations/GIF/miscellaneous/previous.jpg'
        utils.save_image(image, directory, nrow=10, normalize=True, range=(-1, 1)) 
            
        #Update our array of steps
        images[i, :] = torch.flatten(w_i)
    
    return images, flips


def sample_path(model, device, steps=100, offset=0):
    """
    Given a model's generator, GPU device, and a number of steps,
    we create a path in z-space and find the flips we see in
    said interpolations.
    
    Input(s):
    *model(torch.nn.Module): Our generator of interest.
    
    *device(str): The GPU device for our tensors.
    
    *steps(int): The number of steps we take in our interpolation.
    
    *offset(int): An integer that keeps of track of how
    many previous images we've looked at before when we've
    created larger matrices.
    
    Output(s):
    *(torch.tensor, list): A matrix of our image values in W space 
    and the images which corresponds to visual flips.
    """
    
    #The beginning and end points of our interpolations, and our lack of classes
    z_0, z_f, c = torch.randn([1, model.z_dim]).cuda(device), torch.randn([1, model.z_dim]).cuda(device), None
    
    #We define our loss function
    loss_fn = lpips.LPIPS(net='alex')
    
    #We then return the output of interpolations and their visual flips.
    return interpolation_flips(model, device, steps, (z_0, z_f), loss_fn, offset)