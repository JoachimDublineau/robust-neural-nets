import numpy as np
def generate_one_pixel_attacks(model, images, labels, eps=0):
    n,i,j,k = images.shape
    x_index = np.round(np.random.normal(i/2, int(4*i/32), n))
    y_index = np.round(np.random.normal(j/2, int(4*j/32), n))
    # z_index = np.random.randint(0, 3, n)
    attacks = []
    for p in range(n):
        index = int(x_index[p])
        x = (index if index>=0 and index<i else i-1)
        index = int(y_index[p])
        y = (index if index>=0 and index<j else j-1)
        image = images[p]
        image[x,y,:] *= eps 
        attacks.append(image)
    return attacks



    