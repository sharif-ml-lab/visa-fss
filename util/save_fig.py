import matplotlib.pyplot as plt
import matplotlib.image as pltimg

def overlay_img_mask(img, msk, path):
    plt.figure()
    plt.imshow(img, 'gray', interpolation='none')
    plt.imshow(msk, 'gnuplot', interpolation='none', alpha=0.5)
    plt.axis('off')
    plt.savefig(path)
    plt.close()