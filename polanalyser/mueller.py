import numpy as np

def calcMueller(images, radians_light, radians_camera):
    cos_light  = np.cos(2*radians_light)
    sin_light  = np.sin(2*radians_light)
    cos_camera = np.cos(2*radians_camera)
    sin_camera = np.sin(2*radians_camera)
    A = np.array([np.ones_like(radians_light), cos_light, sin_light, cos_camera, cos_camera*cos_light, cos_camera*sin_light, sin_camera, sin_camera*cos_light, sin_camera*sin_light]).T
    A_pinv = np.linalg.inv(A.T @ A) @ A.T #(9, depth)
    img_mueller = np.tensordot(A_pinv, images, axes=(1,2)).transpose(1, 2, 0) #(height, width, 9)
    #height, width, _ = images.shape
    #img_mueller_2D = np.reshape(img_mueller, (height, width, 3, 3))
    return img_mueller
