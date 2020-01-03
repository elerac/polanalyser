import numpy as np

# Polarization 101
# I(upsilon) = (Imax+Imin)/2 + (Imax-Imin)/2*cos(2*upsilon-2*phi)
# Degree of polarization (0~1):DoP = (Imax-Imin)/(Imax+Imin)
#
# Stokes vector
# S = [S0, S1, S2]

class GetStokesParameters:
    """
    Calculate polarization parameters from intensity and hold parameters.
    """
    def __init__(self, img_intensity, upsilon):
        """
        Parameters
        ----------
        img_intensity : array-like list
            camera image list
        upsilon : array-like
            polarizer angle
        """
        self.height, self.width, self.depth = img_intensity.shape
        
        #Calculate Stokes vector
        self.S = self.calculate_stokes(img_intensity, upsilon)

        self.S0 = self.I = S0 = self.S[:,:,0]
        self.S1 = self.Q = S1 = self.S[:,:,1]
        self.S2 = self.U = S2 = self.S[:,:,2]

        norm_S1_S2 = np.sqrt(S1**2+S2**2)
        self.Imax = self.max = (S0+norm_S1_S2)/2.0
        self.Imin = self.min = (S0-norm_S1_S2)/2.0
        self.phi = self.AoLP = np.mod(0.5*np.arctan2(S2, S1), np.pi)
        self.DoP = self.DoLP = norm_S1_S2/S0

    def calculate_stokes(self, img_intensity, upsilon):
        height, width, depth = img_intensity.shape
        A = 0.5*np.array([np.ones(self.depth), np.cos(2*upsilon), np.sin(2*upsilon)]).T #(depth, 3)
        A_pinv = np.dot(np.linalg.inv(np.dot(A.T, A)),  A.T) #(3, depth)

        S = np.tensordot(A_pinv, img_intensity, axes=(1,2)).transpose(1, 2, 0) #(height, width, 3)
        return S
    
    def intensity(self, upsilon):
        Imax = self.max
        Imin = self.min
        phi = self.phi
        return (Imax+Imin)/2 + (Imax-Imin)/2*np.cos(2*upsilon-2*phi)

class GetMullerParameters:
    def __init__(self, I, upsilon_l, upsilon_c):
        self.M = M = self.calculate_muller(I, upsilon_l, upsilon_c)
        self.m11 = M[:,:,0,0]
        self.m12 = M[:,:,0,1]
        self.m13 = M[:,:,0,2]
        self.m21 = M[:,:,1,0]
        self.m22 = M[:,:,1,1]
        self.m23 = M[:,:,1,2]
        self.m31 = M[:,:,2,0]
        self.m32 = M[:,:,2,1]
        self.m33 = M[:,:,2,2]
        
    def calculate_muller(self, I, upsilon_l, upsilon_c):
        height, width, depth = I.shape
        A = np.array([np.ones(depth), np.cos(2*upsilon_l), np.sin(2*upsilon_l), np.cos(2*upsilon_c), np.cos(2*upsilon_c)*np.cos(2*upsilon_l), np.cos(2*upsilon_c)*np.sin(2*upsilon_l), np.sin(2*upsilon_c), np.sin(2*upsilon_c)*np.cos(2*upsilon_l), np.sin(2*upsilon_c)*np.sin(2*upsilon_l)]).T
        A_pinv = np.dot(np.linalg.inv(np.dot(A.T, A)),  A.T) #(9, depth)
        
        _M = np.tensordot(A_pinv, I, axes=(1,2)).transpose(1, 2, 0) #(height, width, 9)
        M = np.reshape(_M, (height, width, 3, 3))
        return M

def main():
    import cv2
    print("<GetStokesParameters test>")
    height = 1
    width = 1
    observe_num = 4
    Imax = 2
    Imin = 1
    phi = np.pi/3
    upsilon = np.linspace(0, np.pi, observe_num)

    I_value = (Imax+Imin)/2 + (Imax-Imin)/2*np.cos(2*upsilon-2*phi)
    img_I = cv2.merge([np.ones((height, width))*val for val in I_value.tolist()])
    print("Input parameters")
    print("Imax:", Imax)
    print("Imin:", Imin)
    print("phi :", phi)
    print()

    polarization = GetStokesParameters(img_I, upsilon)
    
    print("Output parameters")
    print("Imax:\n", polarization.max)
    print("Imin:\n", polarization.min)
    print("phi :\n", polarization.phi)
    print()

    
    print("<GetMullerParameters test>")
    height = 1
    width = 1
    observe_num = 9
    M = np.random.rand(3, 3)
    upsilon_l = np.random.rand(observe_num) * np.pi
    upsilon_c = np.random.rand(observe_num) * np.pi
    
    A = np.array([np.ones(observe_num), np.cos(2*upsilon_l), np.sin(2*upsilon_l), np.cos(2*upsilon_c), np.cos(2*upsilon_c)*np.cos(2*upsilon_l), np.cos(2*upsilon_c)*np.sin(2*upsilon_l), np.sin(2*upsilon_c), np.sin(2*upsilon_c)*np.cos(2*upsilon_l), np.sin(2*upsilon_c)*np.sin(2*upsilon_l)]).T
    I_value = np.dot(A, M.flatten())
    I_img = cv2.merge([np.ones((height, width))*val for val in I_value.tolist()])
    print("Input parameters")
    print("M:", M)
    print()

    muller = GetMullerParameters(I_img, upsilon_l, upsilon_c)

    print("Output parameters")
    print("M:", muller.M)
    print()

if __name__=="__main__":
    main()


