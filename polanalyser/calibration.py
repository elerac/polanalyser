import numpy as np
from numpy import diag
from numpy import dot
from numpy import zeros
from scipy.linalg import svd

# SVD pseudo inverse, sets all diagonals to zero except first k: 9 or 16 elements for a 3x3 or 4x4 matrix, respectively, 
#	3 or 4 for Stokes vector
def svdpinv(I: List[np.ndarray], k) -> np.ndarray::
	# svd, s is reciprocated and V is transposed
	UT, s, V = svd(I, full_matrices=False)
	
	# set all diagonal values to zero except first k elements
	s[k:] = 0

	# get the diagonal matrix for s
	S = diag(s)

	# return the reconstructed matrix
	return UT.dot(S.dot(V))
  
# calculate Mueller calibration matrix W. M is the number of reference materials used. k is the number of singular values to keep.
def calcW(I: List[np.ndarray], PSR: List[np.ndarray], M, k) -> np.ndarray:
	# Check the shape of the input mueller matrices I = [len, xpix, ypix, n], W = [4, 4, n]
	if len(I[0,0,0]) != len(PSR[0]):
		raise ValueError(f"The values of n must be equal, I = {len(I[0,0,0])} PSR = {len(PSR[0, 0])}")
		
	# Convert ArrayLike object to np.ndarray
	I = np.array(I)  # (len, *)
	PSR = np.array(PSR)  # (len, 3, 3) or (len, 4, 4)
	
	I = np.moveaxis(I, 0, -1)  # (*, len)
	PSR = np.moveaxis(PSR, 0, -1)  # (*, len)
	
	# split PSR and I into M reference columns  
	PSR = np.split(PSR, M) # [16xM]
	I = np.split(I, M) # [NxM]
	
	Ihat = svdpinv(I.T, k)
	return np.tensordot(PSR, Ihat, axes=(1, -1)) # W[16xN] = PSR[16xM] * I^-1T[MxN]?
  
# calculate the Meuller matrix using W, MShape = (3,3) or (4,4)
def calcM(I: List[np.ndarray], W: List[np.ndarray], MShape) -> np.ndarray:
	I = np.moveaxis(I, 0, -1)  # (*, len)
	M = np.tensordot(W, I, axes=(1, -1))
	M = np.moveaxis(M, 0, -1)  # (*, 9) or (*, 16)
	M = np.reshape(M, (*M.shape[:-1], *MShape))  # (*, 3, 3) or (*, 4, 4)
	return M
  
# calculate the Stokes vector using W
def calcS(I: List[np.ndarray], W: List[np.ndarray]) -> np.ndarray:
	I = np.moveaxis(I, 0, -1)  # (*, len)
	S = np.tensordot(W, I, axes=(1, -1))
	S = np.moveaxis(M, 0, -1)  # (*, 3) or (*, 4)
	return S
  
# save calibration W matrix to a CSV file
def saveW(W: List[np.ndarray]):
	np.savetxt('WCal.csv', W, delimiter=',')
	
# load calibration W matrix from a CSV file
def loadW() -> np.ndarray: 
	return np.loadtxt('WCal.csv', delimiter=',')
