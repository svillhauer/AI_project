import cv2
import numpy as np
from cv2.cuda import GpuMat
from skimage import img_as_ubyte

class ImageMatcherGPU:
    def __init__(self, matchThreshold=.75):
        self.matchThreshold = matchThreshold
        self._reset_()
        
        # Initialize CUDA SIFT and matcher
        self.sift = cv2.cuda.SIFT_create()
        self.matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)

    def _reset_(self):
        self.gpu_ref = None
        self.gpu_cur = None
        self.kp_ref = None
        self.kp_cur = None
        self.des_ref = None
        self.des_cur = None
        self.hasFailed = True

    def define_images(self, img_ref, img_cur):
        """Process images on GPU"""
        self._reset_()
        
        # Convert to GPU Mats
        self.gpu_ref = GpuMat()
        self.gpu_cur = GpuMat()
        
        # Upload to GPU (assuming uint8 input)
        self.gpu_ref.upload(cv2.cvtColor(img_ref, cv2.COLOR_RGB2GRAY))
        self.gpu_cur.upload(cv2.cvtColor(img_cur, cv2.COLOR_RGB2GRAY))

    def _get_sift_gpu(self, gpu_image):
        """GPU-accelerated SIFT feature extraction"""
        kp_gpu, des_gpu = self.sift.detectAndCompute(gpu_image, None)
        return cv2.cuda_SIFT.downloadKeypoints(self.sift, kp_gpu), des_gpu

    def _get_matches_gpu(self, des_ref, des_cur):
        """GPU-accelerated feature matching"""
        # Match descriptors on GPU
        matches = self.matcher.knnMatchAsync(des_ref, des_cur, k=2)
        
        # Process matches on CPU
        matches = matches.download()
        good = []
        for m,n in matches:
            if m.distance < self.matchThreshold * n.distance:
                good.append(m)
        return np.array([[m.queryIdx, m.trainIdx] for m in good])

    def estimate(self):
        """GPU-accelerated motion estimation pipeline"""
        # Extract features on GPU
        self.kp_ref, self.des_ref = self._get_sift_gpu(self.gpu_ref)
        self.kp_cur, self.des_cur = self._get_sift_gpu(self.gpu_cur)
        
        if self.des_ref.empty() or self.des_cur.empty():
            self.hasFailed = True
            return False
            
        # Match on GPU
        self.theMatches = self._get_matches_gpu(self.des_ref, self.des_cur)
        
        # Rest of your existing motion estimation logic
        if len(self.theMatches) < 15:  # Adjust threshold as needed
            self.hasFailed = True
            return False
            
        # Convert keypoints to GPU arrays
        ref_pts = np.array([kp.pt for kp in self.kp_ref], dtype=np.float32)
        cur_pts = np.array([kp.pt for kp in self.kp_cur], dtype=np.float32)
        
        # GPU-accelerated homography estimation
        H, mask = cv2.cuda.findHomography(
            cv2.cuda_GpuMat(ref_pts), 
            cv2.cuda_GpuMat(cur_pts), 
            cv2.RANSAC
        )
        
        self.theMotion = H if H is not None else np.eye(3)
        self.hasFailed = H is None
        
        return not self.hasFailed
