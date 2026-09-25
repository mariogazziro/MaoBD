import os
from operator import attrgetter
import numpy as np
from numpy import linalg as LA
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import skimage
import skimage.measure
import skimage.morphology
import cv2
import imhandle as imh
import hands_handle as hh

class MaoBD(object):
    which = "RIGHT"
    CMid = 0
    CAd  = 0
    CMed = 0
    CId  = 0
    CPd  = 0
    LMd  = 0
    CPMd = 0
    CMie = 0
    CAe  = 0
    CMee = 0
    CIe  = 0
    CPe  = 0
    LMe  = 0
    CPMe = 0

    @staticmethod
    def vadd(a,b):
        return (a[0]+b[0],a[1]+b[1])

    @staticmethod
    def vsub(a,b):
        return (a[0]-b[0],a[1]-b[1])

    @staticmethod
    def vmul(a,b):
        return (a[0]*b,a[1]*b)

    @staticmethod
    def project(a, b):
        """ project a onto b
            formula: b(dot(a,b)/(|b|^2))
        """
        abdot = (a[0]*b[0])+(a[1]*b[1])
        blensq = (b[0]*b[0])+(b[1]*b[1])
        temp = float(abdot)/float(blensq)
        c = (b[0]*temp,b[1]*temp)
        return c

    def __init__(self, imgPath, imgName, Which, outPath):
        print(imgName)
        self.which = Which
        imageBGR = cv2.imread(os.path.join(imgPath, imgName))
        if imageBGR is None:
            print('Error: bad image')
            return
        imageBGR = imh._resizeImg(imageBGR, 600.0)
        imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
        seg = imh._skinColorSegmentation(imageRGB)
        binarized = hh.binarize_hand_img(seg)
        edge = cv2.Canny(seg, 50, 100)
        kernel = np.ones((3,3),np.uint8)
        edge = cv2.dilate(edge,kernel,iterations = 1)
        if binarized is None or binarized.size == 0:
            print('Error: bad image')
            return
        hand_pts = hh.get_fingertips_and_valleys(binarized, return_center=True)
        if hand_pts is None or len(hand_pts[0]) == 0 or len(hand_pts[1]) == 0:
            print('Error: bad image')
            return
        fingertips, valleys, center = hand_pts
        bin_w_key_pts = np.zeros((binarized.shape[0], binarized.shape[1]), dtype=np.int64)
        bin_w_key_pts[0:binarized.shape[0], 0:binarized.shape[1]] = binarized.copy()
        for pnt in fingertips:
            bin_w_key_pts[max(pnt[0] - 5, 0):pnt[0] + 5, max(pnt[1] - 5, 0):pnt[1] + 5] = 5
        for pnt in valleys:
            bin_w_key_pts[max(pnt[0] - 5, 0):pnt[0] + 5, max(pnt[1] - 5, 0):pnt[1] + 5] = 10
        fingertips_dist_sq = list(map(lambda pnt: ((pnt - np.array(center)) ** 2).sum(), fingertips))
        fingertip_farthest = fingertips[np.argmax(fingertips_dist_sq)]
        #rot_angle = np.degrees(np.pi - hh.aligned_angle(fingertip_farthest - center))
        rot_angle = np.degrees(0)
        bin_rot = scipy.ndimage.rotate(bin_w_key_pts, rot_angle, order=0, reshape=False)
        # Finding center
        med, dt = skimage.morphology.medial_axis(bin_rot, return_distance=True)
        center_rot = (float(np.argmax(dt.ravel()) // dt.shape[1]), float(np.argmax(dt.ravel()) % dt.shape[1]))
        # Acquiring fingertips from rotated image and sorting them by angle
        fingertips_label = skimage.measure.label(bin_rot == 5, connectivity=2)
        fingertips_rot = list(map(attrgetter('centroid'), skimage.measure.regionprops(fingertips_label)))
        fingertips_rot = np.array(fingertips_rot, dtype=np.int64)
        fingertips_angles = np.array(list(map(lambda pnt: hh.aligned_angle(pnt - center_rot), fingertips_rot)))
        fingertips_rot = fingertips_rot[np.argsort(fingertips_angles)]
        # Acquiring valleys from rotated image and sorting them by angle
        valleys_label = skimage.measure.label(bin_rot == 10, connectivity=2)
        valleys_rot = list(map(attrgetter('centroid'), skimage.measure.regionprops(valleys_label)))
        valleys_rot = np.array(valleys_rot, dtype=np.int64)
        valleys_angles = np.array(list(map(lambda pnt: hh.aligned_angle(pnt - center_rot), valleys_rot)))
        valleys_rot = valleys_rot[np.argsort(valleys_angles)]
        if len(fingertips_rot) < 5 or len(valleys_rot) < 4:
            print('Error: Not all key points found')
        else:
            key_points = []
            for i in range(4):
                key_points.extend([fingertips_rot[i], valleys_rot[i]])
            key_points.append(fingertips_rot[4])
            bin_rot_with_lines = np.zeros((bin_rot.shape[0], bin_rot.shape[1], 3), dtype=np.uint8)
            largura = np.zeros((bin_rot.shape[0], bin_rot.shape[1], 3), dtype=np.uint8)
            _max = bin_rot.max()
            _bin_rot_scaled = (bin_rot.astype(np.float64) / max(_max, 1) * 255).astype(np.uint8)
            bin_rot_with_lines[:, :, 0] = bin_rot_with_lines[:, :, 1] = bin_rot_with_lines[:, :, 2] = _bin_rot_scaled
            ci, cj = int(center_rot[0]), int(center_rot[1])
            bin_rot_with_lines[max(ci - 5, 0):ci + 5, max(cj - 5, 0):cj + 5] = 255
            if (self.which == "RIGHT"):
                pnt1 = tuple(key_points[3])
                pnt2 = tuple(key_points[4])
                pnt1 = (pnt1[1], pnt1[0])
                pnt2 = (pnt2[1], pnt2[0])
                A = pnt2
                B = tuple([cj, ci])
                C = pnt1
                AB = self.vsub(B,A)
                AC = self.vsub(C,A)
                AD = self.project(AC,AB)
                D = self.vadd(A,AD)
                cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                self.CMed = LA.norm(AD)
                cv2.line(bin_rot_with_lines, tuple([int(D[0]),int(D[1])]), tuple([cj, ci]), [0, 255, 255])
                O = cj, ci
                DO = self.vsub(O,D)
                self.CPMd = LA.norm(DO)*1.67
                P = self.vmul(DO,1.67)
                P = self.vadd(P,D)
                cv2.line(bin_rot_with_lines, tuple([int(P[0]),int(P[1])]), tuple([cj, ci]), [0, 255, 255])

                pnt1 = tuple(key_points[7])
                pnt2 = tuple(key_points[8])
                pnt1 = (pnt1[1], pnt1[0])
                pnt2 = (pnt2[1], pnt2[0])
                A = pnt2
                B = P
                C = pnt1
                AB = self.vsub(B,A)
                AC = self.vsub(C,A)
                AD = self.project(AC,AB)
                D = self.vadd(A,AD)
                cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                self.CMid = LA.norm(AD)
                K = self.vmul(AD,1.3)
                K = self.vadd(A,K)
                M = K

                pnt1 = tuple(key_points[7])
                pnt2 = tuple(key_points[6])
                pnt1 = (pnt1[1], pnt1[0])
                pnt2 = (pnt2[1], pnt2[0])
                A = pnt2
                B = P
                C = pnt1
                AB = self.vsub(B,A)
                AC = self.vsub(C,A)
                AD = self.project(AC,AB)
                D = self.vadd(A,AD)
                cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                self.CAd = LA.norm(AD)

                pnt1 = tuple(key_points[2])
                pnt2 = tuple(key_points[3])
                pnt1 = (pnt1[1], pnt1[0])
                pnt2 = (pnt2[1], pnt2[0])
                A = pnt1
                B = P
                C = pnt2
                AB = self.vsub(B,A)
                AC = self.vsub(C,A)
                AD = self.project(AC,AB)
                D = self.vadd(A,AD)
                cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                self.CId = LA.norm(AD)
                K = self.vmul(AD,1.25)
                K = self.vadd(A,K)
                N = K

                pnt1 = tuple(key_points[0])
                pnt2 = tuple(key_points[1])
                pnt1 = (pnt1[1], pnt1[0])
                pnt2 = (pnt2[1], pnt2[0])
                A = pnt1
                B = P
                C = pnt2
                AB = self.vsub(B,A)
                AC = self.vsub(C,A)
                AD = self.project(AC,AB)
                D = self.vadd(A,AD)
                cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                self.CPd = LA.norm(AD)

                MN = self.vsub(N,M)
                T = self.vmul(MN,1.50)
                T = self.vadd(M,T)
                TM = self.vsub(M,T)
                Q = self.vmul(TM,1.3)
                Q = self.vadd(T,Q)
                cv2.line(largura, tuple([int(T[0]),int(T[1])]), tuple([int(Q[0]),int(Q[1])]), [255, 255, 255])
                overlap = cv2.bitwise_and(largura, largura, mask=edge)
                kernel = np.ones((5,5),np.uint8)
                overlap = cv2.dilate(overlap,kernel,iterations = 5)
                overlap_label = skimage.measure.label(overlap == 255, connectivity=3)
                overlap_rot = list(map(attrgetter('centroid'), skimage.measure.regionprops(overlap_label)))
                self.LMd = 0
                if (len(overlap_rot)>=2):
                    cv2.line(bin_rot_with_lines, tuple([int(overlap_rot[0][1]),int(overlap_rot[0][0])]), tuple([int(overlap_rot[1][1]),int(overlap_rot[1][0])]), [255, 255, 0])
                    LD =  self.vsub(tuple([int(overlap_rot[0][1]),int(overlap_rot[0][0])]),tuple([int(overlap_rot[1][1]),int(overlap_rot[1][0])]))
                    self.LMd = LA.norm(LD)

                print('%.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f' %(self.CPd,self.CId,self.CMed,self.CAd,self.CMid,self.CPMd,self.LMd))

            else:
                if (self.which == "LEFT"):
                    pnt1 = tuple(key_points[4])
                    pnt2 = tuple(key_points[5])
                    pnt1 = (pnt1[1], pnt1[0])
                    pnt2 = (pnt2[1], pnt2[0])
                    A = pnt1
                    B = tuple([cj, ci])
                    C = pnt2
                    AB = self.vsub(B,A)
                    AC = self.vsub(C,A)
                    AD = self.project(AC,AB)
                    D = self.vadd(A,AD)
                    cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                    self.CMee = LA.norm(AD)
                    cv2.line(bin_rot_with_lines, tuple([int(D[0]),int(D[1])]), tuple([cj, ci]), [0, 255, 255])
                    O = cj, ci
                    DO = self.vsub(O,D)
                    self.CPMe = LA.norm(DO)*1.67
                    P = self.vmul(DO,1.67)
                    P = self.vadd(P,D)
                    cv2.line(bin_rot_with_lines, tuple([int(P[0]),int(P[1])]), tuple([cj, ci]), [0, 255, 255])

                    pnt1 = tuple(key_points[0])
                    pnt2 = tuple(key_points[1])
                    pnt1 = (pnt1[1], pnt1[0])
                    pnt2 = (pnt2[1], pnt2[0])
                    A = pnt1
                    B = P
                    C = pnt2
                    AB = self.vsub(B,A)
                    AC = self.vsub(C,A)
                    AD = self.project(AC,AB)
                    D = self.vadd(A,AD)
                    cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                    self.CMie = LA.norm(AD)
                    K = self.vmul(AD,1.3)
                    K = self.vadd(A,K)
                    M = K

                    pnt1 = tuple(key_points[2])
                    pnt2 = tuple(key_points[1])
                    pnt1 = (pnt1[1], pnt1[0])
                    pnt2 = (pnt2[1], pnt2[0])
                    A = pnt1
                    B = P
                    C = pnt2
                    AB = self.vsub(B,A)
                    AC = self.vsub(C,A)
                    AD = self.project(AC,AB)
                    D = self.vadd(A,AD)
                    cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                    self.CAe = LA.norm(AD)

                    pnt1 = tuple(key_points[5])
                    pnt2 = tuple(key_points[6])
                    pnt1 = (pnt1[1], pnt1[0])
                    pnt2 = (pnt2[1], pnt2[0])
                    A = pnt2
                    B = P
                    C = pnt1
                    AB = self.vsub(B,A)
                    AC = self.vsub(C,A)
                    AD = self.project(AC,AB)
                    D = self.vadd(A,AD)
                    cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                    self.CIe = LA.norm(AD)
                    K = self.vmul(AD,1.25)
                    K = self.vadd(A,K)
                    N = K

                    pnt1 = tuple(key_points[8 - 1])
                    pnt2 = tuple(key_points[8])
                    pnt1 = (pnt1[1], pnt1[0])
                    pnt2 = (pnt2[1], pnt2[0])
                    A = pnt2
                    B = P
                    C = pnt1
                    AB = self.vsub(B,A)
                    AC = self.vsub(C,A)
                    AD = self.project(AC,AB)
                    D = self.vadd(A,AD)
                    cv2.line(bin_rot_with_lines, tuple([int(A[0]),int(A[1])]), tuple([int(D[0]),int(D[1])]), [255, 0, 0])
                    self.CPe = LA.norm(AD)

                    MN = self.vsub(N,M)
                    T = self.vmul(MN,1.50)
                    T = self.vadd(M,T)
                    TM = self.vsub(M,T)
                    Q = self.vmul(TM,1.3)
                    Q = self.vadd(T,Q)
                    cv2.line(largura, tuple([int(T[0]),int(T[1])]), tuple([int(Q[0]),int(Q[1])]), [255, 255, 255])
                    overlap = cv2.bitwise_and(largura, largura, mask=edge)
                    kernel = np.ones((5,5),np.uint8)
                    overlap = cv2.dilate(overlap,kernel,iterations = 5)
                    overlap_label = skimage.measure.label(overlap == 255, connectivity=3)
                    overlap_rot = list(map(attrgetter('centroid'), skimage.measure.regionprops(overlap_label)))
                    self.LMe = 0
                    if (len(overlap_rot)>=2):
                        cv2.line(bin_rot_with_lines, tuple([int(overlap_rot[0][1]),int(overlap_rot[0][0])]), tuple([int(overlap_rot[1][1]),int(overlap_rot[1][0])]), [255, 255, 0])
                        LE =  self.vsub(tuple([int(overlap_rot[0][1]),int(overlap_rot[0][0])]),tuple([int(overlap_rot[1][1]),int(overlap_rot[1][0])]))
                        self.LMe = LA.norm(LE)

                    print('%.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f \t %.2f' %(self.CPe, self.CIe, self.CMee, self.CAe, self.CMie, self.CPMe, self.LMe))
                else:
                    print('Error: Not LEFT or RIGHT defined!')
                    return

            fig = plt.figure()
            imshow(bin_rot_with_lines)
            fig.savefig(os.path.join(outPath, os.path.splitext(imgName)[0] + '.png'), dpi=300)
            plt.close('all')
