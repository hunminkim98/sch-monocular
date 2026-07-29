import cv2
import numpy as np
from .base_matcher import BaseMatcher


class CV2SIFTMather(BaseMatcher):
    def __init__(self, args=None):
        super().__init__()
        self.sift = cv2.SIFT_create()

    def match_np(self, img0, img1):
        # image를 grayscale로 변환합니다.
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # keypoint와 descriptor를 구합니다.
        kp0, des0 = self.sift.detectAndCompute(gray0, None)
        kp1, des1 = self.sift.detectAndCompute(gray1, None)

        # SIFT에 적합한 FLANN matcher로 descriptor를 matching합니다.
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des0, des1, k=2)

        # Lowe ratio test를 통과한 match를 저장합니다.
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            print(
                f"Warning: Only {len(good_matches)} matches found, which might not be enough for reliable pose estimation"
            )

        # matching된 point 좌표를 추출합니다.
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        return pts0, pts1


class CV2ORBMather(BaseMatcher):
    def __init__(self, args=None):
        super().__init__()
        self.orb = cv2.ORB_create()
        self.num_matches = 1024

    def match_np(self, img0, img1):
        # image를 grayscale로 변환합니다.
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        # 기본 ORB 방식
        kp0, des0 = self.orb.detectAndCompute(gray0, None)
        kp1, des1 = self.orb.detectAndCompute(gray1, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des0, des1)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[: self.num_matches]

        # match가 충분한지 확인합니다.
        if len(good_matches) < 8:
            print(
                f"Warning: Only {len(good_matches)} matches found, which might not be enough for reliable pose estimation"
            )
            # 가능한 경우 match를 더 추가합니다.
            if len(matches) > len(good_matches):
                good_matches = matches[: min(100, len(matches))]

        # matching된 point 좌표를 추출합니다.
        pts0 = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        return pts0, pts1
