import math
import cv2
import numpy as np
import os

import ImageFeatureBase
from SubImageRegion import *

class OCRanalysis:
    def __init__(self):
        self.F_FGcount = 0
        self.F_MaxDistX= 1
        self.F_MaxDistY= 2
        #self.F_AvgDistanceCentroide= 3
        #self.F_MaxDistanceCentroide= 4
        #self.F_MinDistanceCentroide= 5
        #self.F_Circularity = 6
        #self.F_CentroideRelPosX = 7
        #self.F_CentroideRelPosY = 8

    def run(self, img_path, tgtCharRow, tgtCharCol):
        # Extract base path of image
        basePath = os.path.split(img_path)[0]

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        FG_VAL = 0
        BG_VAL = 255
        MARKER_VAL = 127
        thresholdVal = 127

        _, binaryImgArr = cv2.threshold(img, thresholdVal, BG_VAL, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(basePath, "binaryOut.png"), binaryImgArr)

        #define the features to evaluate
        features_to_use = []
        features_to_use.append(ImageFeatureF_FGcount)
        features_to_use.append(ImageFeatureF_MaxDistX)
        features_to_use.append(ImageFeatureF_MaxDistY)

        features_to_use.append(ImageFeatureF_AspectRatio)
        features_to_use.append(ImageFeatureF_FgBgRatio)
        features_to_use.append(ImageFeatureF_VerticalAsym)
        features_to_use.append(ImageFeatureF_HorizontalAsym)


        linked_regions = split_characters(binaryImgArr, width, height, BG_VAL, FG_VAL)

        # display all recognized characters
        # highlighted_img = highlight_letters(binaryImgArr, linked_regions)
        # cv2.imshow("highlighted", highlighted_img)
        #cv2.waitKey(0)
        
        #define the reference character
        #tgtCharRow = 3
        #tgtCharCol = 3
        charROI = linked_regions[tgtCharRow][tgtCharCol]

        # test calculate features
        #print('features of reference character is: ')
        feature_res_arr = calc_feature_arr(charROI, BG_VAL, features_to_use)
        #self.printout_feature_res(feature_res_arr, features_to_use)

        #then normalize
        feature_norm_arr = calculate_norm_arr(linked_regions, BG_VAL, features_to_use)
        #print('NORMALIZED features: ')
        #self.printout_feature_res(feature_norm_arr, features_to_use)

        #now check all characters and test, if similarity with reference letter is given:
        # assuming that hitCount, binaryImgArr, FG_VAL, MARKER_VAL are defined elsewhere globally
        # as they're not defined in the current code provided

        hitCount = 0  # make sure to initialize hitCount

        binary_img_arr = binaryImgArr.copy()
        for i in range(len(linked_regions)):
            for j in range(len(linked_regions[i])):
                img_reg = linked_regions[i][j]
                curr_feature_arr = calc_feature_arr(img_reg, BG_VAL, features_to_use)
                is_target_char = is_matching_char(curr_feature_arr, feature_res_arr, feature_norm_arr)
                if is_target_char:
                    hitCount += 1
                    binary_img_arr = self.mark_region_in_image(binary_img_arr, img_reg, BG_VAL, MARKER_VAL)

        #TODO: printout result image with all the marked letters
        cv2.imshow("markedChars", binary_img_arr)
        cv2.waitKey(0)
        #cv2.imwrite(os.path.join(basePath, "markedChars.png"), binary_img_arr)
        #print('num of found characters is = ' + str(hitCount))

        cv2.destroyAllWindows()
        return hitCount

    def printout_feature_res(feature_res_arr, features_to_use):
        print("========== features =========")
        for i in range(len(features_to_use)):
            print("res of F " + str(i) + ", " + features_to_use[i].description + " is " + str(feature_res_arr[i]))

    def mark_region_in_image(self, in_img_arr, img_region, color_to_replace, tgt_color):
        adjustedColors = 0
        for x in range(img_region.width):
            for y in range(img_region.height):
                if img_region.subImgArr[y][x] == color_to_replace:
                    in_img_arr[y + img_region.startY][x + img_region.startX] = tgt_color
                    adjustedColors+=1
        #print('adjusted colors is ' + str(adjustedColors))
        return in_img_arr

    def printout_feature_res(self, feature_res_arr, features_to_use):
        print("========== features =========")
        for i in range(len(features_to_use)):
            print("res of F", i, ",", features_to_use[i], "is", feature_res_arr[i])

def is_empty_column(in_img, height, col_idx, BG_val):
    for y in range(height):
        if in_img[y][col_idx] != BG_val:
            return False
    return True

def is_empty_row(in_img, width, row_idx, BG_val):
    for x in range(width):
        if in_img[row_idx][x] != BG_val:
            return False
    return True

def split_characters_vertically(row_image, BG_val, FG_val, orig_img, row_start_y):
    char_row_regions_list = []
    height, width = row_image.shape
    current_char_start_x = None

    # iterate over all columns
    for x in range(width):
        if is_empty_column(row_image, height, x, BG_val):
            if current_char_start_x is not None:
                char_width = x - current_char_start_x
                region = SubImageRegion(
                    startX=current_char_start_x,
                    startY=row_start_y,
                    width=char_width,
                    height=height,
                    origImgArr=orig_img
                )
                char_row_regions_list.append(region)
                current_char_start_x = None
        else:
            if current_char_start_x is None:
                current_char_start_x = x

    return char_row_regions_list

def minimize_character_bounding_height(inImg: np.array, region: SubImageRegion, FG_val: int) -> SubImageRegion:
    startYOffset, endYOffset = -1, region.height
    for y in range(region.height):
        if FG_val in region.subImgArr[y][0:region.width]:
            if startYOffset <= -1:
                startYOffset = y
            endYOffset = y
    return SubImageRegion(
        startX=region.startX,
        startY=region.startY + startYOffset,
        width=region.width,
        height=endYOffset - startYOffset + 1,
        origImgArr=inImg
    )

def split_characters(in_img, width, height, BG_val, FG_val):
    char_regions_list = []
    current_row_img = []
    row_start_y = None

    for y in range(height):
        if is_empty_row(in_img, width, y, BG_val):
            if current_row_img:
                # found a full row -> split characters vertically and add it to the result
                row_img = np.array(current_row_img)
                char_row_regions_list = split_characters_vertically(
                    row_img, BG_val, FG_val, in_img, row_start_y
                )
                minized_char_row_regions_list = []
                for region in char_row_regions_list:
                    minized_char_row_regions_list.append(minimize_character_bounding_height(in_img, region, FG_val))
                char_regions_list.append(minized_char_row_regions_list)
                current_row_img = []
                row_start_y = None
        else:
            if row_start_y is None:
                row_start_y = y
            # append image row
            current_row_img.append(in_img[y, :])

    return char_regions_list


def highlight_letters(orig_img, sub_image_regions):
    color_img = cv2.cvtColor(orig_img.copy(), cv2.COLOR_GRAY2BGR)
    overlay = color_img.copy()
    color = (140, 240, 140)
    alpha = 0.5

    for line in sub_image_regions:
        for region in line:
            top_left = (region.startX, region.startY)
            bottom_right = (region.startX + region.width - 1, region.startY + region.height - 1)
            cv2.rectangle(
                overlay,
                top_left,
                bottom_right,
                color,
                thickness=cv2.FILLED
            )

    overlay = cv2.addWeighted(overlay, alpha, color_img, 1 - alpha, 0, color_img)
    return overlay


def calculate_norm_arr(input_regions, FG_val, features_to_use):
    # calculate the average per feature to allow for normalization
    return_arr = [0] * len(features_to_use)
    num_of_regions = 0

    for i in range(len(features_to_use)):
        curr_row = input_regions[i]
        for j in range(len(curr_row)):
            curr_feature_vals = calc_feature_arr(curr_row[j], FG_val, features_to_use)
            for k in range(len(return_arr)):
                return_arr[k] += curr_feature_vals[k]

            num_of_regions += 1

    for k in range(len(return_arr)):
        return_arr[k] /= num_of_regions

    return return_arr

def calc_feature_arr(region, FG_val, features_to_use):
    feature_res_arr = [0] * len(features_to_use)
    for i in range(len(features_to_use)):
        curr_feature_val = features_to_use[i].CalcFeatureVal(region, FG_val)
        feature_res_arr[i] = curr_feature_val

    return feature_res_arr

# def is_matching_char(curr_feature_arr, ref_feature_arr, norm_feature_arr):
#     CORR_COEFFICIENT_LIMIT = 0.999
#
#     # first normalize the arrays
#
#     #then calulate correlation_coefficient
#     correlation_coefficient = 1.0 #TODO change
#
#     if correlation_coefficient > CORR_COEFFICIENT_LIMIT:
#         return True
#
#     return False

def is_matching_char(curr_feature_arr, ref_feature_arr, norm_feature_arr):
    """
    Returns True if the normalized feature vector of curr_feature_arr
    correlates sufficiently strongly with ref_feature_arr.

    :param curr_feature_arr: List of feature values of the current region
    :param ref_feature_arr: List of the feature values of the reference character
    :param norm_feature_arr: List of mean values (or scaling values) per feature
    """
    CORR_COEFFICIENT_LIMIT = 0.99999

    curr = np.array(curr_feature_arr, dtype=float)
    ref  = np.array(ref_feature_arr,  dtype=float)
    norm = np.array(norm_feature_arr, dtype=float)

    # Scaling: Divide feature by scaling value
    curr_scaled = curr / norm
    ref_scaled  = ref  / norm

    # Calculate correlation
    # np.corrcoef returns the correlation matrix [[1, corr], [corr, 1]]
    corr_matrix = np.corrcoef(curr_scaled, ref_scaled)
    corr_coeff  = corr_matrix[0, 1]

    # Comparison with threshold value
    return corr_coeff > CORR_COEFFICIENT_LIMIT


class ImageFeatureF_FGcount(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "F1: Pixelanzahl"

    @staticmethod
    def CalcFeatureVal(imgRegion, FG_val):
        count = 0
        for x in range(imgRegion.width):
            for y in range(imgRegion.height):
                if imgRegion.subImgArr[y][x] == FG_val:
                    count += 1
        return count

class ImageFeatureF_MaxDistX(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "maximale Ausdehnung in X-Richtung"

    @staticmethod
    def CalcFeatureVal(imgRegion, FG_val):
        return imgRegion.width

class ImageFeatureF_MaxDistY(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "maximale Ausdehnung in Y-Richtung"

    @staticmethod
    def CalcFeatureVal(imgRegion, FG_val):
        return imgRegion.height

class ImageFeatureF_AspectRatio(ImageFeatureBase.ImageFeatureBase):

    def __init__(self):
        super().__init__()
        self.description = "AspectRatio: relative ration between width and height"

    @staticmethod
    def CalcFeatureVal(imgRegion, FG_val):
        return imgRegion.width * imgRegion.height

class ImageFeatureF_FgBgRatio(ImageFeatureBase.ImageFeatureBase):

    def __init__(self):
        super().__init__()
        self.description = "FgBgRatio: relative ratio between foreground and background"

    @staticmethod
    def CalcFeatureVal(imgRegion, FG_val):
        fg = 0
        bg = 0
        for y in range(imgRegion.height):
            for x in range(imgRegion.width):
                if imgRegion.subImgArr[y][x] == FG_val:
                    fg += 1
                else:
                    bg += 1
        return fg / max(bg, 1)

class ImageFeatureF_VerticalAsym(ImageFeatureBase.ImageFeatureBase):

    def __init__(self):
        super.__init__()
        self.description = "VerticalAsym: Vertical asymmetry"

    @staticmethod
    def CalcFeatureVal(imgRegion, FG_val):
        r = 0.0
        for y in range(imgRegion.height):
            for x in range(imgRegion.width):
                if imgRegion.subImgArr[y][x] == FG_val:
                    r += imgRegion.width / imgRegion.height * x        # f(x) = kx + d
        return r

class ImageFeatureF_HorizontalAsym(ImageFeatureBase.ImageFeatureBase):

    def __init__(self):
        super.__init__()
        self.description = "HorizontalAsymRatio: Horizontal asymmetry"

    @staticmethod
    def CalcFeatureVal(imgRegion, FG_val):
        r = 0.0
        for y in range(imgRegion.height):
            for x in range(imgRegion.width):
                if imgRegion.subImgArr[y][x] == FG_val:
                    r += imgRegion.height / imgRegion.width * y  # f(x) = kx + d
        return r

def main():
    print("OCR")
    inImgPath = os.path.join(os.getcwd(), "altesTestament_ArialBlack.png")
    myAnalysis = OCRanalysis()

    testCases = [
        ('e', 5, 0, 169),
        ('n', 0, 3, 115),
        ('s', 1, 3, 102),
        ('a', 0, 5, 92),
        ('t', 1, 4, 82),
        ('r', 1, 6, 81),
        ('d', 4, 2, 69),
        ('i', 1, 1, 57),
        ('h', 0, 10, 45),
        ('u', 0, 11, 39),
        ('o', 2, 1, 38),
        ('m', 0, 1, 37),
        ('l', 1, 10, 35),
        ('c', 0, 9, 33),
        ('g', 0, 7, 27),
        ('w', 2, 13, 25),
        ('G', 2, 0, 23),
        ('b', 1, 14, 22),
        ('.', 3, 3, 20),
        (',', 2, 47, 16),
        ('L', 2, 18, 10),
        ('v', 3, 22, 10),
        ('E', 2, 11, 8),
        ('W', 6, 7, 8),
        (':', 2, 10, 8),
        ('T', 5, 6, 8),
        ('D', 5, 10, 8),
        ('S', 7, 55, 8),
        ('A', 13, 22, 7),
        ('ü', 13, 8, 7),
        ('ö', 5, 31, 7),
        ('f', 0, 4, 6),
        ('F', 1, 0, 6),
        ('z', 19, 0, 6),
        ('H', 18, 30, 5),
        ('p', 18, 5, 5),
        ('M', 17, 43, 4),
        ('B', 14, 0, 3),
        (';', 0, 30, 2),
        ('U', 1, 20, 2),
        ('N', 18, 57, 2),
        ('k', 10, 39, 2),
        ('j', 15, 17, 2),
        ('P', 13, 29, 2),
        ('ä', 14, 1, 2),
        ('I', 0, 0, 1),
        ('O', 10, 24, 1),
        ('Z', 19, 20, 1),
        ('J', 20, 8, 1)
    ]

    successful_tests = 0
    failed_tests = 0
    for testCase in testCases:
        hits = myAnalysis.run(inImgPath, testCase[1],testCase[2])
        if hits == testCase[3]:
            print(f'SUCCESS character "{testCase[0]}" - expected: {testCase[3]} found: {hits}')
            successful_tests += 1
        else:
            print(f'FAILED character "{testCase[0]}" - expected: {testCase[3]} found: {hits}')
            failed_tests += 1

    print(f'==============================================================')
    print(f'Tests passsed: SUCCESSFULLY {successful_tests} / FAILED {failed_tests}')
    print(f'==============================================================')

if __name__ == "__main__":
    main()