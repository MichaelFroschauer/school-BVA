import math
import cv2
import numpy as np

import ImageFeatureBase
import SubImageRegion


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

    def run(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        FG_VAL = 0
        BG_VAL = 255
        MARKER_VAL = 127
        thresholdVal = 127

        _, binaryImgArr = cv2.threshold(img, thresholdVal, BG_VAL, cv2.THRESH_BINARY)
        cv2.imwrite("/home/michael/school/gitclones/2_BVA/2_3_OCR/binaryOut.png", binaryImgArr)

        #define the features to evaluate
        features_to_use = []
        features_to_use.append(ImageFeatureF_FGcount())
        features_to_use.append(ImageFeatureF_MaxDistX)
        features_to_use.append(ImageFeatureF_MaxDistY)
        
        #linked_regions, lines = split_characters(binaryImgArr, width, height, BG_VAL, FG_VAL)
        linked_regions = split_characters(binaryImgArr, width, height, BG_VAL, FG_VAL)
        
        #define the reference character
        tgtCharRow = 2
        tgtCharCol = 3
        charROI = linked_regions[tgtCharRow][tgtCharCol]

        # test calculate features
        print('features of reference character is: ')
        feature_res_arr = calc_feature_arr(charROI, BG_VAL, features_to_use)
        self.printout_feature_res(feature_res_arr, features_to_use)

        #then normalize
        feature_norm_arr = calculate_norm_arr(linked_regions, BG_VAL, features_to_use)
        print('NORMALIZED features: ')
        self.printout_feature_res(feature_norm_arr, features_to_use)

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
        cv2.imwrite("/home/michael/school/gitclones/2_BVA/2_3_OCR/markedChars.png", binary_img_arr)
        print('num of found characters is = ' + str(hitCount))

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
        print('adjusted colors is ' + str(adjustedColors))
        return in_img_arr

    def printout_feature_res(self, feature_res_arr, features_to_use):
        print("========== features =========")
        for i in range(len(features_to_use)):
            print("res of F", i, ",", features_to_use[i], "is", feature_res_arr[i])

def is_empty_column(in_img, height, col_idx, BG_val):
    #TODO implementation required
    for y in range(height):
        if in_img[y][col_idx] != BG_val:
            return False
    return True

def is_empty_row(in_img, width, row_idx, BG_val):
    #TODO implementation required
    for x in range(width):
        if in_img[row_idx][x] != BG_val:
            return False
    return True

def split_characters_vertically(row_image, BG_val, FG_val, orig_img):
    # TODO implementation required
    return_char_arr = []
    current_col_img = []
    height, width = row_image.shape

    # iterate over all columns
    for x in range(width):
        if is_empty_column(row_image, height, x, BG_val):
            if current_col_img:
                col_img = np.array(current_col_img).T
                return_char_arr.append(col_img.copy())
                current_col_img = []
        else:
            current_col_img.append(row_image[:, x])

    return return_char_arr

def split_characters(in_img, width, height, BG_val, FG_val):
    # TODO implementation required
    return_char_matrix = []
    current_row_img = []

    # iterate over all lines
    for y, img_row in enumerate(in_img):
        if is_empty_row(in_img, width, y, BG_val):
            # found a full row -> split characters vertically and add it to the result
            if current_row_img:
                row_img = np.array(current_row_img)
                single_character_list = split_characters_vertically(row_img, BG_val, FG_val, in_img)
                return_char_matrix.append(single_character_list)
                current_row_img = []
        else:
            current_row_img.append(img_row)

    # for debugging, show single character
    for i, img_row in enumerate(return_char_matrix):
        for j, img_char in enumerate(img_row):
            cv2.imshow(f"split_characters_{j}", img_char)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    return return_char_matrix

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

def is_matching_char(curr_feature_arr, ref_feature_arr, norm_feature_arr):
    CORR_COEFFICIENT_LIMIT = 0.999

    # first normalize the arrays
   
    #then calulate correlation_coefficient
    correlation_coefficient = 1.0 #TODO change

    if correlation_coefficient > CORR_COEFFICIENT_LIMIT:
        return True

    return False


class ImageFeatureF_FGcount(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "F1: Pixelanzahl"

    def CalcFeatureVal(self, imgRegion, FG_val):
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

    def CalcFeatureVal(imgRegion, FG_val):
        return imgRegion.width

class ImageFeatureF_MaxDistY(ImageFeatureBase.ImageFeatureBase):
    def __init__(self):
        super().__init__()
        self.description = "maximale Ausdehnung in Y-Richtung"

    def CalcFeatureVal(imgRegion, FG_val):
        return imgRegion.height


def main():
    print("OCR")
    inImgPath = "/home/michael/school/gitclones/2_BVA/2_3_OCR/altesTestament_ArialBlack.png"
    myAnalysis = OCRanalysis()
    myAnalysis.run(inImgPath)

if __name__ == "__main__":
    main()