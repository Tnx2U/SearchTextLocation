import cv2
import numpy as np
from PIL import Image
import pytesseract


def searchBoxSize(min, max, target):
    if target > min and target < max:
        return 1
    return 0

def erodeDilate(img):
    kernel = np.ones((3, 3), np.uint8)

    img = cv2.dilate(img, kernel,iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)


    return img

def body():
    tesseract_path = 'C:/Program Files (x86)/Tesseract-OCR'
    pytesseract.pytesseract.tesseract_cmd = tesseract_path + '/tesseract.exe'

    image = cv2.imread('image/ex_eng.jpg')
    userInputText = input("검색할 단어를 입력하세요. : ")
    textArray = []

    #resizeing

    # r = 800.0 / userInputImage.shape[0]
    # dim = (int(userInputImage.shape[1] * r), 800)
    # image = cv2.resize(userInputImage, dim, interpolation=cv2.INTER_AREA)
    withContour = image.copy()
    withBox = image.copy()

    #resizing 끝, 이진화 시작

    print("STEP 1: 이진화, contour추출")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    #이진화 끝, 침식팽창 시작

    edged = cv2.Canny(binary, 75, 200)
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(withContour, cnts, -1, (0, 255, 0), 1)


    cv2.imshow("original", image)
    cv2.imshow("binary", binary)
    cv2.imshow("Edged", withContour)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print("STEP 2: 침식 팽창")
    erodeDilated = erodeDilate(binary)

    cv2.imshow("erodeDilate", erodeDilated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    #침식팽창 끝, 예상지역 박싱

    print("STEP 3: 후보 지역 탐색")

    edEdged = cv2.Canny(erodeDilated, 75, 200)
    (_, contours, _) = cv2.findContours(edEdged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i]) #사각형 꼭지점 좌표

        if searchBoxSize(40, 100, h) and searchBoxSize(60, 300, w):
            cv2.rectangle(withBox, (x, y), (x+w, y+h), (0,255,0), 2)
            candidate = gray[y:y + h, x:x + w]
            cv2.imwrite("image/candidate"+str(i)+".jpg", candidate)
            tesseract_image = Image.open('image/candidate'+str(i)+'.jpg')
            text = pytesseract.image_to_string(tesseract_image)
            textArray.append([text, (x,y), (x+w, y+h)])


    cv2.imshow("withBox", withBox)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


    index = 0
    failed = True


    #예상지역 박싱
    for i in range(len(textArray)):
        if userInputText == textArray[i][0]:
            print("일치하는 텍스트가 존재합니다")
            print(textArray[i][1], textArray[i][2])
            cv2.rectangle(image, textArray[i][1], textArray[i][2], (255, 0, 0), 2)
            cv2.imshow("complete",image)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            failed = False

            break

    if failed:
        print("일치하는 텍스트가 없습니다")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == '__main__' :
    body()