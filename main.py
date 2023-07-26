import mediapipe as mp
import cv2
import pickle
import numpy as np
from ar_corrector.corrector import Corrector


corr = Corrector()

model_dict = pickle.load(open('./model.p', 'rb'))

model = model_dict['model']
labels_dict = {0: 'Alef', 1: 'Baa', 3: 'Taa', 4: 'Tha', 5: 'Jiem', 6: 'Haa', 7: 'Khaa', 8: 'Dal', 9: 'Thal', 10: 'Ra', 11: 'Zay', 12: 'Sien', 13: 'Shien', 14: 'Sad', 15: 'Dhad', 16: 'Tah', 17: 'Thah', 18: 'Ain', 19: 'Ghain', 20: 'Faa', 21: 'Qaf', 22: 'Kaf', 23: 'Lam', 24: 'Miem', 25: 'Noon', 26: 'He', 27: 'Waw', 28: 'Ya'}
arabic_alphabet = {
    'Alef': 'ا',
    'Baa': 'ب',
    'Taa': 'ت',
    'Tha': 'ث',
    'Jiem': 'ج',
    'Haa': 'ح',
    'Khaa': 'خ',
    'Dal': 'د',
    'Thal': 'ذ',
    'Ra': 'ر',
    'Zay': 'ز',
    'Sien': 'س',
    'Shien': 'ش',
    'Sad': 'ص',
    'Dhad': 'ض',
    'Tah': 'ط',
    'Thah': 'ظ',
    'Ain': 'ع',
    'Ghain': 'غ',
    'Faa': 'ف',
    'Qaf': 'ق',
    'Kaf': 'ك',
    'Lam': 'ل',
    'Miem': 'م',
    'Noon': 'ن',
    'He': 'ه',
    'Waw': 'و',
    'Ya': 'ي'
}

word = ''
currentLetter = ''
def add_letter_to_string(letter, my_word):
    if len(my_word) == 0 or my_word[-1] != letter:
        my_word += letter
    return my_word

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cam = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:

    while cam.isOpened():
        ret, frame = cam.read()
        H, W, _ = frame.shape

        data_aux = []
        x_ = []
        y_ = []


        imagergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagergb = cv2.flip(imagergb, 1)


        results = hands.process(imagergb)



        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(imagergb,hand, mp_hands.HAND_CONNECTIONS)

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(np.asarray(data_aux)) == 42:
                prediction = model.predict([np.asarray(data_aux)])
            else:
                prediction = "None"
            # predicted_character = labels_dict[int(prediction[0])]

            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            # cv2.putText(frame, prediction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,cv2.LINE_AA)
            if prediction != 'None':

                if (prediction[0] == 'Stop'):
                    if len(word) > 1:
                        print("Word: ",word)
                        if corr.spell_correct(word) == True:
                            print(word)
                        else:
                            print(corr.spell_correct(word)[0][0])
                        word = ''
                else:
                    letter = arabic_alphabet[prediction[0]]
                    print(letter)
                    if(len(currentLetter) == 0):
                        currentLetter = letter
                    else:
                        if (letter == currentLetter[-1]):
                            currentLetter += letter
                        else:
                            if (len(currentLetter) > 5):
                                word = add_letter_to_string(currentLetter[0], word)

                            currentLetter = ''


        cv2.imshow('Webcam', imagergb)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cam.release()
cv2.destroyAllWindows()