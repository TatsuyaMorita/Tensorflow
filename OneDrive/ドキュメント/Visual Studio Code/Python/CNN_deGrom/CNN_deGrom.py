import numpy as np
import cv2
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import statistics

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = "keras/04_img/ex3_data/face_detector/deploy.prototxt"
weightsPath = "keras/04_img/ex3_Data/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("keras/04_img/ex3_Data/mask_detector.model")

  
def main():
    # データの保存先(自分の環境に応じて適宜変更)
    SAVE_DATA_DIR_PATH1 = "keras/04_img/ex1_data/"
    SAVE_DATA_DIR_PATH2 = "keras/04_img/ex2_data/"


    subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    cap = cv2.VideoCapture(0)#内部カメラ使用時、cap = cv2.VideoCapture(0)
                              #外部カメラ使用時、cap = cv2.VideoCapture(1)
    window_name = "face"
   
    
    print("顔が画面の真ん中に来るようにして、sを押してください。")
    person_name =''
    
    frame_counter = 0
    while True:
       
    
       ret, frame = cap.read()

       (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

       for (box, pred) in zip(locs, preds):
           (startX, startY, endX, endY) = box
           (mask, withoutMask) = pred

           color = (0, 255, 0)

           cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
           
           fontpath = 'C:/Windows/Fonts/meiryo.ttc'
           font = ImageFont.truetype(fontpath, 32)
            
           img_pil = Image.fromarray(frame)
           draw = ImageDraw.Draw(img_pil)
           position = (startX + 6, endY + 10)
                # drawにテキストを記載
           draw.text(position, person_name, font=font, fill=(255, 255, 255, 0))
           frame = np.array(img_pil)

           
          
           # 画像のリセット
           if person_name != '':
            frame_counter = frame_counter + 1
            if frame_counter > 30:
               person_name = ''
               frame_counter = 0

       cv2.imshow(window_name,frame)
       
       presskey = cv2.waitKey(1)
       if presskey == ord('s'):
            target_nums = []
          # 入力画像の保存
            if mask > withoutMask:
             for i in range(1,11):
                ret, frame = cap.read()
                cv2.imwrite(os.path.join(SAVE_DATA_DIR_PATH2,"face.jpg".format(i)), frame[startY:startY+100, startX:endX])

                # 入力画像のパラメータ
                img_width = 32 # 入力画像の幅
                img_height = 32 # 入力画像の高さ
                date_ch = 3 # 3ch画像（RGB）

                # 入力データ数
                num_data = 1
                labels = ['森田達也マスクあり', '森田満マスクあり']
                # 保存したモデル構造の読み込み
                model = model_from_json(open(SAVE_DATA_DIR_PATH2 + "model.json", 'r').read())
                # 保存した学習済みの重みを読み込み
                model.load_weights(SAVE_DATA_DIR_PATH2 + "weight.hdf5")
                img = np.array(load_img(SAVE_DATA_DIR_PATH2 + "face.jpg", target_size=(img_width, img_height)))
                img = img.astype('float32')/255.0
                img = np.array([img])

             # 分類機に入力データを与えて予測（出力：各クラスの予想確率）
                y_pred= model.predict(img)
            
             # 最も確率の高い要素番号

                number_pred = np.argmax(y_pred) 
                target_nums.append(number_pred)
                # 予測結果の表示
            

                 # 日本語表示
                if np.argmax(y_pred)  >0.8: 
                  print("y_pred:", y_pred)  # 出力値
                  print("number_pred:", number_pred)  # 最も確率の高い要素番号
                  print('label_pred：', labels[int(number_pred)]) # 予想ラベル（最も確率の高い要素）
                  print("median_pred", statistics.mode(target_nums))
             
                  person_name = labels[statistics.mode(target_nums)]
         
                  print("最も可能性があるのは...", labels[statistics.mode(target_nums)])
        
            
                else: 
                  print("median_pred", statistics.mode(target_nums))
                  print("Unknown")
            
             
             person_name = "Unknown"
            
            else:
             for i in range(1,11):
                ret, frame = cap.read()
                cv2.imwrite(os.path.join(SAVE_DATA_DIR_PATH1,"face.jpg".format(i)), frame[startY:endY, startX:endX])
                # 入力画像のパラメータ
                img_width = 32 # 入力画像の幅
                img_height = 32 # 入力画像の高さ
                date_ch = 3 # 3ch画像（RGB）

            # 入力データ数
                num_data = 1
                labels = ['森田達也', '森田満','Jacob deGrom','Billie Eilish']
                 # 保存したモデル構造の読み込み
                model = model_from_json(open(SAVE_DATA_DIR_PATH1 + "model.json", 'r').read())
                # 保存した学習済みの重みを読み込み
                model.load_weights(SAVE_DATA_DIR_PATH1 + "weight.hdf5")
                img = np.array(load_img(SAVE_DATA_DIR_PATH1 + "face.jpg", target_size=(img_width, img_height)))

                # 分類機に入力データを与えて予測（出力：各クラスの予想確率）
                y_pred= model.predict(img)
            
             # 最も確率の高い要素番号

                number_pred = np.argmax(y_pred) 
                target_nums.append(number_pred)
                # 予測結果の表示
            

                 # 日本語表示
                if np.argmax(y_pred)  >0.8: 
                  print("y_pred:", y_pred)  # 出力値
                  print("number_pred:", number_pred)  # 最も確率の高い要素番号
                  print('label_pred：', labels[int(number_pred)]) # 予想ラベル（最も確率の高い要素）
                  print("median_pred", statistics.mode(target_nums))
             
                  person_name = labels[statistics.mode(target_nums)]
         
                  print("最も可能性があるのは...", labels[statistics.mode(target_nums)])
        
            
                else: 
                  print("median_pred", statistics.mode(target_nums))
                  print("Unknown")
            
             
             person_name = "Unknown"


         
       elif presskey == ord('q'):
             break          
  
   
    cv2.destroyAllWindows()      
    cap.stop()

  
    
if __name__ == '__main__':
    main()
   

