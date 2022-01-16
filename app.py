from flask import Flask
from flask import Blueprint,request,jsonify,make_response,redirect
import os,uuid
import cv2
from main import get_face,get_mask_FCN,get_face_white_bg_photo2cartoon,get_cartoon_face_photo2cartoon,put_text,blend_images,merge_process
import numpy as np
from PIL import Image
from main import InstanceFaceDetect,InstanceFCN
app = Flask(__name__, static_folder='./result')

@app.route('/',methods=['POST','GET'])
def ZYY():
    cdir = os.getcwd()+"\\Image\\dream.jpg"
    image_data = open(cdir, 'rb').read()
    res = make_response(image_data)
    res.headers['Content-Type'] = 'image/png'
    return res

def rundim_file(filename='123456'):
    """文件命名，重名概率接近0"""
    ext = os.path.split(filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename

@app.route('/GANForCartoon',methods=['POST','GET'])
def GANForCartoon():
    # ADDR_PATH = "/result/"
    # flask 无法执行parse_opt()

    shear_rate = 0.8
    text_scale = 90
    SAVE_IMG_PATH = os.path.abspath('.') + '\\result\image' # 人脸图像
    SAVE_IMG_BG_PATH = os.path.abspath('.') + '\\result\image_bg' # 背景图像
    SAVE_IMG_CARTOON_PATH = os.path.abspath('.') + '\\result\cartoon_img' # 卡通图像
    SAVE_IMG_FOREGROUND_CARTOON_PATH = os.path.abspath('.') + '\\result\\foreground_cartoon_img' # 前景融合图像
    SAVE_IMG_BACKGROUND_CARTOON_PATH = os.path.abspath('.') + '\\result\\background_cartoon_img'
    # android传来的数据
    text_content  = request.form.get("text_content") # 签名
    img = request.files.get('img') # 人脸图像'
    method = int(request.form.get("method"))  # 摄像头 -1为激活程序 0为后置 1为前置 2为相册
    img_bg_select =request.form.get('img_bg_select') # 背景图像的选择
    img_bg = request.files.get('img_bg')  # 背景图像
    fusion_method = request.form.get("fusion_method") # 融合方式



    if text_content is None:
        text_content = ""
    if method == -1:
        img = Image.open('Image/nini.png')
    else:
        img_name = rundim_file(img.filename)

    text_content_len = len(text_content)
    text_location = (512-text_content_len*32, 400)


    img_path = os.path.join(os.path.join(SAVE_IMG_PATH, img_name))
    print('img_bg_select==',img_bg_select)
    img_bg_path = os.path.join(os.path.join(SAVE_IMG_BG_PATH, img_name[:-4]+"_bg.png"))

    img_cartoon_path = os.path.join(os.path.join(SAVE_IMG_CARTOON_PATH,img_name[:-4]+ "_cartoon_face.png"))
    img_fore_cartoon_path = os.path.join(os.path.join(SAVE_IMG_FOREGROUND_CARTOON_PATH,img_name[:-4]+ "_fore_cartoon_face.png"))
    img_back_cartoon_path = os.path.join(os.path.join(SAVE_IMG_BACKGROUND_CARTOON_PATH,img_name[:-4]+ "_back_cartoon_face.png"))
    if img_bg_select == '1':
        img_bg.save(img_bg_path)
    if img_bg_select == '2':
        img_bg_path = os.path.abspath('.') + '\\Image\\yourname.jpeg'
    if img_bg_select == '3':
        img_bg_path = os.path.abspath('.') + '\\Image\\weatherson.jpg'
    print('img_bg_path==',img_bg_path)
    img.save(img_path)
    print('method==',method)


    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    if method == 1:  # 前置
        # rows, cols, channels = img.shape
        # rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), -90, 1)
        # img = cv2.warpAffine(img, rotate, (cols, rows))
        img = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    elif method == 0 :#后置
        # rows, cols, channels = img.shape
        # rotate = cv2.getRotationMatrix2D((rows * 0.5, cols * 0.5), 90, 1)
        # img = cv2.warpAffine(img, rotate, (cols, rows))
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imwrite("222.jpg", img)
    height, width = img.shape[:2]
    if height>1024:
        h_scale = 1024/height
        new_height = int(h_scale*height)
        new_width = int(h_scale*width)
        img = cv2.resize(img,(new_width,new_height))
    #cv2.imwrite('test.jpg',img)
    face = get_face(img, shear_rate, img_name)

    mask = get_mask_FCN(face)
    face_white_bg = get_face_white_bg_photo2cartoon(np.dstack((face, mask)))  # 分割后的人脸

    cartoon_face = get_cartoon_face_photo2cartoon(face_white_bg, mask)  # 卡通图
    cartoon_face = cartoon_face[:,:,::-1]
    cv2.imwrite(img_cartoon_path,cartoon_face)

    if fusion_method is None or img_bg_select=='0':
        return "/result/cartoon_img/"+img_name[:-4]+ "_cartoon_face.png"
    if fusion_method == 'pre_fusion':# 前景融合
        background_img = cv2.imread(img_bg_path,cv2.COLOR_BGR2RGB)
        background_img = cv2.resize(background_img,(512,512))

        img_fore_cartoon = blend_images(cartoon_face[:,:,::-1], background_img[:,:,::-1])  # 漫画与背景融合
        img_fore_cartoon.save(img_fore_cartoon_path)

        background_text = put_text(img_fore_cartoon,text_content,text_scale,text_location)  # 加上文字
        #
        # background_text = np.array(background_text)[:, :, ::-1]
        #
        # background_text = Image.fromarray(np.uint8(background_text))
        background_text.save(img_fore_cartoon_path)
        return "/result/foreground_cartoon_img/"+img_name[:-4]+ "_fore_cartoon_face.png"
    if fusion_method == 'back_fusion':# 背景融合
        background_img = cv2.imread(img_bg_path, cv2.COLOR_BGR2RGB)
        background_img = cv2.resize(background_img, (512, 512))

        cartoon_face = cv2.resize(cartoon_face, (384, 384))
        mask = cv2.resize(mask, (384, 384))

        # background_text = put_text(background_img,text_content,text_scale,text_location)  # 加上文字

        merge_cartoon =merge_process(cartoon_face[:,:,::-1],background_img,mask)
        merge_cartoon = Image.fromarray(cv2.cvtColor(merge_cartoon,cv2.COLOR_BGR2RGB))
        merge_cartoon = put_text(merge_cartoon, text_content, text_scale, text_location)  # 加上文字
        merge_cartoon.save(img_back_cartoon_path)
        # cv2.imwrite(img_back_cartoon_path, merge_cartoon) # 荣和
        return "/result/background_cartoon_img/"+img_name[:-4]+ "_back_cartoon_face.png"



if __name__ == '__main__':
    app.run()