"""
第2章SSDで予測結果を画像として描画するクラス

"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2  # OpenCVライブラリ
import torch
import time
from utils.dataset import DatasetTransform as DataTransform
import torch.nn as nn

class SSDPredictShow(nn.Module):
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, eval_categories, net, device, TTA=True, image_size=300):
        super(SSDPredictShow, self).__init__()  # 親クラスのコンストラクタ実行
        print(device)
        self.eval_categories = eval_categories  # クラス名
        self.net = net.to(device).eval()  # SSDネットワーク
        self.device = device
        self.TTA=TTA

        color_mean = (104, 117, 123)  # (BGR)の色の平均値
        input_size = image_size  # 画像のinputサイズを300×300にする
        self.transform = DataTransform(input_size, color_mean)  # 前処理クラス

    def show(self, image_file_path, data_confidence_level):
        """
        物体検出の予測結果を表示をする関数。

        Parameters
        ----------
        image_file_path:  str
            画像のファイルパス
        data_confidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        SSDで予測させる関数。

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # rgbの画像データを取得
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = img.shape  # 画像のサイズを取得
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像の前処理
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            img, phase, "", "")  # アノテーションが存在しないので""にする。
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1).to(self.device)
        print(img.size())
        
        # SSDで予測
        #self.net.eval()  # ネットワークを推論モードへ
        x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])

        detections = self.net(x)
        # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

        # confidence_levelが基準以上を取り出す
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        # 条件以上の値を抽出
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
            if (find_index[1][i]) > 0:  # 背景クラスでないもの
                sc = detections[i][0]  # 確信度
                bbox = detections[i][1:] * [width, height, width, height]
                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i]-1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores
    def ssd_predict2(self, image_file_path, data_confidence_level=0.5):
        """
        SSDで予測させる関数。

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # rgbの画像データを取得
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = img.shape  # 画像のサイズを取得
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像の前処理
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            img, phase, "", "")  # アノテーションが存在しないので""にする。
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1).to(self.device)

        # SSDで予測
        #self.net.eval()  # ネットワークを推論モードへ
        x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])
        
        with torch.no_grad():
            detections = self.net(x)
        # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

        # confidence_levelが基準以上を取り出す
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        try:
            detections = detections.cpu().detach().numpy()
        except:
            detections = detections.detach().numpy()

        # 条件以上の値を抽出
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
            if (find_index[1][i]) > 0:  # 背景クラスでないもの
                sc = detections[i][0]  # 確信度
                detections[i][1:] *= [width, height, width, height]
                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i]-1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                #predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return detections, pre_dict_label_index
    def ssd_inference(self, dataloader, all_boxes, data_confidence_level=0.05):
        """
        SSDで予測させる関数。

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """
        
        empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
        iii=0 # image number
        width = 300
        height = 300
        for img, _ in dataloader:
            num_batch = len(img)
            # SSDで予測
            self.net.eval().to(self.device)  # ネットワークを推論モードへ
            tick = time.time()
            with torch.no_grad():
                x = img.to(self.device)  # ミニバッチ化：torch.Size([1, 3, 300, 300])               
                detections = self.net(x)
                
            tock = time.time()
            # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

            # confidence_levelが基準以上を取り出す
            predict_bbox = []
            pre_dict_label_index = []
            scores = []
            detections = detections.cpu().detach().numpy()
            print(detections.shape)

            # 条件以上の値を抽出
            took = time.time()
            for batch, detection in enumerate(detections):
                for cls in range(21):
                    box = []
                    for j,pred in enumerate(detection[cls]):
                        if pred[0] > data_confidence_level:
                            pred[1:] *= width
                            box.append([pred[0],pred[1],pred[2],pred[3],pred[4]])
                    if not box == []:
                        all_boxes[cls][iii*num_batch + batch] = box
                    else:
                        all_boxes[cls][iii*num_batch + batch] = empty_array
                    
            teek = time.time()
            #if i%100==0:
            print("iter:", iii)            
            iii += 1
            
            print("sort boxes. detection was {} and post took {} and allboxappend took {}".format(tock-tick, took-tock, teek-took))
            
        return all_boxes
    
    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        物体検出の予測結果を画像で表示させる関数。

        Parameters
        ----------
        rgb_img:rgbの画像
            対象の画像データ
        bbox: list
            物体のBBoxのリスト
        label_index: list
            物体のラベルへのインデックス
        scores: list
            物体の確信度。
        label_names: list
            ラベル名の配列

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """

        # 枠の色の設定
        num_classes = len(label_names)  # クラス数（背景のぞく）
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 画像の表示
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        # BBox分のループ
        for i, bb in enumerate(bbox):

            # ラベル名
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

            # 枠につけるラベル　例：person;0.72　
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # 枠の座標
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # 長方形を描画する
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))

            # 長方形の枠の左上にラベルを描画する
            currentAxis.text(xy[0], xy[1], display_txt, bbox={
                             'facecolor': color, 'alpha': 0.5})

def vis_bbox(rgb_img, bbox, label_index, scores, label_names):
        """
        物体検出の予測結果を画像で表示させる関数。

        Parameters
        ----------
        rgb_img:rgbの画像
            対象の画像データ
        bbox: list
            物体のBBoxのリスト
        label_index: list
            物体のラベルへのインデックス
        scores: list
            物体の確信度。
        label_names: list
            ラベル名の配列

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """

        # 枠の色の設定
        num_classes = len(label_names)  # クラス数（背景のぞく）
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 画像の表示
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        # BBox分のループ
        for i, bb in enumerate(bbox):

            # ラベル名
            label_name = label_names[int(label_index[i])]
            color = colors[int(label_index[i])]  # クラスごとに別の色の枠を与える

            # 枠につけるラベル　例：person;0.72　
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # 枠の座標
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # 長方形を描画する
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))

            # 長方形の枠の左上にラベルを描画する
            currentAxis.text(xy[0], xy[1], display_txt, bbox={
                             'facecolor': color, 'alpha': 0.5})

# wrap to function..
def show_detection(img, hm, regr, thresh=0.7, input_size=256):
    hm = F.sigmoid(torch.from_numpy(hm)).numpy()
    print(hm.shape)
    where = np.where(hm > thresh)

    # center = [xmin,ymin, y, class, score]
    # box = xmin, ymin, xmax, ymax, class, score
    clss = np.asarray(where[0])
    xs = np.asarray(where[1])
    ys = np.asarray(where[2])

    center = []; box = []
    for y,x,cls in zip(xs, ys, clss):
        score = hm[cls,x,y]
        w, h = regr[:, x, y] * input_size//MODEL_SCALE
        print(w)
        print(h)

        box.append(np.asarray([x-w/2,y-h/2,x+w/2,y+h/2,cls,score]))
        center.append(np.asarray([x,y,cls,score]))

    center = np.asarray(center)
    box = np.asarray(box)

    # scale box scale to original image
    box[:, 0:4] *= MODEL_SCALE
    center[:, 0:2] *= MODEL_SCALE
    
    # Let's see if the center is encoded right
    img2 = cv2.resize(img, (input_size, input_size))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # plot stuff centers
    for c in center:
        img2 = cv2.circle(img2, (int(c[0]), int(c[1])), 20, (255, 0, 0), 5)

    print("plot center points")
    plt.imshow(img2/255)
    plt.show()
    
    # plot boxes
    # Let's see if the center is encoded right
    img2 = cv2.resize(img, (input_size, input_size))
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    vis_bbox(img2, box, label_index=box[:,4], scores=box[:,5], label_names=voc_classes)
            
from utils.ssd import Detect_Flip

class SSDPredictShowFlip(nn.Module):
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, eval_categories, net, device, TTA=True, softnms=False):
        super(SSDPredictShowFlip, self).__init__()  # 親クラスのコンストラクタ実行
        print(device)
        self.eval_categories = eval_categories  # クラス名
        self.net = net.to(device)  # SSDネットワーク
        self.device = device
        self.TTA =TTA

        color_mean = (104, 117, 123)  # (BGR)の色の平均値
        input_size = 300  # 画像のinputサイズを300×300にする
        self.transform = DataTransform(input_size, color_mean)  # 前処理クラス
        
        self.Det = Detect_Flip(TTA=TTA, softnms=softnms).to(self.device).eval()

    def show(self, image_file_path, data_confidence_level):
        """
        物体検出の予測結果を表示をする関数。

        Parameters
        ----------
        image_file_path:  str
            画像のファイルパス
        data_confidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        SSDで予測させる関数。

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # rgbの画像データを取得
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = img.shape  # 画像のサイズを取得
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像の前処理
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            img, phase, "", "")  # アノテーションが存在しないので""にする。
        #print("img shape:", img_transformed.shape)        
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1).to(self.device)

        # SSDで予測
        #self.net.eval()  # ネットワークを推論モードへ
        x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])
        with torch.no_grad():
            detections = self.net(x)
        # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値
        
        ## Flip inference
        x_flip = torch.flip(img, [2])
        x_flip = x_flip.unsqueeze(0)
        with torch.no_grad():
            detections_flip = self.net(x_flip)
        
        #print("check box: ", (detections[2]==detections_flip[2]).sum().numpy())
        
        ## Gather detections.
        detections_box = self.Det(detections[0], detections[1], detections_flip[0], detections_flip[1], detections[2].to(self.device))
        
        # confidence_levelが基準以上を取り出す
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections_box.cpu().detach().numpy()
        

        # 条件以上の値を抽出
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
            if (find_index[1][i]) > 0:  # 背景クラスでないもの
                sc = detections[i][0]  # 確信度
                bbox = detections[i][1:] * [width, height, width, height]
                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i]-1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        物体検出の予測結果を画像で表示させる関数。

        Parameters
        ----------
        rgb_img:rgbの画像
            対象の画像データ
        bbox: list
            物体のBBoxのリスト
        label_index: list
            物体のラベルへのインデックス
        scores: list
            物体の確信度。
        label_names: list
            ラベル名の配列

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """

        # 枠の色の設定
        num_classes = len(label_names)  # クラス数（背景のぞく）
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 画像の表示
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        # BBox分のループ
        for i, bb in enumerate(bbox):

            # ラベル名
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

            # 枠につけるラベル　例：person;0.72　
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # 枠の座標
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # 長方形を描画する
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))

            # 長方形の枠の左上にラベルを描画する
            currentAxis.text(xy[0], xy[1], display_txt, bbox={
                             'facecolor': color, 'alpha': 0.5})

            
class SSDPredictShowTest(nn.Module):
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, eval_categories, net, device):
        super(SSDPredictShowTest, self).__init__()  # 親クラスのコンストラクタ実行
        print(device)
        self.eval_categories = eval_categories  # クラス名
        self.net = net.to(device)  # SSDネットワーク
        self.device = device

        color_mean = (104, 117, 123)  # (BGR)の色の平均値
        input_size = 300  # 画像のinputサイズを300×300にする
        self.transform = DataTransform(input_size, color_mean)  # 前処理クラス

    def show(self, image_file_path, data_confidence_level):
        """
        物体検出の予測結果を表示をする関数。

        Parameters
        ----------
        image_file_path:  str
            画像のファイルパス
        data_confidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        SSDで予測させる関数。

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # rgbの画像データを取得
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = img.shape  # 画像のサイズを取得
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 画像の前処理
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            img, phase, "", "")  # アノテーションが存在しないので""にする。
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # SSDで予測
        #self.net.eval()  # ネットワークを推論モードへ
        x = img.unsqueeze(0).to(self.device)  # ミニバッチ化：torch.Size([1, 3, 300, 300])

        detections = self.net(x)
        # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

        # confidence_levelが基準以上を取り出す
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        # 条件以上の値を抽出
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
            if (find_index[1][i]) > 0:  # 背景クラスでないもの
                sc = detections[i][0]  # 確信度
                bbox = detections[i][1:] * [width, height, width, height]
                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i]-1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        物体検出の予測結果を画像で表示させる関数。

        Parameters
        ----------
        rgb_img:rgbの画像
            対象の画像データ
        bbox: list
            物体のBBoxのリスト
        label_index: list
            物体のラベルへのインデックス
        scores: list
            物体の確信度。
        label_names: list
            ラベル名の配列

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """

        # 枠の色の設定
        num_classes = len(label_names)  # クラス数（背景のぞく）
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 画像の表示
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        # BBox分のループ
        for i, bb in enumerate(bbox):

            # ラベル名
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

            # 枠につけるラベル　例：person;0.72　
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # 枠の座標
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # 長方形を描画する
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))

            # 長方形の枠の左上にラベルを描画する
            currentAxis.text(xy[0], xy[1], display_txt, bbox={
                             'facecolor': color, 'alpha': 0.5})