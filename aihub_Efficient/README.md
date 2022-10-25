# EfficientNet
- 데이터 전처리</br>
AI Hub의 [유형별 두피 이미지](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=216)에서 데이터셋 다운받음
탈모(alopecia)의 경우 train, validation만 있으므로, test를 임의로 train에서 나눠서 가져옴.

- 폴더 구조
```
aihub_Efficient
|- EfficientNet.py
|- scalp_weights
|- train_data
   |- model6
      |- test
      |- train
         |- alopecia_0
            |- 0013_A2LEBJJDE00060O_1606550825417_3_TH.jpg
            |- ...
         |- alopecia_1
         |- ...
      |- validation
         |- alopecia_0
         |- ...
```

- 실행
```
python EfficientNet.py
```
