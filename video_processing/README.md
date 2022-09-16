# SRVC Data Preparation

## Download video files
The authors originally used 28 videos from Vimeo as well as 4 long video sequences from the Xiph video dataset. However, due to restricted storage and time constraints, I only trained and tested my implementation of SRVC on 11 Vimeo videos. Here is the list of videos that I used. You can click on each video and download it individually. To use more videos for training, please refer to the official [**GitHub repo**](https://github.com/AdaptiveVC/SRVC) for instructions to download them.
* [**vimeo_166010169**](https://vimeo.com/166010169)
* [**vimeo_204439769**](https://vimeo.com/204439769)
* [**vimeo_209989434**](https://vimeo.com/209989434)
* [**vimeo_262893710**](https://vimeo.com/262893710)
* [**vimeo_279024382**](https://vimeo.com/279024382)
* [**vimeo_315720775**](https://vimeo.com/315720775)
* [**vimeo_325535466**](https://vimeo.com/325535466)
* [**vimeo_333836715**](https://vimeo.com/333836715)
* [**vimeo_374021613**](https://vimeo.com/374021613)
* [**vimeo_390213492**](https://vimeo.com/390213492)
* [**vimeo_417690570**](https://vimeo.com/417690570)

Once downloaded, put all video files in the ``./data/original_videos`` directory.

## Create YUV raw video files
Change the current directory to the ``./video_processing`` directory, then execute the following command:
```
cd video_processing
bash ./create_yuv.sh ../data/original_videos ../data/yuv_videos
```
Use the following command to create raw YUV files from the original video files that you downloaded earlier. As described in the paper, all videos will be shortened to 10 minutes with a 1080p resolution. 

## Encode videos using H.265 codec
Execute the following command to encode the raw YUV files into H.265-encoded ``.mp4`` files with size (270 X 480). 
```
bash ./content_encoder.sh ../data/yuv_vidoes ../data/content_encoded_videos ../data/original_videos
```
You can adjust the CRF (constant rate factor) values that will be used to encode the videos in line 27 of the ``./content_encoder.sh`` file. Path to original video file is required in order to get the fps(frames per second) information, which is needed to encode the video. 

## Reencode videos using H.264/H.265 codec
Execute the following command to reencode the raw YUV files into H.264/H.265-encoded ``.mp4`` files without downsampling or resizing (size maintained as 1080 X 1920).
```
bash ./content_reencoder.sh ../data/yuv_vidoes ../data/content_reencoded_videos ../data/original_videos
```
The output will be used as the 1080p H.264/H.265 baseline for comparing the performance of the SRVC model to the standard codecs. Note that you may change the CRF values in line 26 of the ``./content_reencoder.sh`` file in order to control the bitrate of the output files. 





