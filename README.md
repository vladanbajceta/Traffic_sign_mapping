# Traffic_sign_mapping

**Mapping traffic signs using GPX and video files.**
## The idea behind the project
The idea for this project was to create a simple tool for surveying traffic sign layouts on urban or country road sides without the use of expensive survey equipment. 

![](https://github.com/1230grams/Traffic_sign_mapping/blob/master/traffic_signs.gif?raw=true)

 
## Prerequisite
The GPX file recorded for this repo was done using [Geo Tracker](https://play.google.com/store/apps/details?id=com.ilyabogdanovich.geotracker&hl=sr). 
For reading video metadata i used [exiftools](https://exiftool.org/).

## Description
* I created a 2D spline from gpx data (Time = f(Lattitude, Longitude)) to which I input video timestamps to interpolate the exact coordinate the video frames were recorded at. 
* Next I used tensorflow to load a pretrained model I found on [THIS](https://github.com/aarcosg/traffic-sign-detection) github repo, and detect traffic signs on video frames and save those frames.
* Lastly i used Folium to display the saved frames as popups on clickable markers on a map.  

## Data
You can download the data from these links
* [Sample data - video and gpx files](https://drive.google.com/drive/folders/1WGGuoO9ZqqNkTFZLdVBmFOVBYLCIjcqr?usp=sharing)
* [Weights files - PB files](https://drive.google.com/drive/folders/1Y5YD9j2xio2uv2akseHdhUsFx0z2WfAV?usp=sharing)

## Result
![](https://github.com/1230grams/Traffic_sign_mapping/blob/master/Map.gif?raw=true)
