import base64
import fractions
import os
import re
import subprocess
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt, atan2, pi
from typing import Tuple, List
import pathlib

import cv2
import numpy as np
import piexif
import tensorflow as tf
from PIL import Image
from folium import IFrame, Map, Popup, Marker, Icon
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from scipy.interpolate import splprep, splev

EarthRadius = 6371008.8


# ================================================================================================================================

class GpxHandler:
    """
    -Takes a GPX file created by the 'Geo Tracker' aplication *(https://play.google.com/store/apps/details?id=com.ilyabogdanovich.geotracker&hl=sr)*
    -Creates a 2D spline using lattitude, longitude and timestamp values
    -interpolates latitude and longitude values from input timestamp(float) values
    """

    def __init__(self, path):

        """
        path -- path to gpx file
        """

        file = open(path, "r")
        f = file.read()
        lat = re.findall('(?<=lat=").*?(?=\")', f)
        self.lat = [float(l) for l in lat]
        lon = re.findall('(?<=lon=").*?(?=\")', f)
        self.lon = [float(l) for l in lon]

        self.time_gpx = re.findall('(?<=<time>).*?(?=Z\<\/time>)', f)
        self.time = [datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%S.%f') for stamp in self.time_gpx]
        self.creation_time = self.time[0]
        self.timestamps = self.time[1:]
        self.timestamps_float = [t.timestamp() for t in self.timestamps]

    def calculate_distance(self) -> List[float]:
        """
        Calculates distance between GPX points
        """
        dist = []
        for i in range(len(self.lat) - 1):
            lat1 = radians(self.lat[i])
            lon1 = radians(self.lon[i])
            lat2 = radians(self.lat[i + 1])
            lon2 = radians(self.lon[i + 1])
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            d = sin(delta_lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon * 0.5) ** 2
            c = 2 * EarthRadius * asin(sqrt(d))
            dist.append(c)
        return dist

    def calculate_speed(self) -> List[float]:
        """
        Calculates speed between GPX points
        """
        dist = self.calculate_distance()
        timedeltas = [self.timestamps[i + 1].timestamp() - self.timestamps[i].timestamp() for i in
                      range(len(self.timestamps) - 1)]
        speed = [d / t for d, t in zip(dist, timedeltas)]

        return speed

    def calculate_bearing(self) -> List[float]:
        """
         Calculates bearing between GPX points
         """
        bearing = []
        for i in range(len(self.lat) - 1):
            lat1 = radians(self.lat[i])
            lon1 = radians(self.lon[i])
            lat2 = radians(self.lat[i + 1])
            lon2 = radians(self.lon[i + 1])
            delta_lon = lon2 - lon1
            x = sin(delta_lon) * cos(lat2)
            y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lon)
            theta = atan2(x, y)
            theta = (theta + pi) % pi
            bearing.append(theta)
        return bearing

    def get_coordinates(self, timestamps_float: List[float]) -> Tuple[float, float, float]:
        """
        creates a 3D spline with:
         x = lattitude,
         y = longitude,
         z = timestamp vales
         Returns u values in the same domain as the z axis in order to calculate the x and y values at exact moments in time

        Parameters
        ----------
        timestamps_float -- time value

        Returns
        -------
        coordinates -- Tuple(Lattitude(float), Longitude(float))
        """
        coordinates = []
        dist = self.calculate_distance()
        tck, u = splprep([self.lat[:-1],
                          self.lon[:-1],
                          self.timestamps_float[:-1]],
                         u=self.timestamps_float[:-1],
                         ub=self.timestamps_float[0],
                         ue=self.timestamps_float[-1]
                         )
        for new_timestamp in timestamps_float:
            gpx_worked = True
            if new_timestamp < self.timestamps_float[0] or new_timestamp > self.timestamps_float[-1]:
                gpx_worked = False
                c = (None, None)
                coordinates.append(c)
            else:
                lon_t, lat_t, time_t = splev(new_timestamp, tck)
                c = (float(lon_t), float(lat_t))
                coordinates.append(c)
                pass
        if gpx_worked == False:
            print('some coordinates could not be calculated, gpx was not recorded at all times')

        return coordinates


# ================================================================================================================================

class VideoFrameHandler:
    """
    -Takes video and calculates timestamps(datetime) for every frame of the video
    -Requires exidtool (https://exiftool.org/)
    """

    def __init__(self, path):
        """
        Parameters
        ----------
        path -- path to video
        """
        self.path = path
        self.cap = cv2.VideoCapture(self.path)

    def get_video_timestamps(self, time_float=True):
        """
        Calculates video timestampes

        Parameters
        ----------
        time_float -- if True returns timestamps in float, if false returnd datetime

        Returns
        -------
        video timestamps float or datetime
        """
        exe = '/usr/bin/exiftool'
        process = subprocess.Popen([exe, self.path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,
                                   universal_newlines=True)
        metadata = {}
        for output in process.stdout:
            info = {}
            line = output.strip().split(' :')
            info[line[0].strip()] = line[1:]
            metadata = {**metadata, **info}


        video_duration = datetime.strptime(metadata['Duration'][0][1:], '%H:%M:%S')
        video_duration = timedelta(hours=video_duration.hour, minutes=video_duration.minute,
                                   seconds=video_duration.second)
        video_create_date = datetime.strptime(metadata['Create Date'][0][1:], '%Y:%m:%d %H:%M:%S')
        video_start = video_create_date - video_duration
        self.cap = cv2.VideoCapture(self.path)

        video_frame_timestamp = [video_start]
        while (self.cap.isOpened()):
            frame_exists, curr_frame = self.cap.read()
            if frame_exists:
                video_frame_timestamp.append(
                    video_frame_timestamp[-1] + timedelta(0, 0, self.cap.get(cv2.CAP_PROP_POS_MSEC)))
            else:
                break

        if time_float == True:
            video_frame_timestamp = [t.timestamp() for t in video_frame_timestamp]
        else:
            pass

        return video_frame_timestamp


# ================================================================================================================================

def degrees_fractions(dd: float) -> [Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Converts decimal degrees to DMS

    Parameters
    ----------
    dd -- Decimal degrees (Longitude or Lattitude)

    Returns
    -------
    Degrees in DMS
    """
    mnt, sec = divmod(dd * 3600, 60)
    deg, mnt = divmod(mnt, 60)

    # seconds need to be rounded
    # piexif displays error with large fractions
    # this reduces accuracy

    sec = round(sec)

    [deg, mnt, sec] = [fractions.Fraction(i) for i in [deg, mnt, sec]]

    return (deg.numerator, deg.denominator), (mnt.numerator, mnt.denominator), (sec.numerator, sec.denominator)


def set_coordinates(filename, coordinates: Tuple[float, float]):
    """
    Geotags saved image

    Parameters
    ----------
    filename -- path to image
    coordinates -- Tuple containing lattitude and longitude

    """
    lat = degrees_fractions(coordinates[0])
    lon = degrees_fractions(coordinates[1])

    gps_ifd = {
        piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
        piexif.GPSIFD.GPSAltitudeRef: 0,
        piexif.GPSIFD.GPSLatitude: lat,
        piexif.GPSIFD.GPSLongitude: lon,
        piexif.GPSIFD.GPSLatitudeRef: 'N',
        piexif.GPSIFD.GPSLongitudeRef: 'E',
    }

    exif_dict = {"GPS": gps_ifd}
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, filename)


# ================================================================================================================================

class TrafficSignDetect:
    """
    -Uses tensorflow to detect a traffic sign
    -Saves image of detected traffic sign
    -Creates folium map of detected signs
    """

    def __init__(self, file_input, path_to_pb_file, path_to_labels, file_output='./Saved_frames/', num_classes=3,
                 create_map=True, play_video=False):
        """
        Parameters
        ----------
        file_input -- video path
        path_to_pb_file -- pb file path
        path_to_labels -- pbtxt file
        file_output -- output path
        num_classes -- number of classes
        create_map --
        play_video --
        """

        self.create_map = create_map
        self.play_video = play_video
        self.cap = cv2.VideoCapture(file_input)
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_pb_file, 'rb') as fid:
                self.serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

        label_map = label_map_util.load_labelmap(path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))

        self.out = cv2.VideoWriter(file_output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0,
                                   (frame_width, frame_height))
        if create_map == True:
            self.map = Map(location=[45.568028, 19.6470089], zoom_start=13, tiles='Stamen Terrain')

    def photos_to_map(self, image_name, coordinate):
        """
        Parameters
        ----------
        image_name -- image name
        coordinate -- coordinate tuple

        Returns
        -------
        Adds markers to self.map
        Clicking on the markers displays popup containing image of detected traffic sign

        """
        image = Image.open(image_name)
        image = image.resize((380, 270), Image.ANTIALIAS)
        image.save(image_name, quality=100)
        encoded = base64.b64encode(open(image_name, 'rb').read())
        html = '<img src="data:image/png;base64,{}">'.format
        iframe = IFrame(html(encoded.decode('UTF-8')), width=380, height=270)
        popup = Popup(iframe, max_width=500)
        icon = Icon(color='red', icon='ok')
        Marker(location=[coordinate[0], coordinate[1]], popup=popup, icon=icon).add_to(self.map)

    def detect(self, coordinates):
        """
        Detection using tensorflow

        Parameters
        ----------
        coordinates of video timeframes -- tuple with coordinates


        """
        with self.detection_graph.as_default():

            with tf.Session(graph=self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # if output directory is not empty, remove all images
                if os.listdir('./Saved_frames/') != 0:
                    files = os.listdir('./Saved_frames/')
                    for f in files:
                        os.remove('./Saved_frames/' + f)

                # set a frame counter
                counter = 0

                while (self.cap.isOpened()):
                    ret, frame = self.cap.read()
                    counter += 1

                    image_np_expanded = np.expand_dims(frame, axis=0)

                    if ret == True:
                        #Detection.
                        (boxes, scores, classes, num) = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})

                        # if score is above 80% box is visualised and image is saved
                        # counter % 10 limits the number of frames saved

                        if scores[0][0] > 0.8 and counter % 10 == 0:
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                frame,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                self.category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8)

                            file_name = os.path.join('./Saved_frames/', str(counter) + '.jpg')

                            cv2.imwrite(file_name, frame)

                            #geotag image
                            set_coordinates(file_name, coordinates[counter])

                            if self.create_map == True:
                                self.photos_to_map(file_name, coordinates[counter])

                        self.out.write(frame)

                        if self.play_video == True:
                            cv2.imshow('Traffic sign detection', frame)

                        # Close window when "Q" button pressed
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        break

        # release the video capture and video, write objects, save map to html
        self.map.save(outfile='Map.html')
        self.cap.release()
        self.out.release()

        # Closes all the frames
        cv2.destroyAllWindows()



if __name__ == '__main__':

    gpx_path = './Sample_data/sample_file.gpx'
    video_path = './Sample_data/sample_video.mp4'

    GH = GpxHandler(gpx_path)
    VFH = VideoFrameHandler(video_path)
    TSD = TrafficSignDetect(file_input=video_path, path_to_pb_file='./PB_FILES/ssd.pb',
                            path_to_labels='./Labels/label_map.pbtxt', play_video=True)
    # For better accuracy use path_to_pb_file='./PB_FILES/faster_r_cnn.pb'

    video_frame_time = VFH.get_video_timestamps()
    video_frame_coordinates = GH.get_coordinates(video_frame_time)

    TSD.detect(video_frame_coordinates)
