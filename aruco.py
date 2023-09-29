import rclpy.time
import cv2
import numpy as np
from cv2 import aruco
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener
import sys


def detect_aruco(image):
    '''
    Description:    Function to perform aruco detection and return each detail of aruco detected 
                    such as marker ID, distance, angle, width, center point location, etc.

    Args:
        image                   (Image):    Input image frame received from respective camera topic

    Returns:
        center_aruco_list       (list):     Center points of all aruco markers detected
        distance_from_rgb_list  (list):     Distance value of each aruco markers detected from RGB camera
        angle_aruco_list        (list):     Angle of all pose estimated for aruco marker
        width_aruco_list        (list):     Width of all detected aruco markers
        ids                     (list):     List of all aruco marker IDs detected in a single frame 
    '''

    # Convert input BGR image to GRAYSCALE for aruco detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use aruco parameters: Dictionary 4x4_50 (4x4 only until 50 aruco IDs)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

    # Detect aruco markers in the image and store 'corners' and 'ids'
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)

    # Initialize lists to store detected aruco marker details
    center_aruco_list = []
    distance_from_rgb_list = []
    angle_aruco_list = []
    width_aruco_list = []

    # Check if any markers were detected
    if ids is not None:
        for i in range(len(ids)):
            # Calculate area and width of detected aruco marker
            area, width = calculate_rectangle_area(corners[i][0])
            if area < aruco_area_threshold:
                # Remove markers that are too small (far away from the camera)
                continue
            
            # Estimate pose of the aruco marker
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], size_of_aruco_m, cam_mat, dist_mat)
            
            # Calculate the angle of the pose estimated
            rot_mat = cv2.Rodrigues(rvec)[0]
            r = R.from_matrix(rot_mat)
            angles = r.as_euler('xyz', degrees=True)
            corrected_angle = (0.788 * angles[0]) - ((angles[0] ** 2) / 3160)
            
            # Append detected marker details to the respective lists
            center_aruco_list.append(np.mean(corners[i], axis=0))
            distance_from_rgb_list.append(tvec[0][2] / 1000.0)  # Convert mm to meters
            angle_aruco_list.append(corrected_angle)
            width_aruco_list.append(width)
    print(center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids)
    return center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids

from subscribing_node import MySubscriber
def calculate_rectangle_area(coordinates):
    '''
    Description:    Function to calculate area or detected aruco

    Args:
        coordinates (list):     Coordinates of detected aruco (4 set of (x,y) coordinates)

    Returns:
        area        (float):    Area of detected aruco
        width       (float):    Width of detected aruco
    '''
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    x3, y3 = coordinates[2]
    x4, y4 = coordinates[3]

    # Calculate the width of the aruco marker
    width = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Calculate the area of the aruco marker (assuming it's a rectangle)
    area = width * width

    return area, width

# Other variables and matrices as defined in the original code
aruco_area_threshold = 1500
cam_mat = np.array([[931.1829833984375, 0.0, 640.0], [0.0, 931.1829833984375, 360.0], [0.0, 0.0, 1.0]])
dist_mat = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
size_of_aruco_m = 0.15

class FrameListener(Node):
    def __init__(self):
        super().__init__('sample_tf2_frame_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(1.0, self.on_time)
    def on_timer(self):
        from_frame_rel = 'obj_1'                                                                        # frame from which transfrom has been sent
        to_frame_rel = 'base_link'                                                                      # frame to which transfrom has been se
        try:
            t = self.tf_buffer.lookup_transform( to_frame_rel, from_frame_rel, rclpy.time.Time())       # look up for the transformation between 'obj_1' and 'base_link' frames
            self.get_logger().info(f'Successfully received data!')
        except TransformException as e:
            self.get_logger().info(f'Could not transform {to_frame_rel} to {from_frame_rel}: {e}')
            return
        # Logging transform data...
        self.get_logger().info(f'Translation X:  {t.transform.translation.x}')
        self.get_logger().info(f'Translation Y:  {t.transform.translation.y}')
        self.get_logger().info(f'Translation Z:  {t.transform.translation.z}')
        self.get_logger().info(f'Rotation X:  {t.transform.rotation.x}')                                # NOTE: rotations are in quaternions
        self.get_logger().info(f'Rotation Y:  {t.transform.rotation.y}')
        self.get_logger().info(f'Rotation Z:  {t.transform.rotation.z}')
        self.get_logger().info(f'Rotation W:  {t.transform.rotation.w}')
class aruco_tf(Node):
    '''
    __CLASS__

    Description:    Class which servers purpose to define process for detecting aruco marker and publishing tf on pose estimated.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class aruco_tf
                        All classes have a function called _init_(), which is always executed when the class is being initiated.
                        The _init_() function is called automatically every time the class is being used to create a new object.
                        You can find more on this topic here -> https://www.w3schools.com/python/python_classes.asp
        '''

        super().__init__('aruco_tf_publisher')                                          # registering node

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw', self.depthimagecb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        image_processing_rate = 0.2                                                     # rate of time to process image (seconds)
        self.bridge = CvBridge()                                                        # initialise CvBridge object for image conversion
        self.tf_buffer = tf2_ros.buffer.Buffer()                                        # buffer time used for listening transforms
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)                                    # object as transform broadcaster to send transform wrt some frame_id
        self.timer = self.create_timer(image_processing_rate, self.process_image)       # creating a timer based function which gets called on every 0.2 seconds (as defined by 'image_processing_rate' variable)
        
        self.cv_image = None                                                            # colour raw image variable (from colorimagecb())
        self.depth_image = None  
        self.frame_listener = self.FrameListener()  # Create an instance of FrameListener
                                                       # depth image variable (from depthimagecb())
           
import cv2
from cv_bridge import CvBridge

def depthimagecb(self, data):
    '''
    Description:    Callback function for aligned depth camera topic. 
                    Use this function to receive image depth data and convert to CV2 image

    Args:
        data (Image):    Input depth image frame received from aligned depth camera topic

    Returns:
    '''

    

    # Initialize a CvBridge object
    bridge = CvBridge()

    try:
        # Use the CvBridge to convert the ROS Image message to a CV2 image
        depth_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        
        # You can now work with the 'depth_image' as a regular CV2 image
        # For example, you can display it using cv2.imshow()
        cv2.imshow('Depth Image', depth_image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
        
    except CvBridgeError as e:
        # Handle any potential CvBridge errors
        print("CvBridge Error:", e)

#colour image
import cv2
from cv_bridge import CvBridge

def colorimagecb(self, data):
    '''
    Description:    Callback function for color camera raw topic.
                    Use this function to receive raw image data and convert to CV2 image

    Args:
        data (Image):    Input color raw image frame received from image_raw camera topic

    Returns:
    '''

   

    # Initialize a CvBridge object
    bridge = CvBridge()

    try:
        # Use the CvBridge to convert the ROS Image message to a CV2 image
        color_image = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        
        # You can now work with the 'color_image' as a regular CV2 image
        # For example, you can display it using cv2.imshow()
        cv2.imshow('Color Image', color_image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()
        
    except CvBridgeError as e:
        # Handle any potential CvBridge errors
        print("CvBridge Error:", e)

def process_image(self):
        # These are the camera parameters obtained from CameraInfo topic
        sizeCamX = 1280
        sizeCamY = 720
        centerCamX = 640
        centerCamY = 360
        focalX = 931.1829833984375
        focalY = 931.1829833984375
        
        # Get ArUco marker details from detect_aruco_center function
        center_aruco_list, distance_from_rgb_list, angle_aruco_list, width_aruco_list, ids = self.detect_aruco_center(image)

        # Initialize TF2 ROS broadcaster
        tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Loop over detected ArUco markers
        for i, aruco_id in enumerate(ids):
            # Correct the input ArUco angle
            corrected_angle = (0.788 * angle_aruco_list[i]) - ((angle_aruco_list[i] ** 2) / 3160)

            # Calculate quaternions from roll, pitch, and corrected yaw (yaw is the corrected ArUco angle)
            r = R.from_euler('xyz', [0, 0, corrected_angle], degrees=True)
            quaternion = r.as_quat()

            # Get RealSense depth and convert to meters
            depth_meters = distance_from_rgb_list[i] / 1000.0

            # Rectify x, y, z based on focal length, center value, and image size
            cX, cY = center_aruco_list[i][0]
            x = depth_meters * (sizeCamX - cX - centerCamX) / focalX
            y = depth_meters * (sizeCamY - cY - centerCamY) / focalY
            z = depth_meters

            # Mark center points on the image frame
            cv2.circle(image, (int(cX), int(cY)), 5, (0, 0, 255), -1)

            # Publish transform between camera_link and ArUco marker
            camera_link_to_aruco = tf2_ros.TransformStamped()
            camera_link_to_aruco.header.stamp = rospy.Time.now()
            camera_link_to_aruco.header.frame_id = 'camera_link'
            camera_link_to_aruco.child_frame_id = f'cam_{aruco_id}'
            camera_link_to_aruco.transform.translation.x = x
            camera_link_to_aruco.transform.translation.y = y
            camera_link_to_aruco.transform.translation.z = z
            camera_link_to_aruco.transform.rotation.x = quaternion[0]
            camera_link_to_aruco.transform.rotation.y = quaternion[1]
            camera_link_to_aruco.transform.rotation.z = quaternion[2]
            camera_link_to_aruco.transform.rotation.w = quaternion[3]
            tf_broadcaster.sendTransform(camera_link_to_aruco)

            # Lookup transform between base_link and obj frame
            try:
                trans = self.tfBuffer.lookup_transform('base_link', f'cam_{aruco_id}', rospy.Time(0), rospy.Duration(1.0))
                # Publish transform between obj frame and base_link
                obj_to_base = tf2_ros.TransformStamped()
                obj_to_base.header.stamp = rospy.Time.now()
                obj_to_base.header.frame_id = 'base_link'
                obj_to_base.child_frame_id = f'obj_{aruco_id}'
                obj_to_base.transform = trans.transform
                tf_broadcaster.sendTransform(obj_to_base)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue

        # Show the image with detected markers and center points
        cv2.imshow('Aruco Markers', image)
        cv2.waitKey(1)
# Import the subscribing node class



def main():
    '''
    Description:    Main function which creates a ROS node and spin around for the aruco_tf class to perform it's task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('aruco_tf_process')                    # creating ROS node

    node.get_logger().info('Node created: Aruco tf process')        # logging information
    subscribing_node = MySubscriber()
    aruco_tf_class = aruco_tf()                                     # creating a new object for class 'aruco_tf'

    rclpy.spin(aruco_tf_class)                                      # spining on the object to make it alive in ROS 2 DDS

    aruco_tf_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':
    '''
    Description:    If the python interpreter is running that module (the source file) as the main program, 
                    it sets the special _name_ variable to have a value “_main_”. 
                    If this file is being imported from another module, _name_ will be set to the module’s name.
                    You can find more on this here -> https://www.geeksforgeeks.org/what-does-the-if-_name-__main_-do/
    '''

    main()        