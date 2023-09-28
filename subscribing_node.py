import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class MySubscriber(Node):
    def __init__(self):
        super().__init__('my_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',  # Replace with your desired topic
            self.image_callback,
            10  # QoS profile depth
        )

    def image_callback(self, msg):
        # Implement your callback logic here
        # This function will be called when a message is received on the subscribed topic
        pass

def main():
    rclpy.init()
    node = MySubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
