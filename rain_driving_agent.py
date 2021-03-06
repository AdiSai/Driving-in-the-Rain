# Adapted from https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/basic_agent.py


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import skimage.io
import numpy as np
import tensorflow as tf
try:
    sys.path.append('./image_training')
except IndexError:
    pass
from image_training import training_v2 as Network

tf.compat.v1.disable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select GPU device
pre_trained_model_path = './image_training/model/trained/model'
tf.compat.v1.reset_default_graph()


try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append('../PythonAPI/carla')
except IndexError:
	pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

class RainDrivingAgent(Agent):

    def __init__(self, vehicle, target_speed=20):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(RainDrivingAgent, self).__init__(vehicle)

        self._proximity_tlight_threshold = 5.0  # meters
        self._proximity_vehicle_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.4,
            'K_I': 0,
            'dt': 1.0/20.0}
        self._local_planner = LocalPlanner(
            self._vehicle, opt_dict={'target_speed' : target_speed,
            'lateral_control_dict':args_lateral_dict})
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None
        
        # create the camera
        camera_bp = self._world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(1920//2))
        camera_bp.set_attribute('image_size_y', str(1080//2))
        camera_bp.set_attribute('fov', str(90))
        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self._camera = self._world.spawn_actor(camera_bp, camera_transform, attach_to=self._vehicle)
        self._camera.listen(lambda image: self._process_image(image))
        self._curr_image = None
        self._save_count = 0
    
    def _process_image(self, image):
        self._curr_image = image
        """
        image_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_array = np.reshape(image_array, (image.height, image.width, 4))
        image_array = image_array[:, :, :3]
        image_array = image_array[:, :, ::-1]
        """
        file_name = 'curr.jpg'
        image.save_to_disk(file_name)
        def parse_file(filename):
            image_string = tf.io.read_file(filename)  
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)  
            return tf.cast(image_decoded, tf.float32)/255.0
        whole_path = [file_name]
        filename_tensor = tf.convert_to_tensor(value=whole_path, dtype=tf.string)     
        dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
        dataset = dataset.map(parse_file)    
        dataset = dataset.prefetch(buffer_size=1)
        dataset = dataset.batch(batch_size=1).repeat()  
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        image_array = iterator.get_next() 
        output = Network.inference(image_array, is_training=False, middle_layers=12)
        output = tf.clip_by_value(output, 0., 1.)
        output = output[0,:,:,:]
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True   
        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.Session(config=config) as sess: 
            with tf.device('/gpu:0'): 
                saver.restore(sess, pre_trained_model_path)
                derained, ori = sess.run([output, image_array])              
                derained = np.uint8(derained* 255.)
                skimage.io.imsave('curr_derained.png', derained)
                if self._save_count % 6 == 0:
                    image.save_to_disk('_out/%08d_orig' % image.frame)
                    skimage.io.imsave('_out/%08d_derained.png' % image.frame, derained)
        self._save_count += 1

    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)

        self._local_planner.set_global_plan(route_trace)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def run_step(self, debug=False):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        light_state, traffic_light = self._is_light_red(lights_list)
        if light_state:
            if debug:
                print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

            self._state = AgentState.BLOCKED_RED_LIGHT
            hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            control = self._local_planner.run_step(debug=debug)

        return control

    def done(self):
        """
        Check whether the agent has reached its destination.
        :return bool
        """
        return self._local_planner.done()
