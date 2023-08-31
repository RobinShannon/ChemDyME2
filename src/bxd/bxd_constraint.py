from abc import abstractmethod
import numpy as np
from ase.io import read, write
import os
from copy import deepcopy
import src.bxd.bxd_bound as bound
import src.bxd.bxd_box as b
import src.utility.bxd_plotter as plt

class BXD:
    """
    bxd base class, there are currently three derived versions of bxd:
    1. Adaptive bxd, controlling the placement of bxd boundaries.
    2. Fixed bxd, use bxd constraints to fix the dynamics in a single box. If the trajectory starts outside the box,
       a bxd inversion is applied each time the dynamics moves further away from the box unitl the dyanmics is contained
       in the box
    3. Converging bxd, uses a set of bxd boundaries generated from an adaptive bxd object and controls a converging
       run where each boundary is hit a specified number of times. This derived class also implements the free
       energy analysis methods.

    """
    def __init__(self, progress_metric, bxd_iterations = 1, connected_BXD = None):
        self.progress_metric = progress_metric
        self.steps_since_any_boundary_hit = 0
        self.box_list = []
        self.box = 0
        self.inversion = False
        self.reverse = False
        self.bound_hit = "none"
        self.s = 0
        self.old_s = 0
        self.completed_runs = 0
        self.furthest_progress = 0
        self.geometry_of_furthest_point = 0
        self.bxd_iterations = bxd_iterations
        self.inversion = False
        self.connected_BXD = connected_BXD
        self.all_data = []
        self.all_bounds = []

    def __len__(self):
        return len(self.box_list)

    def __getitem__(self, item):
        return self.box_list[item]

    @abstractmethod
    def update(self, mol, decorrelated=True):
        pass

    @abstractmethod
    def boundary_check(self):
        pass

    def reached_end(self, projected):
        pass

    def del_constraint(self, mol):
        """
        interfaces with the progress metric to return the delta_phi for the constraint at whichever boundary is hit
        :param mol: ASE atoms object with the current geometry
        :return:
        """
        norm = 0
        if self.bound_hit == 'upper':
            norm = self.box_list[self.box].upper.n
        if self.bound_hit == 'lower':
            norm = self.box_list[self.box].lower.n
        delta = self.progress_metric.collective_variable.get_delta(mol, norm)
        return delta

    def path_del_constraint(self, mol):
        """
        Interfaces with the progress metric to return the del_phi for the path constraint parallel to the current path
        segment
        :param mol: ASE atoms object with the current geometry
        :return:
        """
        return self.progress_metric.get_path_delta(mol)

    def initialise_files(self):
        pass

    def complete(self):
        return self.completed_runs >= self.bxd_iterations

    def print_bounds(self, file="bounds.txt"):
        """
        Prints the bxd boundaries to file.
        :param file:
        :return:
        """
        f = open(file, 'w')
        f.write("bxd boundary list \n\n")
        string = ("Boundary\t" + str(0) + "\tD\t=\t" + str(self.box_list[0].lower.d) + "\tn\t=\t" +
                  str(self.box_list[0].lower.n) + "\n")
        string = string.replace('\n', '')
        f.write(string + "\n")
        for i in range(0, len(self.box_list)):
            string = "Boundary\t" + str(i+1) + "\tD\t=\t" + str(self.box_list[i].upper.d) + "\tn\t=\t" + \
                     str(self.box_list[i].upper.n) + "\tS\t=\t" + str(self.box_list[i].upper.s_point)
            string = string.replace('\n', '')
            f.write(string + "\n")
        f.close()

    def hit(self, mol):
        s = self.progress_metric.collective_variable.get_s(mol)
        if self.box_list[self.box].upper.hit(s, 'up'):
            return True
        elif self.box_list[self.box].lower.hit(s, 'down'):
            return True
        else:
            return False

    def close(self, temp_dir):
        pass


class Adaptive(BXD):
    """
    Derived bxd class controlling and Adaptive bxd run. This class controls the adaptive placing of boundaries and
    determines whether or not a boundary has been hit and whether an inversion is required.
    :param progress_metric: ProgressMetric object which manages the CollectiveVariable representation, transforms
                            the current MD geometry into "progess" between the bxd start and end points and
                            optionally contains a path object if bxd is following a guess path
    :param fix_to_path: Boolean, DEFAULT = False.
                        If True all boundaries will be aligned such that they are perpendicular to the path. Should
                        only be used for a curve progress_metric
    :param adaptive_steps: Integer, DEFAULT = 10000
                           Number of MD steps sampled before placing a new boundary in the direction of bxd progress
    :param epsilon: Float. DEFAULT = 0.95
                           Used in histograming to determine the proportion of the adaptive sampling points which
                           should be outside the new adaptive boundary. The cumulative probability of points outside
                           the new boundary should be ( 1.0 - epsilon)
    :param reassign_rate: DEFAULT = 5.
                          If an adaptive bound has not been hit after adaptive_steps * reassign_rate then the
                          boundary will be moved based on new sampling
    :param one_direction: Boolean, DEFAULT = False,
                          If True, then the adaptive bxd run will be considerd complete once it reached the
                          progress_metric end_point and will not attempt to place extra boundaries in the reverse
                          direction
    """

    def __init__(self, progress_metric, fix_to_path=True, adaptive_steps=10000, epsilon=0.95, reassign_rate=5,
                 one_direction=False, connected_BXD = None):
        # call the base class init function to set up general parameters
        super(Adaptive, self).__init__(progress_metric, connected_BXD=connected_BXD)
        self.fix_to_path = fix_to_path
        self.one_direction = one_direction
        self.adaptive_steps = adaptive_steps
        self.histogram_boxes = int(np.sqrt(adaptive_steps))
        self.epsilon = epsilon
        self.reassign_rate = reassign_rate
        # set up the first box based upon the start and end points of the progress metric, others will be added as bxd
        # progresses
        s1, s2 = self.progress_metric.start, self.progress_metric.end
        b1, b2 = self.get_starting_bounds(s1, s2)
        box = b.BXDBox(b1, b2, 'adap')
        # Add starting box to box list
        self.box_list.append(box)
        # If we have just created a new box, the new_box parameter indicates that a representative geometry should be
        # stored for this box
        self.new_box = False



    def update(self, mol, decorrelated=True ):
        """
        General book-keeping method. Takes an ASE atoms object, stores data from progress_metric at the current
        geometry, and calls the update_adaptive_bounds and boundary_check methods to add new boundaries and to keep
        track of which box we are in and whether and inversion is necessary
        :param mol: ASE atoms object
        :return: none
        """

        # If this is the first step in a new box then store its geometry in the box object, then set new_box to False
        if self.new_box:
            self.box_list[self.box].geometry = mol.copy()
        self.new_box = False

        # get the current value of the collective variable and the progress data
        self.s = self.progress_metric.collective_variable.get_s(mol)
        projected_data = self.progress_metric.project_point_on_path(self.s)
        distance_from_bound = self.progress_metric.get_dist_from_bound(self.s, self.box_list[self.box].lower)

        # Check if we have reached the start or end points of the bxd run
        if self.box_list[self.box].type == "normal":
            self.reached_end(projected_data)

        # Reset inversion related parameters to False
        self.inversion = False
        self.bound_hit = "none"

        # Check whether we are in an adaptive sampling regime.
        # If so update_adaptive_bounds checks current number of samples and controls new boundary placement
        if self.box_list[self.box].type == "adap":
            self.update_adaptive_bounds()

        # If we have sampled for a while and not hit the upper bound then reassign the boundary.
        # How often the boundary is reassigned depends upon the reassign_rate parameter
        if self.box_list[self.box].type == "fixed" and len(self.box_list[self.box].data) > \
                self.reassign_rate * self.adaptive_steps:
            self.reassign_boundary()
            self.reassign_rate *= 2

        # Check whether a velocity inversion is needed, either at the boundary or back towards the path
        self.inversion = self.boundary_check() or self.progress_metric.reflect_back_to_path()
        if self.inversion and self.bound_hit == 'none':
            self.bound_hit = 'path'

        # update counters depending upon whether a boundary has been hit
        if self.inversion:
            self.steps_since_any_boundary_hit = 0
        else:
            self.steps_since_any_boundary_hit += 1
            self.old_s = self.s
            # Provided we are close enough to the path, store the data of the current point
            if not self.progress_metric.reflect_back_to_path():
                self.box_list[self.box].data.append((self.s, projected_data, distance_from_bound))
                # If this is point is the largest progress metric so far then store its geometry.
                # At the end of the run this will print the geometry of the furthest point along the bxd path
                if projected_data > self.furthest_progress:
                    self.furthest_progress = projected_data
                    self.geometry_of_furthest_point = mol.copy()

    def update_adaptive_bounds(self):
        """
        If a box is in an adaptive sampling regime, this method checks the number of data points and determines whether
        or not to add a new adaptive bound. If bxd is in the reverse direction the new a new box is created between the
        current box and the previous one the self.box_list.
        :return:
        """
        # If adaptive sampling has ended then add a boundary based on sampled data
        if len(self.box_list[self.box].data) > self.adaptive_steps:
            # Fist indicate the box no longer needs adaptive sampling
            self.box_list[self.box].type = "normal"
            # If not reversing then update the upper boundary and add a new adaptive box on the end of the list
            if not self.reverse:
                # Histogram the box data to get the averaged top and bottom values of s based on the assigned epsilon
                self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
                bottom = self.box_list[self.box].bot
                top = self.box_list[self.box].top
                # use the bottom and top s to generate a new upper bound
                b1 = self.convert_s_to_bound(bottom, top)
                # copy this bound as it will form the lower bound of the next box
                b2 = deepcopy(b1)
                # copy the current upper bound which will be used for the new box, this upper bound is a dummy bound
                # which can never be hit
                b3 = deepcopy(self.box_list[self.box].upper)
                b3.invisible = True
                b3.s_point = self.box_list[self.box].upper.s_point
                # assign b1 to the current upper bound and create a new box which is added to the end of the list
                self.box_list[self.box].upper = b1
                self.box_list[self.box].upper.transparent = True
                new_box = b.BXDBox(b2, b3, 'adap')
                self.box_list.append(new_box)
            elif self.reverse:
                # same histograming procedure as above but this time it is the lower bound which is updated
                self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
                bottom = self.box_list[self.box].bot
                top = self.box_list[self.box].top
                b1 = self.convert_s_to_bound(bottom, top)
                b2 = deepcopy(b1)
                b3 = deepcopy(self.box_list[self.box].lower)
                self.box_list[self.box].lower = b1
                new_box = b.BXDBox(b3, b2, 'adap')
                # at this point we partition the  current box into two by inserting a new box at the correct point
                # in the box_list
                self.box_list.insert(self.box, new_box)
                self.box_list[self.box].active = True
                self.box_list[self.box].upper.transparent = False
                self.box += 1
                self.box_list[self.box].lower.transparent = True

    def reached_end(self, projected):
        """
        Method to check whether bxd has either reached the end point and needs reversing or whether the bxd run should
        be considered finished
        :param projected: Projected distance along path
        :return:
        """
        # If we are going forward check if the progress metric end point (either a distance or number of boxes) has been
        # reached. If one_direction = True, then stop the adaptive run, otherwise set reverse = True.
        if not self.reverse:
            if self.progress_metric.end_type == 'distance':
                if projected >= self.progress_metric.full_distance:
                    del self.box_list[-1]
                    if self.one_direction:
                        self.completed_runs += 1
                    else:
                        self.reverse = True
                        self.box_list[self.box].type = 'adap'
                        self.box_list[self.box].data = []
                        self.progress_metric.bxd_reverse = True
            elif self.progress_metric.end_type == 'boxes':
                if self.box >= self.progress_metric.end_point:
                    del self.box_list[-1]
                    if self.one_direction:
                        self.completed_runs += 1
                    else:
                        self.reverse = True
                        self.box_list[self.box].type = 'adap'
                        self.box_list[self.box].data = []
                        self.progress_metric.bxd_reverse = True
        else:
            if self.box == 0:
                self.completed_runs += 1
                self.reverse = False
                self.progress_metric.bxd_reverse = True

    def reassign_boundary(self):
        """
        repeates the procedure of producing an adaptive bound and replaces the exsisting upper or lower bound
         depending whether self.reverse is true or false respectively
        :return:
        """
        print("re-assigning boundary")
        self.box_list[self.box].get_s_extremes(self.histogram_boxes, self.epsilon)
        bottom = self.box_list[self.box].bot
        top = self.box_list[self.box].top
        b = self.convert_s_to_bound(bottom, top)
        b2 = self.convert_s_to_bound(bottom, top)
        b.transparent = True
        if self.reverse:
            self.box_list[self.box].lower = b
            self.box_list[self.box - 1].upper = b2
        else:
            self.box_list[self.box].upper = b
            self.box_list[self.box+1].lower = b2
        self.box_list[self.box].data = []

    def convert_s_to_bound(self, low_s, high_s):
        """
        Takes appropriate high and low values of the collective variable in the current box based upon the probability
        distriubtion and creates a bxd boundary perpendicular to the path between these points.
        :param low_s: Low value of collective variable s in current box
        :param high_s: High value of collective variable s in current box
        :return:
        """
        if not self.fix_to_path:
            b = self.convert_s_to_bound_general(low_s, high_s)
        else:
            if self.reverse:
                b = self.convert_s_to_bound_on_path(low_s)
            else:
                b = self.convert_s_to_bound_on_path(high_s)
        return b

    def convert_s_to_bound_general(self, s1, s2):
        """
        General method to pick between two types of boundary placement
        :param s1: Low value of collective variable s in current box
        :param s2: High value of collective variable s in current box
        :return:
        """
        if self.reverse:
            n2 = (s2 - s1) / np.linalg.norm(s1 - s2)
            d2 = -1 * np.vdot(n2, s1)
        else:
            n2 = (s2 - s1) / np.linalg.norm(s1 - s2)
            d2 = -1 * np.vdot(n2, s2)
        b2 = bound.BXDBound(n2, d2)
        b2.s_point = s2
        return b2

    def convert_s_to_bound_on_path(self, s):
        """
        Takes an appropriate value of the collective variable in the current box based upon the probability
        distriubtion and creates a bxd boundary perpendicular to the guess path at this point.
        :param s: Value of collective variable s in current box
        :return:
        """
        n = self.progress_metric.get_norm_to_path(s)
        d = -1 * np.vdot(n, s)
        b = bound.BXDBound(n, d)
        b.s_point = s
        return b

    def boundary_check(self):
        """
        Checks whether the upper or lower boundary has been hit at the current collective variable value self.s.
        Does book-keeping regarding whether or not the dynamics may travel from one box to another
        :return: True if bxd inversion is required and False otherwise
        """
        self.bound_hit = 'none'
        # Check for hit against upper boundary
        if self.box_list[self.box].upper.hit(self.s, 'up'):
            if not self.reverse:
                self.box_list[self.box].upper.transparent = False
                self.box_list[self.box].lower.transparent = True
                self.print_snapshot()
                self.box_list[self.box].data = []
                self.box += 1
                self.new_box = True
                self.box_list[self.box].data = []
                return False
            else:
                self.bound_hit = 'upper'
                return True
        elif self.box_list[self.box].lower.hit(self.s, 'down'):
            if self.reverse and not self.box_list[self.box].type == 'adap':
                self.print_snapshot()
                self.box_list[self.box].data = []
                self.box_list[self.box].data = []
                self.box -= 1
                self.box_list[self.box].data = []
                self.new_box = True
                self.box_list[self.box].type = 'adap'
                return False
            else:
                self.bound_hit = 'lower'
                self.box_list[self.box].lower.hits += 1
                if self.reverse:
                    self.box_list[self.box].type = 'normal'
            return True
        else:
            return False

    def print_snapshot(self):

        os.makedirs("snapshots", exist_ok=True)
        os.makedirs( "snapshots/snapshot"+str(self.box), exist_ok=True)
        direct = "snapshots/snapshot"+str(self.box)
        fig = plt.bxd_plotter_2d(self.progress_metric.path.s, zoom = False, all_bounds = True)
        ar = []
        data = [d[0] for d in self.box_list[self.box].data]
        self.all_data += data
        for i in self.box_list:
            ar.append(i.upper.get_bound_array2D())
            self.all_bounds.append(i.upper.get_bound_array2D())
        fig.plot_bxd_from_array(self.all_data, self.all_bounds, save_file=True, save_root = direct)
        del fig


    def close(self, temp_dir, mol):
        """
        Close bxd gracefully and do final printing
        :param temp_dir: directory for printing output
        :param mol: ASE atoms object containing final geometry
        :return:
        """
        os.makedirs(temp_dir, exist_ok=True)
        box_geoms = open(temp_dir + '/box_geoms.xyz', 'w')
        furthest_geom = open(temp_dir + '/furthest_geometry.xyz', 'w')
        end_geom = open(temp_dir + '/final_geometry.xyz', 'w')
        for box in self.box_list:
            write(box_geoms, box.geometry, format='xyz', append=True)
        write(furthest_geom, self.geometry_of_furthest_point, format='xyz')
        write(end_geom, mol, format='xyz')

    def get_starting_bounds(self, low_s, high_s):
        if self.fix_to_path:
            b1 = self.convert_s_to_bound_on_path(low_s)
            b2 = self.convert_s_to_bound_on_path(high_s)

        else:
            n1 = (high_s - low_s) / np.linalg.norm(high_s - low_s)
            n2 = n1
            d1 = -1 * np.vdot(n2, low_s)
            d2 = -1 * np.vdot(n2, high_s)
            b1 = bound.BXDBound(n1, d1)
            b2 = bound.BXDBound(n2, d2)
        b2.invisible = True
        b1.s_point = low_s
        b2.s_point = high_s
        return b1, b2

class Fixed(BXD):

    def __init__(self, progress_metric):
        super(Fixed, self).__init__(progress_metric=progress_metric)
        s1, s2 = self.progress_metric.start, self.progress_metric.end
        b1, b2 = self.get_starting_bounds(s1, s2)
        b2.invisible = False
        box = b.BXDBox(b1, b2,'fixed')
        self.old_s = 100000
        self.box_list.append(box)
        self.active = False

    def update(self, mol, decorrelated=True):
        # update current and previous s(r) values
        self.s = self.progress_metric.collective_variable.get_s(mol)
        self.inversion = False
        self.bound_hit = "none"

        # First check whether we are inside the box
        if not self.box_list[self.box].upper.hit(self.s, 'up'):
            # if so box type is now fixed
            self.active = True

        # If we are outside the top boundary and moving further away then invert velocities
        # Check whether a velocity inversion is needed, either at the boundary or back towards the path
        self.inversion = self.boundary_check() or self.progress_metric.reflect_back_to_path()
        self.old_s = self.s

    def get_starting_bounds(self, low_s, high_s):
        n1 = (high_s - low_s) / np.linalg.norm(high_s - low_s)
        n2 = n1
        d1 = -1 * np.vdot(n2, low_s)
        d2 = -1 * np.vdot(n2, high_s)
        b1 = bound.BXDBound(n1, d1)
        b2 = bound.BXDBound(n2, d2)
        b2.invisible = True
        b1.s_point = low_s
        b2.s_point = high_s
        return b1, b2

    def boundary_check(self):
        self.bound_hit = 'none'
        # Check for hit against upper boundary
        if self.box_list[self.box].upper.hit(self.s, 'up'):
            # If we are adaptive check if we are getting closer or further away
            closer = self.box_list[self.box].upper.compare_distances_from_bound(self.s, self.old_s, 'up')
            if closer:
                return False
            else:
                self.bound_hit = 'upper'
                return True

        elif self.box_list[self.box].lower.hit(self.s, 'down'):
            closer = self.box_list[self.box].lower.compare_distances_from_bound(self.s, self.old_s, 'down')
            if closer:
                return False
            else:
                self.bound_hit = 'lower'
                return True
        else:
            return False

    def hit(self, mol):
        if self.active is False:
            return False
        else:
            s = self.progress_metric.collective_variable.get_s(mol)
            if self.box_list[self.box].upper.hit(s, 'up'):
                return True
            elif self.box_list[self.box].lower.hit(s, 'down'):
                return True
            else:
                return False


class Converging(BXD):
    """
    Derived bxd class controlling a Converging bxd run. This class keeps track of the number of hits on each
    boundary and stores data such as mean first passage times. It also determines when sufficient boundary hits have
    occurred to move to the next box. One all the data has been collected, this class also contains methods for
    generating a free energy profile
    :param progress_metric: ProgressMetric object which manages the CollectiveVariable representation, transforms
                            the current MD geometry into progress" between the bxd start and end points and
                            optionally contains a path object if bxd is following a guess path
    :param bound_file: String DEFAULT = "bounds.txt"
                       Filename containing a list of bxd boundaries from an adaptive run.
    :param geom_file: String DEFAULT = 'box_geoms.xyz'
                      Filename containing representative geometries for each box.
    :param bound_hits: Integer DEFAULT = 100
                       Number of boundary hits before moving to the next box
    :param convert_fixed_boxes: Boolean DEFAULT = False
                                Niche case where you dont have a bounds file to read from and want to create boxes
                                of a fixed size. NOT TESTED
    :param box_width: If convert_fixed_boxes = True then this defines the width of the box.
    :param number_of_boxes: If convert_fixed_boxes = True then this defines the number of boxes
    :param boxes_to_converge: List.
                              Specifies a subset of the total boxes which to converge.
                              e.g if boxes_to_converge = [3,6] then only boxes 3 to 6 inclusive will be converged
    :param print_directory: String, DEFAULT="Converging_Data"
                            Directory name for printing converging data. If this directory already exsist the new
                            data will be appended to the exsisting
    :param converge_ends: Boolean DEFAULT = False
                          If True then the start and end boxes will be fully converged. This means that the start
                          box will aim for "bound_hits" at the lower boundary and the top box will aim for
                          "bound_hits" at the top boundary
    """
    def __init__(self, progress_metric, bound_file="bounds_out.txt", geom_file='box_geoms.xyz', bound_hits=10,
                 read_from_file=True, convert_fixed_boxes=False, box_width=0, number_of_boxes=0,
                 boxes_to_converge=None, print_directory='Converging_Data', converge_ends=True, bxd_iterations = 1, plot_box_data = False):
        super(Converging, self).__init__(progress_metric, bxd_iterations = bxd_iterations)
        self.bound_file = bound_file
        self.geom_file = geom_file
        self.dir = str(print_directory)
        self.read_from_file = read_from_file
        self.converge_ends = converge_ends
        self.convert_fixed_boxes = convert_fixed_boxes
        self.box_width = box_width
        self.number_of_boxes = number_of_boxes
        self.boxes_to_converge = boxes_to_converge
        self.plot = plot_box_data
        if self.read_from_file:
            self.box_list = self.read_exsisting_boundaries(self.bound_file)

        elif self.convert_fixed_boxes:
            self.box_list = self.create_fixed_boxes(self.box_width, self.number_of_boxes, progress_metric.start_s)

        self.old_s = 0
        self.number_of_hits = bound_hits
        if boxes_to_converge is not None:
            self.start_box = self.boxes_to_converge[0]
            self.box = self.start_box
            self.end_box = self.boxes_to_converge[1]
        else:
            self.start_box = 0
            self.end_box = len(self.box_list)-1
        self.box_list[self.start_box].open_box()

    def reset(self, output_directory):
        """
        Function resetting a converging bxd object to its original state but with a different output directory
        :param output_directory:
        :return:
        """
        self.__init__(self.progress_metric,  self.bound_file, self.geom_file, self.number_of_hits, self.read_from_file,
                      self.convert_fixed_boxes, self.box_width, self.number_of_boxes, self.boxes_to_converge,
                      output_directory, self.converge_ends)

    def update(self, mol, decorrelated=True):
        """
        Does the general bxd bookkeeping and management. First gets the progress_metric data from the mol object and
        then calls functions to check whether a bxd inversion is required and whether we need to move to the next box.
        :param mol: ASE atoms object
        :return:
        """

        # update current and previous s(r) values
        self.s = self.progress_metric.collective_variable.get_s(mol)
        self.progress_metric.project_point_on_path(self.s)
        # make sure to reset the inversion and bound_hit flags to False / none
        self.inversion = False
        self.bound_hit = "none"

        # Check whether bxd direction should be changed and update accordingly
        self.reached_end()
        if self.progress_metric.reflect_back_to_path():
            self.bound_hit = "path"
           # self.box_list[self.box].last_hit = 'path'
        # Check whether a boundary has been hit and if so update the hit boundary
        self.inversion = self.boundary_check(decorrelated) or self.bound_hit is "path"

        # If there is a bxd inversion increment the stuck counter and set the steps_since_any_boundary_hit counter to 0
        if not self.inversion:
            self.box_list[self.box].points_in_box += 1
            self.box_list[self.box].data.append(self.s)



    def create_fixed_boxes(self, width, number_of_boxes, start_s):
        box_list = []
        s = deepcopy(start_s)
        lower_bound = bound.BXDBound(1.0,-1.0*deepcopy(s))
        for i in range(0,number_of_boxes):
            temp_dir = self.dir + ("/box_" + str(i))
            s += width
            upper_bound = bound.BXDBound(1.0,-1.0*deepcopy(s))
            box = box(lower_bound, upper_bound, "fixed", True, dir = temp_dir, plot = self.plot)
            box_list.append(box)
            lower_bound = deepcopy(upper_bound)
        return box_list

    def read_exsisting_boundaries(self, file):
        """
        Read bxd boundaries from a file
        :param file: string giving the filename of the bounds file
        :return:
        """
        box_list = []
        lines = open(file,"r").readlines()
        for i in range(2, len(lines)-1):
            temp_dir = self.dir + ("/box_" + str(i-2))
            words = lines[i].split("\t")
            d_lower = (float(words[4]))
            n_l = (words[7]).strip("[]\n")
            norm_lower = (n_l.split())
            for l in range(0,len(norm_lower)):
                norm_lower[l] = float(norm_lower[l])
            lower_bound = bound.BXDBound(norm_lower,d_lower)
            words = lines[i+1].split("\t")
            d_upper = (float(words[4]))
            n_u = (words[7]).strip("[]\n")
            norm_upper = (n_u.split())
            for l2 in range(0,len(norm_upper)):
                norm_upper[l2] = float(norm_upper[l2])
            upper_bound = bound.BXDBound(norm_upper,d_upper)
            box = b.BXDBox(lower_bound, upper_bound, "fixed", dir=temp_dir, plot = self.plot)
            box_list.append(box)
        return box_list

    def reached_end(self):
        """
        Checks whether the converging run either needs to be reversed or whether it is complete
        :return:
        """
        # First if we are not currently reversing check whether or not we have reached the end box and reversing should
        # be turned on.
        if self.box == self.end_box and self.reverse is False:
            # If converge_ends then make sure the final bound meets the bound_hits criteria before reversing
            if not self.converge_ends or self.criteria_met(self.box_list[self.box].upper):
                self.reverse = True
                print('reversing')
                self.box_list[self.box].upper.transparent = False
        # If we are reversing then check whether we are back in box 0 and the run is complete
        elif self.box == self.start_box and self.reverse is True:
            # If converge_ends then make sure the first bound meets the bound_hits criteria
            if not self.converge_ends or self.criteria_met(self.box_list[self.box].lower):
                self.reverse = False
                self.completed_runs += 1
                for bx in self.box_list:
                    bx.upper.hits = 0
                    bx.lower.hits = 0

    def criteria_met(self, boundary):
        """
        Check whether a boundary has exceeded the specified number of hits
        :param boundary: BXDbound object
        :return:
        """
        return boundary.hits >= self.number_of_hits


    def boundary_check(self, decorrelated):
        """
        Check upper and lower boundaries for hits and return True if an inversion is required. Also determines the mean
        first passage times for hits against each bound.
        :return: Boolean indicating whether or not a bxd inversion should be performed
        """
        self.bound_hit = 'none'
        self.box_list[self.box].milestoning_count += 1
        self.box_list[self.box].upper_non_milestoning_count += 1
        self.box_list[self.box].lower_non_milestoning_count += 1

        # Check for hit against upper boundary
        if self.box_list[self.box].upper.hit(self.s, 'up'):
            self.bound_hit = 'upper'
            if self.progress_metric.reflect_back_to_path():
                return True
            elif self.box_list[self.box].upper.transparent and not self.progress_metric.reflect_back_to_path():
                self.box_list[self.box].upper.transparent = False
                self.box_list[self.box].close_box(path=self.progress_metric.path.s)
                self.box += 1
                self.box_list[self.box].open_box()
                self.box_list[self.box].last_hit = 'lower'
                return False
            else:
                self.bound_hit = 'upper'
                if decorrelated:
                    self.box_list[self.box].upper.hits += 1
                    self.box_list[self.box].upper_rates_file.write\
                        (str(self.box_list[self.box].upper_non_milestoning_count) + '\t' + '\n')
                    if self.box_list[self.box].last_hit == 'lower':
                        self.box_list[self.box].upper_milestoning_rates_file.write\
                            (str(self.box_list[self.box].milestoning_count) + '\n')
                        self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].decorrelation_count = 0
                self.box_list[self.box].last_hit = 'upper'
                self.box_list[self.box].upper_non_milestoning_count = 0
                if self.box_list[self.box].last_hit == 'lower':
                    self.box_list[self.box].milestoning_count = 0
                if not self.reverse:
                    self.box_list[self.box].upper.transparent = self.criteria_met(self.box_list[self.box].upper)
                return True
        elif self.box_list[self.box].lower.hit(self.s, 'down'):
            self.bound_hit = 'lower'
            if self.progress_metric.reflect_back_to_path():
                return True
            if self.box_list[self.box].lower.transparent and not self.progress_metric.outside_path():
                self.box_list[self.box].lower.transparent = False
                self.box_list[self.box].close_box(path=self.progress_metric.path.s)
                self.box -= 1
                self.box_list[self.box].open_box()
                self.box_list[self.box].last_hit = 'upper'
                if self.box == 0:
                    self.reverse = False
                    self.completed_runs += 1
                return False
            else:
                self.bound_hit = 'lower'
                if decorrelated:
                    self.box_list[self.box].lower.hits += 1
                    self.box_list[self.box].lower_rates_file.write(str(self.box_list[self.box].lower_non_milestoning_count) + '\n')
                    if self.box_list[self.box].last_hit == 'upper':
                        self.box_list[self.box].lower_milestoning_rates_file.write(str(self.box_list[self.box].milestoning_count) + '\n')
                        self.box_list[self.box].milestoning_count = 0
                self.box_list[self.box].last_hit = 'lower'
                self.box_list[self.box].lower_non_milestoning_count = 0
                if self.box_list[self.box].last_hit == 'upper':
                    self.box_list[self.box].milestoning_count = 0
                return True
        else:
            return False

    def close(self, temp_dir):
        limits = open(temp_dir + '/box_limits', 'w')
        for box in self.box_list:
            limits.write('lowest box point = ' + str(min(box.data)) + ' highest box point = ' + str(max(box.data) + '\n'))
        for i, box2 in enumerate(self.box_list):
            temp_dir = self.dir + ("/box_" + str(i))
            if not os.path.isdir(temp_dir):
                os.makedirs(temp_dir)
            file = open(temp_dir + '/final_geometry.xyz', 'w')
            file.write(str(box2.data))


    def output(self):
        out = " box = " + str(self.box) + ' Lower Bound Hits = ' + str(self.box_list[self.box].lower.hits) + \
              ' Upper Bound Hits = ' + str(self.box_list[self.box].upper.hits) + ' Sampled points = ' + str(len(self.box_list[self.box].data))
        return out
