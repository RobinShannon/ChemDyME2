from abc import abstractmethod
import numpy as np
from ase.atoms import Atoms

class ProgressMetric(object):
    """
    Abstract Base Class controlling the metric used to define bxd progress for a given point in collective variable
    space.The base class measures the bxd progress as the distance from the lower boundary to the a given point: The
    line and curve derived classes project a given point onto a line or curve respectively describing reaction from
    reactants to products.
    All derived classes must implement two methods:

    project_point_on_path: Function to transform any point in collective variable space into into progress toward the
                           defined bxd end point

    reflect_back_to_path: Function returns true if the bxd run has stayed to far from the path and False if otherwise.
                          In the base class this allways evaluates as False
    :param collective_variable: An instance of the collectiveVariable class which holds the particular distances
                                considered
    :param start_point: Either a list defining the collective variable at the starting point or an ASE atoms object
                        defining the starting geometry
    :param end_point: Either a list defining the collective variable at the target geometry or an ASE atoms object
                      defining the target geometry
    :param end_type: DEFAULT("distance")
                     "distance" An adaptive run is considered to have reached the end if the when the progress
                                metric reaches a particular value equal to that at the specified end_point
    """
    def __init__(self, collective_variable, start_point, end_point, end_type='distance'):
        self.collective_variable = collective_variable
        # Convert start geometry into the appropriate collective variable
        # If the start point is a list then assume this corresponds to the collective variable at the start point
        # Otherwise it is an ASE atoms type and the collective variable object is used to convert
        if isinstance(start_point, Atoms):
            self.start = collective_variable.get_s(start_point)
        else:
            self.start = np.asarray(start_point)
        # Same for end point
        if isinstance(end_point, Atoms):
            self.end = collective_variable.get_s(end_point)
        else:
            self.end = np.asarray(end_point)
        self.full_distance = 0
        self.end_type = end_type
        # Keeps track of whether the bxd trajectory is going in a forward or reverse direction
        self.bxd_reverse = False

    @staticmethod
    def get_dist_from_bound(s, bound):
        return np.vdot(s, bound.n) + bound.d

    # Get the current distance of bxd trajectory along defined path
    @abstractmethod
    def project_point_on_path(self, s):
        project = s - self.start
        return project

    # This method returns a bool signifying whether the current max distance from the path has been exceeded
    # In the simple case there is no path and this returns False

    def reflect_back_to_path(self):
        return False


class Curve(ProgressMetric):
    """
    Subclass of ProgressMetric for the instance where one wishes to project bxd onto a linearly interpolated guess
    path. The algorithm for determining progress is briefly as follows:
    1. Store closest path segment self.path_segment (starting at segment 0)
    2. Convert MD frame to collective variable (s)
    3. Considering  all path segments between self.path_segment - max_nodes_skiped and self.path_segment + max_nodes_
       skiped get the distance of the shortest line between the current value of s and each path segment.
    4. Get the cumulative distance along the path up to the new closest segment (d) This is stored in the path object.
    5. Get the scalar projection of the point s onto the line defining the closest path segment and add this to d
    6. Return d as the projected distance for the given MD frame
    7. Update self.path_segment with the new closest path segment
    :param collective_variable: CollectiveVarible object defining the collective variable space
    :param path: A path object defining the nodes of the guess path in collective variable space
    :param max_nodes_skiped: DEFAULT = 1
                             The ProgressMetric stores the closest path segment at a given point in the bxd
                             trajectory. The max_nodes_skipped parameter defines the number how muany adjacent path
                             segments should be considered when determining the closest path segment for the next
                             point in the bxd trajectory
    :param end_type: DEFAULT = "distances"
                     "distances" the end_point defines a distance along the path at which the bxd run is considered
                                 complete
                     "boxes" end_point defined number of boxes for the adaptive run. If the adaptive bxd run goes
                             in both directions then more boxes may be added in the reverse direction. See
                             BXDConstraint class
    """

    def __init__(self, collective_variable, path, max_nodes_skiped=1, end_type='distance'):
        super().__init__(collective_variable, path.s[0], path.s[-1], end_type)
        self.path = path
        self.max_nodes_skipped = max_nodes_skiped
        # current closest linear segment
        self.path_segment = 0
        self.proj = 0
        # get the current distance from the path
        self.distance_from_path = 0
        self.old_distance_from_path = 0
        self.percentage_along_segment = 0
        self.countdown = 0
        if end_type =='distance':
            self.full_distance = self.path.total_distance[-1]

    @staticmethod
    def distance_to_segment(s, segment_end, segment_start):
        """
        Get the distance of the shortest line between s and a given path segment, also return the scalar projection of s
        onto the line segment_end - segment start
        :param s: Current colective variable
        :param segment_end: Starting node of path segment
        :param segment_start: End node of path segment
        :return: Distance, scalar projection, scalar projection as fraction of total distance.
        """
        # Get vector from segStart to segEnd
        segment = segment_end - segment_start
        # Scalar projection of S onto segment
        scalar_projection = np.vdot((s - segment_start), segment) / np.linalg.norm(segment)
        # Length of segment
        path_segment_length = np.linalg.norm(segment)
        # Vector projection of S onto segment
        vector_projection = segment_start + (scalar_projection * (segment / path_segment_length))
        # Length of this vector projection gives distance from line
        # If the vector projection is past the start or end point of the segment then use distance to the distance to
        # segStart or segEnd respectively
        if scalar_projection < 0:
            dist = np.linalg.norm(s - segment_start)
        elif scalar_projection > path_segment_length:
            dist = np.linalg.norm(s - segment_end)
        else:
            dist = np.linalg.norm(s - vector_projection)
        return dist, scalar_projection, scalar_projection/path_segment_length

    @staticmethod
    def vector_to_segment(s, segment_end, segment_start):
        """
        Get a unit vector from s to a given path segment. Used to define the norm for bxd reflection back towards the
        path
        :param s: Current colective variable
        :param segment_end: Starting node of path segment
        :param segment_start: End node of path segment
        :return:
        """
        # Get vector from segStart to segEnd
        segment = segment_end - segment_start
        # Length of segment
        path_segment_length = np.linalg.norm(segment)
        # Scalar projection of s onto segment
        scalar_projection = np.vdot((s - segment_start), (segment / path_segment_length))
        # Vector projection of S onto segment
        vector_projection = segment_start + (scalar_projection * (segment / path_segment_length))
        norm = (s - vector_projection)/np.linalg.norm(s - vector_projection)
        return norm

    def project_point_on_path(self, s):
        """
        Get projected distance along the path for a given geometry with collective variable s
        :param s: collective variable value for current MD frame
        :return: distance along path.
        """
        # Set up tracking variables to log the closest segment and the distance to it
        minim = float("inf")
        proj = 0
        closest_segment = 0
        # Use self.max_nodes_skipped to set up the start and end points for looping over path segments.
        start = self.path_segment - self.max_nodes_skipped
        end = self.path_segment + self.max_nodes_skipped
        start = max(start, 0)
        end = min(end, len(self.path.s) - 1)
        # Now loop over all segments considered and get the distance from S to that segment and the projected distance
        # of S along that segment
        percentage = 0
        for i in range(start, end):
            dist, projection, percent = self.distance_to_segment(s, self.path.s[i+1], self.path.s[i])
            if dist < minim:
                percentage = percent
                closest_segment = i
                minim = dist
                proj = projection
        # Update the current distance from path and path segment
        self.old_distance_from_path = self.distance_from_path
        self.distance_from_path = minim
        self.percentage_along_segment = percentage
        self.path_segment = closest_segment
        # To get the total distance along the path add the total distance along all segments seg < minPoint
        proj += self.path.total_distance[closest_segment]
        self.proj = proj
        return proj

    def get_node(self, s):
        """
        Get closes path segment to a given value of s, ignoring  self.max_nodes_skipped and self.path_segment
        :param s: Collective variable value
        :return: Closest node
        """
        # Set up tracking variables to log the closest segment and the distance to it
        minim = float("inf")
        closest_segment = 0
        # Use self.max_nodes_skipped to track set up the start and end points for looping over path segments.
        start = 0
        end = len(self.path.s) - 1
        # Now loop over all segments considered and get the distance from S to that segment and the projected distance
        # of S along that segment
        for i in range(start, end):
            dist, projection, percent = self.distance_to_segment(s, self.path.s[i+1], self.path.s[i])
            if dist < minim:
                closest_segment = i
                minim = dist
        return closest_segment

    def reflect_back_to_path(self):
        """
        Determine whether the current point is outside the defined max_distance_from_path.
        If we are outside the path, but moving closer to the path then return False instead of True
        :return: Boolean
        """
        if self.countdown >= 50:
            self.path.max_distance[self.path_segment] = self.distance_from_path + (0.5 * self.distance_from_path)
            print(str(self.distance_from_path))
            try:
                self.path.max_distance[self.path_segment+1] = self.distance_from_path + (0.5 * self.distance_from_path)
                self.path.max_distance[self.path_segment-1] = self.distance_from_path + (0.5 * self.distance_from_path)
            except:
                pass
            print("expanding path distance")
        if self.distance_from_path > self.path_bound_distance_at_point():
            self.old_distance_from_path = self.distance_from_path
            self.countdown +=1
            return True
        else:
            self.old_distance_from_path = self.distance_from_path
            self.countdown = 0
            return False



    def outside_path(self):
        """
        Determine whether the current point is outside the defined max_distance_from_path.
        If we are outside the path, but moving closer to the path then return False instead of True
        :return: Boolean
        """
        # Check whether the current distance to the path is outside of the maximum allowable
        if self.distance_from_path > self.path_bound_distance_at_point():
            return True
        else:
            return False


    def path_bound_distance_at_point(self):
        """
        Get max distance from path at the current point. This is only neccessay when different max_distances_from_path
        have been defined for different segments.
        :return:
        """
        try:
            gradient = self.path.max_distance[self.path_segment + 1] \
                       - self.path.max_distance[self.path_segment]
            return self.path.max_distance[self.path_segment] + gradient * self.percentage_along_segment
        except IndexError:
            return self.path.max_distance[self.path_segment]

    def get_path_delta(self, mol):
        """
        Gets the delta_phi for a boundary hit on the path_boundary and sends this to the BXDConstraint object.
        Usually this would be dealt with in the BXDConstraint object but using bxd to reflect back towards the path is
        an exception
        :param mol: ASE atoms type
        :return: Array with del_phi
        """
        s = self.collective_variable.get_s(mol)
        seg_start = self.path.s[self.path_segment]
        seg_end = self.path.s[self.path_segment+1]
        norm = self.vector_to_segment(s, seg_end, seg_start)
        return self.collective_variable.get_delta(mol, norm)

    def get_norm_to_path(self, s):
        """
        Get vector norm to current path segment. BXDConstraint will use this to perform the bxd inversion
        :param s: Collective variable value
        :return: vector norm
        """
        seg = self.get_node(s)
        seg_start = self.path.s[seg]
        seg_end = self.path.s[seg+1]
        n = (seg_end - seg_start) / np.linalg.norm(seg_end - seg_start)
        return n


class Line(ProgressMetric):
    """
    Subclass of "Projection" where the path is a line connecting start and end geometries.
    :param collective_variable: Collective variable object
    :param start_mol: ASE atoms object defining starting geometry
    :param end_point: Target point in bxd trajectory, either an ASE atoms object or a list corresponding to the
                      collective variable at the end point
    :param end_type: DEFAULT = "distances"
                     "distances" the end_point defines a distance along the path at which the bxd run is considered
                                 complete
                     "boxes" end_point defined number of boxes for the adaptive run. If the adaptive bxd run goes
                             in both directions then more boxes may be added in the reverse direction. See
                             BXDConstraint class
    :param max_distance_from_path: DEFAULT = "inf"
    """
    def __init__(self, collective_variable, start_mol, end_point, end_type='distance',
                 max_distance_from_path=float("inf")):
        super().__init__(collective_variable, start_mol, end_point, end_type)
        self.max_distance_from_path = max_distance_from_path
        self.line = self.end - self.start
        self.distance_from_path = 0
        self.full_distance = np.linalg.norm(self.end - self.start)

    def project_point_on_path(self, s):
        """
        Get projected distance along the path for a given geometry with collective variable s
        :param s: collective variable value for current MD frame
        :return: distance along path.
        """
        # get line from the start coordinate to current S
        baseline = s - self.start
        # project this line onto the path line
        project = np.vdot(baseline, self.line) / np.linalg.norm(self.line)
        # Also get vector projection
        vector_projection = (np.vdot(baseline, self.line) / np.vdot(self.line, self.line)) * self.line
        # Length of this vector projection gives distance from line
        self.distance_from_path = np.linalg.norm(vector_projection)
        return project

    def reflect_back_to_path(self):
        """
        Determine whether the current point is outside the defined max_distance_from_path.
        If we are outside the path, but moving closer to the path then return False instead of True
        :return: Boolean
        """
        if self.distance_from_path > self.max_distance_from_path:
            return True
        else:
            return False
