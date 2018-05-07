import numpy, os, re
from operator import attrgetter
# local imports
import prm_file


def load_data(dirname, pattern='velocity_([0-9]{4}).([0-9]{3}).dat', dim=2):
    """
    Keyword Arguments:
    dir     --
    pattern -- (default 'velocity_([0-9]{4}).([0-9]{3}).dat')

    Returns:
    {frame : data}

    where data is a list of dictionaries
    each one of the form:
    {'pos' : numpy.array, 'C' : numpy.array}
    """

    ro, subdirs, files = os.walk(dirname).next()
    frames_dict = {}
    for fi in files:
        match = re.match(pattern, fi)
        if match:
            frame = int(match.group(1))
            buffer = numpy.loadtxt(os.path.join(dirname, fi))
            if len(buffer.shape) == 1:
                local_data = [{'pos': buffer[:dim], 'data': buffer[dim:]}]
            else:
                local_data = [{'pos': row[:dim],
                               'data': row[dim:]}
                              for row in buffer]
            if frame in frames_dict:
                frames_dict[frame].extend(local_data)
            else:
                frames_dict[frame] = local_data
        # # TODO sort
        for key, v in frames_dict.items():
            frames_dict[key] = sorted(v, key=lambda x: list(x['pos']))

    return frames_dict


class VelocityFieldReader:
    def __init__(self, dirname='./'):
        self.params = prm_file.read()
        self.velocity_data = load_data(dirname)
        self.dt = self.params['dt']
        self.T = self.params['dt'] * self.params['steps']
        self.ts = numpy.arange(0, self.T, self.params['dt'])

        # collect evaluation points
        self.points = [elem['pos'] for elem in self.velocity_data[0]]

    def get_points(self):
        """
        Return available x's
        """
        return self.points

    def get_coefficients(self, index, t):
        """
        Keyword Arguments:
        self  --
        index -- point index
        t     -- time
        """
        if t > self.T:
            raise Exception('not contained in interval')
        tmp = t / self.dt
        u = int(numpy.ceil(tmp))
        l = int(numpy.floor(tmp))

        print('u = %d, l = %d' % (u, l))
        tl = float(l * self.dt)
        if l < u:
            wl = (1 - (t - tl) / self.dt)
            wu = (t-tl) / self.dt
            return (wl * self.velocity_data[l][index]['data'] +
                    wu * self.velocity_data[u][index]['data'])
        else:
            return self.velocity_data[u][index]['data']
