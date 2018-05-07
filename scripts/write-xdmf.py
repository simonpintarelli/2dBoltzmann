#!/usr/bin/env python2

#import xml.etree.ElementTree as ET
from lxml import etree as ET
import os,sys,re, glob
import h5py
from copy import copy

import yaml

time_offset = 0


basename = 'clean-'
#basename = ''
if __name__ == '__main__':
    dirname = sys.argv[1]

    mesh_path = os.path.join(dirname, basename+'mesh.hdf5')


    files = glob.glob(os.path.join(dirname,basename+'solution-*'))
    files = sorted(files)

    print 'Found %d files for processing' % len(files)

    # load config
    config = yaml.load(open(os.path.join(dirname, 'config.yaml'), 'r').read())
    dt = config['TimeStepping']['dt']

    # get number of nodes
    fh5 = h5py.File(os.path.join(dirname, basename+'mesh.hdf5'), 'r')
    n_nodes = fh5['nodes'].shape[0]
    n_cells = fh5['cells'].shape[0]

    # prepare xml output
    grid_temporal = ET.Element('Grid')
    grid_temporal.attrib['Name'] = 'CellTime'
    grid_temporal.attrib['GridType'] = 'Collection'
    grid_temporal.attrib['CollectionType'] = 'Temporal'

    # find tags in file
    fh5_data = h5py.File(files[0], 'r')
    keys = fh5_data.keys()

    geometry = ET.Element('Geometry', attrib={'GeometryType' : 'XY'})
    data_item = ET.SubElement(geometry, 'DataItem' , attrib={'Dimensions' : '%d %d' % (n_nodes, 2),
                                                             'NumberType' : 'Float',
                                                             'Precision' : '8',
                                                             'Format' : 'HDF'})
    data_item.text = mesh_path + ':/nodes'

    topology = ET.Element('Topology', attrib={'TopologyType' : 'Quadrilateral',
                                              'NumberOfElements' : str(n_cells)})
    data_item = ET.SubElement(topology, 'DataItem', attrib={'Dimensions' : '%d 4' % n_cells,
                                                            'NumberType' : 'UInt',
                                                            'Format' : 'HDF'})
    data_item.text= mesh_path + ':/cells'

    for fname in files:
        # iterate over timesteps
        match = re.match('.*solution-([0-9]*).*', fname)
        if match:
            frame = int(match.group(1))
        else:
            raise Exception('invalid filename')
        time_step = ET.Element('Grid', attrib={'Name' : 'mesh', 'GridType' : 'Uniform'})
        ET.SubElement(time_step, 'Time', attrib={'Value' : str(dt*frame + time_offset)})
        time_step.append(copy(geometry))
        time_step.append(copy(topology))

        for key in keys:
            # iterate over timestep data
            attribute = ET.Element('Attribute', attrib={'Name' : key, 'AttributeType' : 'Scalar', 'Center' : 'Node'})
            data_item = ET.SubElement(attribute, 'DataItem', attrib={'Dimensions' : '%d 1' % n_nodes,
                                                                     'NumberType' : 'Float',
                                                                     'Precision' : '8',
                                                                     'Format' : 'HDF'})
            data_item.text = fname + ':/' + key

            time_step.append(attribute)
        # add current time step to root
        grid_temporal.append(time_step)

    xdmf = ET.Element('Xdmf', attrib={'Version' : '2.0'})
    domain = ET.SubElement(xdmf, 'Domain')
    domain.append(grid_temporal)
    tree = ET.ElementTree(xdmf)

    out_string = ET.tostring(tree, pretty_print=True, xml_declaration=True, doctype='<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>', encoding='UTF-8')
    f = open('test.xdmf', 'w')
    f.write(out_string)
    f.close()
