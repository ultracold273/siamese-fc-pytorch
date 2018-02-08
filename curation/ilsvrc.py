
from __future__ import print_function, division
import os
# import torch
# from skimage import io, transform
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
import xml.etree.ElementTree as ET

CLASS_IDS = [
    'n02691156', 'n02419796', 'n02131653', 'n02834778', 'n01503061', 
    'n02924116', 'n02958343', 'n02402425', 'n02084071', 'n02121808', 
    'n02503517', 'n02118333', 'n02510455', 'n02342885', 'n02374451', 
    'n02129165', 'n01674464', 'n02484322', 'n03790512', 'n02324045', 
    'n02509815', 'n02411705', 'n01726692', 'n02355227', 'n02129604', 
    'n04468005', 'n01662784', 'n04530566', 'n02062744', 'n02391049'
]

class XMLParser(object):
    ''' Parser for xml file in ILSVRC dataset '''
    @staticmethod
    def get_frame_size(element):
        assert element.tag == 'size' and len(element) == 2
        assert element[0].tag == 'width' and element[1].tag == 'height'
        width, height = float(element[0].text), float(element[1].text)
        return (width, height)
    
    @staticmethod
    def get_bndbox(element):
        assert element.tag == 'bndbox' and len(element) == 4
        assert element[0].tag == 'xmax' and element[1].tag == 'xmin'
        assert element[2].tag == 'ymax' and element[3].tag == 'ymin'
        xmax, xmin = float(element[0].text), float(element[1].text)
        ymax, ymin = float(element[2].text), float(element[3].text)
        bdbox_x = xmax - xmin + 1
        bdbox_y = ymax - ymin + 1
        return (xmin, ymin, bdbox_x, bdbox_y)

    @staticmethod
    def get_trackid(element):
        assert element.tag == 'trackid'
        return float(element.text)
    
    @staticmethod
    def get_class_idx(element):
        assert element.tag == 'name'
        assert element.text in CLASS_IDS
        return CLASS_IDS.index(element.text)

    @staticmethod
    def parse_object(element):
        assert element.tag == 'object'
        trackid = XMLParser.get_trackid(element.find('trackid'))
        class_idx = XMLParser.get_class_idx(element.find('name'))
        bndbox = XMLParser.get_bndbox(element.find('bndbox'))
        return (trackid, class_idx, bndbox)

class FrameAnnotation(object):
    ''' Annotation for a frame '''
    def __init__(self, nobjs, frame_size, obj_info=()):
        self.num_objs = nobjs
        self.frame_size = frame_size
        self.objs_info = obj_info
    
    def __repr__(self):
        return 'FrameAnnotation (nobjs={0:d}, frame_size={1:s}, obj_info={2:s})'.format(self.num_objs, repr(self.frame_size), repr(self.objs_info))

    @classmethod
    def from_annotation_file(cls, file_path):
        assert file_path.endswith('.xml')
        xml_root = ET.parse(file_path).getroot()
        frame_size = XMLParser.get_frame_size(xml_root.find('size'))
        objects_elem = xml_root.findall('object')
        if len(objects_elem) == 0:
            return cls(0, frame_size)
        else:
            objs_info = tuple(map(XMLParser.parse_object, objects_elem))
            return cls(len(objects_elem), frame_size, objs_info)

class VideoAnnotation(object):
    ''' Annotation for a video '''
    def __init__(self, frame_names=(), frame_annotations=()):
        self.nframes = len(frame_names)
        self.frame_annotations = frame_annotations
        self.frame_names = frame_names
        # self.valid_trackids = self.count()

    def __repr__(self):
        pass

    @classmethod
    def from_annotation_directory(cls, dir_path):
        frame_names = os.listdir(dir_path)
        frame_names.sort()
        frame_path_gen = lambda name: os.path.join(dir_path, name)
        frame_annos = tuple(map(FrameAnnotation.from_annotation_file, map(frame_path_gen, frame_names)))
        # Delete '.xml' suffix
        frame_names = tuple([name[:-4] for name in frame_names])
        return cls(frame_names=frame_names, frame_annotations=frame_annos)

class VideoDetectionDataset(object):
    ''' ILSVRC2015 Video frame dataset '''
    def __init__(self, root_dir, data_dir='Data/VID/train', anno_dir='Annotations/VID/train'):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, data_dir)
        self.anno_dir = os.path.join(root_dir, anno_dir)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

if __name__ == '__main__':
    frame_anno = FrameAnnotation.from_annotation_file('D:\\ILSVRC2015\\Annotations\\VID\\train\\ILSVRC2015_VID_train_0000\\ILSVRC2015_train_00000000\\000000.xml')
    s = eval(repr(frame_anno))
    print(type(s))
    print(s)
    