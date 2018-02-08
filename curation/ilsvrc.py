
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
        ''' Get the frame size from XML file '''
        assert element.tag == 'size' and len(element) == 2
        assert element[0].tag == 'width' and element[1].tag == 'height'
        width, height = float(element[0].text), float(element[1].text)
        return (width, height)
    
    @staticmethod
    def get_bndbox(element):
        ''' Get the bounding box from XML file '''
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
        ''' Get the track id from XML file '''
        assert element.tag == 'trackid'
        return int(element.text)
    
    @staticmethod
    def get_class_idx(element):
        ''' Get the class name from XML file and convert it to index '''
        assert element.tag == 'name'
        assert element.text in CLASS_IDS
        return CLASS_IDS.index(element.text)

    @staticmethod
    def parse_object(element):
        ''' Get the information for an object from XML file '''
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
        ''' Generate a FrameAnnotation from an XML file '''
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
        video_objects, trackids = self.count_objects()
        self.video_objects = video_objects
        self.trackid_dict = trackids
        self.trackid_cnt = {key: len(trackids[key]) for key in trackids}

    def __repr__(self):
        return 'VideoAnnotation (frame_names={0:s}, frame_annotations={1:s}'.format(repr(self.frame_names), repr(self.frame_annotations))

    def count_objects(self):
        ''' Count objects in the current video frames '''
        assert self.frame_annotations
        video_objects = 0
        trackids = {}
        for frame_idx, frame_annotation in enumerate(self.frame_annotations):
            video_objects += frame_annotation.num_objs
            for obj_idx, obj_info in enumerate(frame_annotation.objs_info):
                trackid = obj_info[0]
                if trackid not in trackids:
                    trackids[trackid] = [(frame_idx, obj_idx)]
                else:
                    trackids[trackid].append((frame_idx, obj_idx))
        return video_objects, trackids
    
    @classmethod
    def from_annotation_directory(cls, dir_path):
        ''' Generate Video annotation from a directory of XML files '''
        frame_names = os.listdir(dir_path)
        frame_names.sort()
        frame_path_gen = lambda name: os.path.join(dir_path, name)
        frame_annos = tuple(map(FrameAnnotation.from_annotation_file, map(frame_path_gen, frame_names)))
        # Delete '.xml' suffix
        frame_names = tuple([name[:-4] for name in frame_names])
        return cls(frame_names=frame_names, frame_annotations=frame_annos)

class VideoPathManager(object):
    ''' Manage the paths for videos '''
    def __init__(self, data_dir, rec_level=2):
        self.rec_level = 2
        self.path_tree = self.parse_data_dir(data_dir, rec_level)
        self.index_tree = self.build_index_path_tree()
        self.nvideos = len(self.index_tree)

    def __len__(self):
        return self.nvideos

    def __getitem__(self, idx):
        path_tuple_idx = self.index_tree[idx]
        path_tuple = self.path_tree
        accumulate_path_str = ''
        for i in path_tuple_idx:
            directory_name = path_tuple[i][0]
            path_tuple = path_tuple[i][1]
            accumulate_path_str = os.path.join(accumulate_path_str, directory_name)
        return accumulate_path_str

    def build_path_tree_index_rec(self, path_tuple, index_list, ret_list):
        assert len(path_tuple) > 0
        if len(path_tuple[0][1]) == 0:
            make_final_index = lambda idx: tuple(index_list[:] + [idx])
            final_index = tuple(map(make_final_index, range(len(path_tuple))))
            ret_list.extend(final_index)
        else:
            for idx, nxt_lvl_tuple in enumerate(path_tuple):
                self.build_path_tree_index_rec(nxt_lvl_tuple[1], index_list[:] + [idx], ret_list)

    def build_index_path_tree(self):
        path_tree_index = []
        self.build_path_tree_index_rec(self.path_tree, [], path_tree_index)
        return path_tree_index

    def parse_data_dir(self, cur_dir, rec_level=2):
        ''' Parse the data path '''
        assert rec_level >= 1
        if rec_level == 1:
            all_video_names = os.listdir(cur_dir)
            all_video_names.sort()
            return tuple([(v_name, ()) for v_name in all_video_names])
        else:
            dir_names = []
            next_dirs = os.listdir(cur_dir)
            next_dirs.sort()
            for next_dir in next_dirs:
                abs_path = os.path.join(cur_dir, next_dir)
                if not os.path.isdir(abs_path):
                    dir_names.append((next_dir, ()))
                else:
                    tpl = (next_dir, self.parse_data_dir(abs_path, rec_level=rec_level-1))
                    dir_names.append(tpl)
            return tuple(dir_names)

class VideoDetectionDataset(object):
    ''' ILSVRC2015 Video frame dataset '''
    def __init__(self, root_dir, data_dir='Data/VID/train', anno_dir='Annotations/VID/train'):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, data_dir)
        self.anno_dir = os.path.join(root_dir, anno_dir)
        # print(self.data_dir)
        self.video_path_man = VideoPathManager(self.anno_dir)
        self.annotation = self.get_all_annotation()

    def __len__(self):
        self.video_path_man.nvideos

    def __getitem__(self, idx):
        video_path = self.video_path_man[idx]
        full_video_path = os.path.join(self.data_dir, video_path)
        return full_video_path

    def get_all_annotation(self):
        ''' Get all annotation '''
        annotations = []
        for path in self.video_path_man:
            print('Processing annotations: {0:s}'.format(path))
            annotation_path = os.path.join(self.anno_dir, path)
            video_anno = VideoAnnotation.from_annotation_directory(annotation_path)
            annotations.append(video_anno)

    # def parse_paths_str(paths_str):
    #     ''' Get the index for each video path '''
    #     index = 0
    #     pass
    
if __name__ == '__main__':
    # frame_anno = FrameAnnotation.from_annotation_file('D:\\ILSVRC2015\\Annotations\\VID\\train\\ILSVRC2015_VID_train_0000\\ILSVRC2015_train_00000000\\000000.xml')
    # s = eval(repr(frame_anno))
    # print(type(s))
    # print(s)
    # video_anno = VideoAnnotation.from_annotation_directory('D:\\ILSVRC2015\\Annotations\\VID\\train\\ILSVRC2015_VID_train_0000\\ILSVRC2015_train_00000000')
    # print(video_anno.nframes)
    # print(video_anno.trackid_cnt)
    # print(video_anno.trackid_dict)
    # print(video_anno)
    # video_path = VideoPathManager('D:\\ILSVRC2015\\Annotations\\VID\\train')
    # print(len(video_path.path_tree))
    # s = len(video_path)
    # print(s)
    # for idx in range(s):
        # print(idx, video_path[idx])
    
    dataset = VideoDetectionDataset('D:\\ILSVRC2015')
    # for idx, full_path in enumerate(dataset):
        # print(idx, full_path)
    
