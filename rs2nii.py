# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   8 Oct., 2021
# Implementation of convert DICOM RT-Struct to mask.
# This file was adapted from [AutoContour](https://github.com/AutoContour)

import glob
import os

import numpy as np
import pydicom as dicom
import SimpleITK as sitk
from scipy import ndimage


def interplote(points):
    added = []
    for i in range(len(points)-1):
        dist = np.linalg.norm(np.array(points[i+1]) - np.array(points[i]))
        if dist > 1.4:
            pair = [points[i], points[i+1]]

            if np.abs(points[i][0]-points[i+1][0]) > np.abs(points[i][1]-points[i+1][1]):

                min_idx = np.argmin([points[i][0], points[i+1][0]])
                xx = np.linspace(start=pair[min_idx][0], stop=pair[1-min_idx]
                                 [0], num=pair[1-min_idx][0]-pair[min_idx][0]+2, dtype='int32')
                interp = np.interp(
                    xx, [pair[min_idx][0], pair[1-min_idx][0]], [pair[min_idx][1], pair[1-min_idx][1]])
                for dummy in zip(xx, interp):
                    added.append([int(dummy[0]), int(dummy[1])])

            else:
                min_idx = np.argmin([points[i][1], points[i+1][1]])
                yy = np.linspace(start=pair[min_idx][1], stop=pair[1-min_idx]
                                 [1], num=pair[1-min_idx][1]-pair[min_idx][1]+2, dtype='int32')
                interp = np.interp(
                    yy, [pair[min_idx][1], pair[1-min_idx][1]], [pair[min_idx][0], pair[1-min_idx][0]])
                for dummy in zip(interp, yy):
                    added.append([int(dummy[0]), int(dummy[1])])

    return [list(x) for x in set(tuple(x) for x in added+points)]


class RS2NII(object):
    def __init__(self, subject_folder_path, save_folder_path=None):

        self.temp_contours = None
        self.segmentation = None
        self.origin = {}
        self.pixel_spacing = {}
        modality = "ct"
        path = subject_folder_path + "/ct"
        self.save_folder_path = save_folder_path
        if self.save_folder_path is None:
            self.save_folder_path = subject_folder_path
        self.info_dict = {}
        imageReader = sitk.ImageSeriesReader()
        series_file_names = imageReader.GetGDCMSeriesFileNames(path)
        imageReader.SetFileNames(series_file_names)
        raw_image = imageReader.Execute()
        sitk.WriteImage(raw_image, self.save_folder_path + "/image.nii.gz")
        raw_image_size = raw_image.GetSize()

        slices = []
        dicom_series = sorted(glob.glob(os.path.join(
            subject_folder_path, modality, '*.DCM')))
        for s in dicom_series:
            slices.append(dicom.read_file(s, force=True))
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        self.origin = slices[0].ImagePositionPatient
        self.pixel_spacing = [slices[0].PixelSpacing[0], slices[0].PixelSpacing[1],
                              slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]]
        structure_set_file = glob.glob(os.path.join(
            subject_folder_path, modality, '*StrctrSets.dcm'))
        structure = dicom.read_file(structure_set_file[0], force=True)

        for item in structure.StructureSetROISequence:
            self.info_dict[item.ROINumber] = item.ROIName

        for roi in structure.ROIContourSequence:
            self.contour = np.zeros(raw_image_size)
            self.segmentation = np.zeros(raw_image_size)
            number = roi.ReferencedROINumber
            for plane_contour in roi.ContourSequence:
                contour_points = zip(*[iter(plane_contour.ContourData)]*3)
                contour_points_temp = [i for i in contour_points]
                z_voxel = int(
                    round((contour_points_temp[0][2] - self.origin[2]) / self.pixel_spacing[2]))
                test_aa = []
                for point in contour_points_temp:
                    x_voxel = int(
                        round((point[0] - self.origin[0]) / self.pixel_spacing[0]))
                    y_voxel = int(
                        round((point[1] - self.origin[1]) / self.pixel_spacing[1]))
                    test_aa.append([x_voxel, y_voxel])
                test_aa.append(test_aa[0])
                temp_contour = interplote(test_aa)
                temp_contour = np.array(temp_contour)
                self.contour[temp_contour[:, 1], temp_contour[:, 0],
                             z_voxel] = 1
                seg = ndimage.binary_fill_holes(self.contour[:, :, z_voxel])
                self.segmentation[:, :, z_voxel] = seg
                seg_itk = sitk.GetImageFromArray(
                    self.segmentation.transpose(2, 0, 1))
                seg_itk.CopyInformation(raw_image)
                sitk.WriteImage(seg_itk, self.save_folder_path +
                                "/{}.nii.gz".format(self.info_dict[number]))


path = "/data/DataSet_all/NPC/NANFang/NANFANG_NPC_DICOM/200285"
"""
folder structure
./NANFANG_NPC_DICOM/
    200285/
        t1/
        t1c/
        t2/
        ct/
        ......
"""
RS2NII(path)
