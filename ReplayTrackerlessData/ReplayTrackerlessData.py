import os
from pyexpat import model
import unittest
# from matplotlib.pyplot import get
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from vtk.util import numpy_support
import numpy as np
import torch
import math
from sys import platform
from PIL import Image
from Resources import layers
import re
import json

class ReplayTrackerlessData(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Bakse/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Replay Trackerless Data"
    self.parent.categories = ["Navigation"]
    self.parent.dependencies = []
    self.parent.contributors = ["Franklin King, Megha Kalia"]
    self.parent.helpText = """
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
"""
    # Set module icon from Resources/Icons/<ModuleName>.png
    moduleDir = os.path.dirname(self.parent.path)
    for iconExtension in ['.svg', '.png']:
      iconPath = os.path.join(moduleDir, 'Resources/Icons', self.__class__.__name__ + iconExtension)
      if os.path.isfile(iconPath):
        parent.icon = qt.QIcon(iconPath)
        break

class ReplayTrackerlessDataWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    ScriptedLoadableModuleWidget.__init__(self, parent)
    self.stepCount = 1
    self.stepTimer = qt.QTimer()
    self.stepTimer.timeout.connect(self.onStepImage)
    self.surfaceRegistrationModule = slicer.modules.surfaceregistration.widgetRepresentation().self()

  def cleanup(self):
    self.stepTimer.stop()
    self.onResetStepCount()

  def onReload(self,moduleName="ReplayTrackerlessData"):
    self.stepTimer.stop()
    self.onResetStepCount()
    globals()[moduleName] = slicer.util.reloadScriptedModule(moduleName)    

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # IO collapsible button
    self.IOCollapsibleButton = ctk.ctkCollapsibleButton()
    self.IOCollapsibleButton.text = "I/O"
    self.IOCollapsibleButton.collapsed = False
    self.layout.addWidget(self.IOCollapsibleButton)
    IOLayout = qt.QFormLayout(self.IOCollapsibleButton)

    self.imageSelector = slicer.qMRMLNodeComboBox()
    self.imageSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
    self.imageSelector.selectNodeUponCreation = False
    self.imageSelector.noneEnabled = True
    self.imageSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow('Image: ', self.imageSelector)

    self.inputTransformSelector = slicer.qMRMLNodeComboBox()
    self.inputTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.inputTransformSelector.selectNodeUponCreation = False
    self.inputTransformSelector.noneEnabled = False
    self.inputTransformSelector.addEnabled = True
    self.inputTransformSelector.removeEnabled = True
    self.inputTransformSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow("Camera Transform: ", self.inputTransformSelector)

    self.baseModelSelector = slicer.qMRMLNodeComboBox()
    self.baseModelSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.baseModelSelector.selectNodeUponCreation = False
    self.baseModelSelector.addEnabled = False
    self.baseModelSelector.removeEnabled = True
    self.baseModelSelector.renameEnabled = True
    self.baseModelSelector.noneEnabled = False
    self.baseModelSelector.showHidden = False
    self.baseModelSelector.showChildNodeTypes = False
    self.baseModelSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow("Base Model: ", self.baseModelSelector)

    self.icpTransformSelector = slicer.qMRMLNodeComboBox()
    self.icpTransformSelector.nodeTypes = ["vtkMRMLLinearTransformNode"]
    self.icpTransformSelector.selectNodeUponCreation = False
    self.icpTransformSelector.noneEnabled = False
    self.icpTransformSelector.addEnabled = True
    self.icpTransformSelector.removeEnabled = True
    self.icpTransformSelector.setMRMLScene(slicer.mrmlScene)
    IOLayout.addRow("ICP Transform: ", self.icpTransformSelector)

    # Step mode Recorded Data collapsible button
    self.stepModeCollapsibleButton = ctk.ctkCollapsibleButton()
    self.stepModeCollapsibleButton.text = "Step Mode - Recorded Data"
    self.stepModeCollapsibleButton.collapsed = False
    self.layout.addWidget(self.stepModeCollapsibleButton)
    stepModeLayout = qt.QFormLayout(self.stepModeCollapsibleButton)

    self.gtPathBox = qt.QLineEdit("D:/MeghaData/Data/upsample_4gauss_mask_0.0001/data")
    self.gtPathBox.setReadOnly(True)
    self.gtPathButton = qt.QPushButton("...")
    self.gtPathButton.clicked.connect(self.select_directory_gt)
    gtPathBoxLayout = qt.QHBoxLayout()
    gtPathBoxLayout.addWidget(self.gtPathBox)
    gtPathBoxLayout.addWidget(self.gtPathButton)
    stepModeLayout.addRow("Data: ", gtPathBoxLayout)

    self.methodSelectionComboBox = qt.QComboBox()
    self.methodSelectionComboBox.addItem('ICP')
    self.methodSelectionComboBox.addItem('ICP with Pose')
    self.methodSelectionComboBox.addItem('Ground Truth')
    self.methodSelectionComboBox.addItem('None')
    self.methodSelectionComboBox.setCurrentIndex(0)
    stepModeLayout.addRow('Registration Method:', self.methodSelectionComboBox)

    # self.gtCheckBox = qt.QCheckBox("Use Ground Truth Data")
    # self.gtCheckBox.setChecked(False)
    # stepModeLayout.addRow(self.gtCheckBox)

    # self.icpCheckBox = qt.QCheckBox("Use ICP")
    # self.icpCheckBox.setChecked(False)
    # stepModeLayout.addRow(self.icpCheckBox)

    self.loadPredButton = qt.QPushButton("Load Predictions")
    stepModeLayout.addRow(self.loadPredButton)
    self.loadPredButton.connect('clicked()', self.onLoadPred)    

    self.stepButton = qt.QPushButton("Step Image")
    stepModeLayout.addRow(self.stepButton)
    self.stepButton.connect('clicked()', self.onStepImage)

    self.resetStepButton = qt.QPushButton("Reset Step Count")
    stepModeLayout.addRow(self.resetStepButton)
    self.resetStepButton.connect('clicked()', self.onResetStepCount)

    self.stepSkipBox = qt.QSpinBox()
    self.stepSkipBox.setSingleStep(1)
    self.stepSkipBox.setMaximum(100)
    self.stepSkipBox.setMinimum(1)
    self.stepSkipBox.value = 3
    stepModeLayout.addRow("Step skip: ", self.stepSkipBox)

    self.stepLabel = qt.QLabel("1")
    stepModeLayout.addRow("Image: ", self.stepLabel)

    self.stepTimerButton = qt.QPushButton("Step Timer")
    self.stepTimerButton.setCheckable(True)
    stepModeLayout.addRow(self.stepTimerButton)
    self.stepTimerButton.connect('clicked()', self.onStepTimer)

    self.stepFPSBox = qt.QSpinBox()
    self.stepFPSBox.setSingleStep(1)
    self.stepFPSBox.setMaximum(144)
    self.stepFPSBox.setMinimum(1)
    self.stepFPSBox.setSuffix(" FPS")
    self.stepFPSBox.value = 10
    stepModeLayout.addRow(self.stepFPSBox)

    self.scaleSliderWidget = ctk.ctkSliderWidget()
    self.scaleSliderWidget.setDecimals(2)
    self.scaleSliderWidget.minimum = 0.00
    self.scaleSliderWidget.maximum = 10000.00
    self.scaleSliderWidget.singleStep = 0.01
    self.scaleSliderWidget.value = 200.00
    stepModeLayout.addRow("Scale Factor:", self.scaleSliderWidget)

    # # Model Mode collapsible button
    # self.modelModeCollapsibleButton = ctk.ctkCollapsibleButton()
    # self.modelModeCollapsibleButton.text = "Model mode - Live data"
    # self.modelModeCollapsibleButton.collapsed = False
    # self.layout.addWidget(self.modelModeCollapsibleButton)
    # stepModeLayout = qt.QFormLayout(self.modelModeCollapsibleButton)

    # Depth map point cloud collapsible button
    self.depthMapModelCollapsibleButton = ctk.ctkCollapsibleButton()
    self.depthMapModelCollapsibleButton.text = "Depth Map Model"
    self.depthMapModelCollapsibleButton.collapsed = False
    self.layout.addWidget(self.depthMapModelCollapsibleButton)
    depthMapModeLayout = qt.QFormLayout(self.depthMapModelCollapsibleButton)    

    self.pointCloudSelector = slicer.qMRMLNodeComboBox()
    self.pointCloudSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.pointCloudSelector.selectNodeUponCreation = True
    self.pointCloudSelector.addEnabled = True
    self.pointCloudSelector.removeEnabled = True
    self.pointCloudSelector.renameEnabled = True
    self.pointCloudSelector.noneEnabled = False
    self.pointCloudSelector.showHidden = False
    self.pointCloudSelector.showChildNodeTypes = False
    self.pointCloudSelector.setMRMLScene(slicer.mrmlScene)
    depthMapModeLayout.addRow("Point Cloud Model: ", self.pointCloudSelector)

    self.centerlineSelector = slicer.qMRMLNodeComboBox()
    self.centerlineSelector.nodeTypes = ( ("vtkMRMLModelNode"), "" )
    self.centerlineSelector.selectNodeUponCreation = True
    self.centerlineSelector.addEnabled = True
    self.centerlineSelector.removeEnabled = True
    self.centerlineSelector.renameEnabled = True
    self.centerlineSelector.noneEnabled = False
    self.centerlineSelector.showHidden = False
    self.centerlineSelector.showChildNodeTypes = False
    self.centerlineSelector.setMRMLScene(slicer.mrmlScene)
    depthMapModeLayout.addRow("Center Line Model: ", self.centerlineSelector)

    self.maskSelectionComboBox = qt.QComboBox()
    self.maskSelectionComboBox.addItems(['Mask Image', 'Crop'])
    self.maskSelectionComboBox.setCurrentIndex(0)
    self.maskSelectionComboBox.currentIndexChanged.connect(self.onMaskSelection)
    depthMapModeLayout.addRow('Mask method:', self.maskSelectionComboBox)

    self.cornerCutBox = qt.QSpinBox()
    self.cornerCutBox.setSingleStep(1)
    self.cornerCutBox.setMaximum(100)
    self.cornerCutBox.setMinimum(0)
    self.cornerCutBox.setSuffix(" px")
    self.cornerCutBox.value = 40
    self.cornerCutBox.setEnabled(False)
    depthMapModeLayout.addRow("Corner size to cut:", self.cornerCutBox)

    self.borderCutBox = qt.QSpinBox()
    self.borderCutBox.setSingleStep(1)
    self.borderCutBox.setMaximum(100)
    self.borderCutBox.setMinimum(0)
    self.borderCutBox.setSuffix(" px")
    self.borderCutBox.value = 40
    self.borderCutBox.setEnabled(False)
    depthMapModeLayout.addRow("Border size to cut:", self.borderCutBox)    

    self.focalLengthBox = ctk.ctkDoubleSpinBox()
    self.focalLengthBox.maximum = 100000.0
    self.focalLengthBox.minimum = 0.0
    self.focalLengthBox.setDecimals(6)
    self.focalLengthBox.setValue(128.7)
    depthMapModeLayout.addRow("Focal Length:", self.focalLengthBox)

    self.imageScaleBox = ctk.ctkDoubleSpinBox()
    self.imageScaleBox.setValue(1.0)
    self.imageScaleBox.maximum = 100000.0
    self.imageScaleBox.minimum = 0.0
    depthMapModeLayout.addRow("Image Scale: ", self.imageScaleBox)

    self.depthScaleBox = ctk.ctkDoubleSpinBox()
    self.depthScaleBox.setValue(1.0)
    self.depthScaleBox.maximum = 100000.0
    self.depthScaleBox.minimum = 0.0
    depthMapModeLayout.addRow("Depth Scale: ", self.depthScaleBox)

    self.thresholdBox = qt.QSpinBox()
    self.thresholdBox.setSingleStep(1)
    self.thresholdBox.setMaximum(100)
    self.thresholdBox.setMinimum(0)
    self.thresholdBox.setSuffix("%")
    self.thresholdBox.value = 60
    depthMapModeLayout.addRow("Threshold:", self.thresholdBox)

    # Add vertical spacer
    self.layout.addStretch(1)

  def select_directory_gt(self):
    directory = qt.QFileDialog.getExistingDirectory(self.parent, "Select Directory")
    if directory:
      self.gtPathBox.setText(directory)

  def onStepTimer(self):
    if self.stepTimerButton.isChecked():
      self.stepTimer.start(int(1000/int(self.stepFPSBox.value)))
      self.stepFPSBox.enabled = False
    else:
      self.stepTimer.stop()
      self.stepFPSBox.enabled = True

  def onResetStepCount(self):
    self.stepCount = 1
    self.stepLabel.setText('1')

    layoutManager = slicer.app.layoutManager()
    red = layoutManager.sliceWidget('Red')
    redLogic = red.sliceLogic()
    redLogic.SetSliceOffset(0)

    green = layoutManager.sliceWidget('Green')
    greenLogic = green.sliceLogic()
    greenLogic.SetSliceOffset(0)
    
    resultMatrix = vtk.vtkMatrix4x4()
    if self.inputTransformSelector.currentNode():
      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(resultMatrix)
  
  def onMaskSelection(self, index):
    if index == 1:
      self.cornerCutBox.setEnabled(True)
      self.borderCutBox.setEnabled(True)
    else:
      self.cornerCutBox.setEnabled(False)
      self.borderCutBox.setEnabled(False)

  def onLoadPred(self):
    self.translations_pred = np.load(f'{self.gtPathBox.text}/translation.npz')
    self.axis_angle_pred = np.load(f'{self.gtPathBox.text}/axisangle.npz')
    self.pose_pred = np.load(f'{self.gtPathBox.text}/pose_prediction.npz')

  def createPointCloud(self, stepCountString, maskImage):
    suffix = "_depth.npy"
    depthMapFilename = ""
    for filename in os.listdir(self.gtPathBox.text):
      if filename.endswith(suffix):
        stripped_filename = re.sub(suffix + '$', '', filename).lstrip('0').split('.')[0]
        if stripped_filename == stepCountString:
          depthMapFilename = filename
    depthMap = np.load(f'{self.gtPathBox.text}/{depthMapFilename}')[0][0]

    if self.maskSelectionComboBox.currentIndex == 1:
      prefix = "mask_"
      suffix = ".jpeg"
      maskFilename = ""
      for filename in os.listdir(self.gtPathBox.text):
        if filename.startswith(prefix) and filename.endswith(suffix):
          base_name = filename.split(".")[0]
          number_str = base_name.split("_")[1]
          if number_str == stepCountString:
            maskFilename = filename
      maskImage = np.load(f'{self.gtPathBox.text}/{maskFilename}')[0][0]
    if self.imageSelector.currentNode():
      self.depthMapToPointCloud(depthMap, slicer.util.arrayFromVolume(self.imageSelector.currentNode())[self.stepCount], maskImage)
    else:
      self.depthMapToPointCloud(depthMap, None, maskImage)

  def adjustSliceOffset(self):
    layoutManager = slicer.app.layoutManager()

    # Depth Map
    red = layoutManager.sliceWidget('Red')
    redLogic = red.sliceLogic()
    redLogic.SetSliceOffset(self.stepCount//self.stepSkipBox.value - 1)

    # RGB
    green = layoutManager.sliceWidget('Green')
    greenLogic = green.sliceLogic()
    greenLogic.SetSliceOffset(self.stepCount - 1)

  def onStepImage(self):
    stepCountString = str(self.stepCount)
    maskImage = None
    # ------------------------------ Ground Truth ------------------------------
    if self.methodComboBox.currentText == "Ground Truth":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start Load Ground Truth:

      prefix = "frame_data"
      frameDataFilename = ""
      for filename in os.listdir(self.gtPathBox.text):
        if filename.startswith(prefix):
          stripped_filename = re.sub('^' + prefix, '', filename).lstrip('0').split('.')[0]
          if stripped_filename == stepCountString:
            frameDataFilename = filename

      # Grab pose data and use it
      f = open(f'{self.gtPathBox.text}/{frameDataFilename}')
      data = json.load(f)
      cameraPose = data['camera-pose']
      
      matrix = vtk.vtkMatrix4x4()
      matrix.SetElement(0, 0, cameraPose[0][0]); matrix.SetElement(0, 1, cameraPose[0][1]); matrix.SetElement(0, 2, cameraPose[0][2]); matrix.SetElement(0, 3, cameraPose[0][3])
      matrix.SetElement(1, 0, cameraPose[1][0]); matrix.SetElement(1, 1, cameraPose[1][1]); matrix.SetElement(1, 2, cameraPose[1][2]); matrix.SetElement(1, 3, cameraPose[1][3])
      matrix.SetElement(2, 0, cameraPose[2][0]); matrix.SetElement(2, 1, cameraPose[2][1]); matrix.SetElement(2, 2, cameraPose[2][2]); matrix.SetElement(2, 3, cameraPose[2][3])
      matrix.SetElement(3, 0, cameraPose[3][0]); matrix.SetElement(3, 1, cameraPose[3][1]); matrix.SetElement(3, 2, cameraPose[3][2]); matrix.SetElement(3, 3, cameraPose[3][3])
      self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(matrix)
    # ------------------------------ None ------------------------------
    elif self.methodComboBox.currentText == "None":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()
    # ------------------------------ ICP ------------------------------
    elif self.methodComboBox.currentText == "ICP":
      if self.pointCloudSelector.currentNode():
        self.createPointCloud(stepCountString, maskImage)
          
      # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start ICP:
    # ------------------------------ ICP with Pose ------------------------------
    elif self.methodComboBox.currentText == "ICP with Pose":
      pass
      # if self.pointCloudSelector.currentNode():
      #   self.createPointCloud(stepCountString, maskImage)
          
      # # Display image by moving slice offset
      self.adjustSliceOffset()

      # Start ICP:
    # ------------------------------ Model ------------------------------
    elif self.methodComboBox.currentText == "Model":
      pass
      # if self.pointCloudSelector.currentNode():
      #   self.createPointCloud(stepCountString, maskImage)

      # # Display image by moving slice offset
      # self.adjustSliceOffset()

      # scale = self.scaleSliderWidget.value

      # previousMatrix = vtk.vtkMatrix4x4()
      # self.inputTransformSelector.currentNode().GetMatrixTransformToParent(previousMatrix)

      # previousRotationMatrix = vtk.vtkMatrix4x4()
      # for i in range(3):
      #   for j in range(3):
      #     previousRotationMatrix.SetElement(i, j, previousMatrix.GetElement(i, j))

      # pred_poses = []
      # pred_poses.append(layers.transformation_from_parameters(torch.from_numpy(self.axis_angle_pred['arr_0'][self.stepCount-1:self.stepCount, 0]), torch.from_numpy(self.translations_pred['arr_0'][self.stepCount-1:self.stepCount, 0]) * scale).cpu().numpy())
      # pred_poses = np.concatenate(pred_poses)
      # dump_our = np.array(self.dump(self.vtk_to_numpy_matrix(previousMatrix), pred_poses))

      # self.inputTransformSelector.currentNode().SetAndObserveMatrixTransformToParent(self.numpy_to_vtk_matrix(dump_our[1]))

    self.stepCount = self.stepCount + self.stepSkipBox.value
    self.stepLabel.setText(stepCountString)

  def vtk_to_numpy_matrix(self, vtk_matrix):
    numpy_matrix = np.zeros((4, 4))
    for i in range(4):
      for j in range(4):
        numpy_matrix[i, j] = vtk_matrix.GetElement(i, j)
    return numpy_matrix
  
  def numpy_to_vtk_matrix(self, numpy_matrix):
    # Ensure the numpy matrix is 4x4
    assert numpy_matrix.shape == (4, 4), "The input numpy matrix must be 4x4"

    # Create an empty vtkMatrix4x4
    vtk_matrix = vtk.vtkMatrix4x4()

    # Fill the vtkMatrix4x4 with values from the numpy matrix
    for i in range(4):
      for j in range(4):
        vtk_matrix.SetElement(i, j, numpy_matrix[i, j])

    return vtk_matrix

  def dump(self, cam_to_world, source_to_target_transformations):
    Ms = []
    #cam_to_world = np.eye(4)
    Ms.append(cam_to_world)
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        Ms.append(cam_to_world)
    return Ms

  def MatrixFromRotation(self, angle, x, y, z, matrix):
    matrix.Identity()

    if (angle == 0.0 or (x == 0.0 and y == 0.0 and z == 0.0)):
      return

    # convert to radians
    angle = vtk.vtkMath.RadiansFromDegrees(angle)

    # make a normalized quaternion
    w = math.cos(0.5 * angle)
    f = math.sin(0.5 * angle) / math.sqrt(x * x + y * y + z * z)
    x = x * f
    y = y * f
    z = z * f

    # convert the quaternion to a matrix
    ww = w * w
    wx = w * x
    wy = w * y
    wz = w * z

    xx = x * x
    yy = y * y
    zz = z * z

    xy = x * y
    xz = x * z
    yz = y * z

    s = ww - xx - yy - zz

    matrix.SetElement(0,0, xx * 2 + s)
    matrix.SetElement(1,0, (xy + wz) * 2)
    matrix.SetElement(2,0, (xz - wy) * 2)

    matrix.SetElement(0,1, (xy - wz) * 2)
    matrix.SetElement(1,1, yy * 2 + s)
    matrix.SetElement(2,1, (yz + wx) * 2)

    matrix.SetElement(0,2, (xz + wy) * 2)
    matrix.SetElement(1,2, (yz - wx) * 2)
    matrix.SetElement(2,2, zz * 2 + s)
  
  def registerToBaseModel(self):
    # ICP Registration
    self.surfaceRegistration.inputFixedModelSelector.currentNodeID = self.baseModelSelector.currentNodeID
    
    self.surfaceRegistration.inputMovingModelSelector.currentNodeID = self.pointCloudSelector.currentNodeID
    self.surfaceRegistration.outputTransformSelector.currentNodeID = self.transientTransformSelector.currentNodeID
    #self.surfaceRegistration.landmarkTransformTypeButtonsSimilarity.checked = True
    self.surfaceRegistration.numberOfIterations.setValue(1000)
    self.surfaceRegistration.numberOfLandmarks.setValue(250)
    self.surfaceRegistration.onComputeButton()

    # # Add transient transform to total transform
    # transientMatrix = vtk.vtkMatrix4x4()
    # matrix = vtk.vtkMatrix4x4()
    # resultMatrix = vtk.vtkMatrix4x4()
    # self.transformSelector.currentNode().GetMatrixTransformToParent(matrix)
    # self.transientTransformSelector.currentNode().GetMatrixTransformToParent(transientMatrix)
    # vtk.vtkMatrix4x4.Multiply4x4(transientMatrix, matrix, resultMatrix)
    # self.transformSelector.currentNode().SetMatrixTransformToParent(resultMatrix)

  def depthMapToPointCloud(self, depthImage, rgbImage, maskImage = None):
    height = len(depthImage)
    width = len(depthImage[0])

    if maskImage:
      # Mask
      depthImage[~maskImage] = np.nan
    else:
      # Cut corners and border
      cornerSize = self.cornerCutBox.value
      borderSize = self.borderCutBox.value
      depthImage[:cornerSize, :cornerSize] = np.nan
      depthImage[:cornerSize, -cornerSize:] = np.nan
      depthImage[-cornerSize:, :cornerSize] = np.nan
      depthImage[-cornerSize:, -cornerSize:] = np.nan
      depthImage[:borderSize, :] = np.nan
      depthImage[-borderSize:, :] = np.nan
      depthImage[:, :borderSize] = np.nan
      depthImage[:, -borderSize:] = np.nan

    minimumDepth = np.nanmin(depthImage)
    maximumDepth = np.nanmax(depthImage)
    zThreshold = minimumDepth + ((maximumDepth - minimumDepth) * ((self.thresholdBox.value)/100))

    points = vtk.vtkPoints()

    fx_d = self.focalLengthBox.value
    fy_d = self.focalLengthBox.value

    colorArray = None
    # if rgbImage.any():
    #   colorArray = vtk.util.numpy_support.numpy_to_vtk(rgbImage.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    #   colorArray.SetName("PointCloudColor")
    #   colorArray.SetNumberOfComponents(3)
    if rgbImage.any():
      colorArray = vtk.vtkUnsignedCharArray()
      colorArray.SetName("PointCloudColor")
      colorArray.SetNumberOfComponents(3)

    thresholdedColorArray = vtk.vtkUnsignedCharArray()
    for u in range(height):
      for v in range(width):
        vflipped = width - (v + 1)
        z = depthImage[u][vflipped]
        #z = depthImage[u][v]

        if (z is not np.nan) and (z < zThreshold):
          world_x = (z * (u - (height/2)) / fx_d) * self.imageScaleBox.value
          world_y = (z * (v - (width/2)) / fy_d) * self.imageScaleBox.value
          world_z = z * self.depthScaleBox.value
          points.InsertNextPoint(world_x, world_y, world_z)
          #points.InsertNextPoint(np.array([z * (u - (height/2)) / fx_d, z * (v - (width/2)) / fy_d, z / self.depthDividerBox.value]))
          if colorArray:
            colorArray.InsertNextTuple3(rgbImage[u, vflipped][0], rgbImage[u, vflipped][1], rgbImage[u, vflipped][2])


    # Create junk polygons so that Slicer can actually display the point cloud
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    poly = vtk.vtkPolygon()
    poly.GetPointIds().SetNumberOfIds(points.GetNumberOfPoints())
    for n in range(points.GetNumberOfPoints()):
      poly.GetPointIds().SetId(n, n)
    polys = vtk.vtkCellArray()
    polys.InsertNextCell(poly)
    polydata.SetPolys(polys)

    if colorArray:
      polydata.GetPointData().SetScalars(colorArray)

    # Display the point cloud
    modelNode = self.pointCloudSelector.currentNode()

    modelNode.SetAndObservePolyData(polydata)
    modelDisplayNode = modelNode.GetModelDisplayNode()
    if modelDisplayNode is None:
      modelDisplayNode = slicer.vtkMRMLModelDisplayNode()
      modelDisplayNode.SetScene(slicer.mrmlScene)
      slicer.mrmlScene.AddNode(modelDisplayNode)
      modelNode.SetAndObserveDisplayNodeID(modelDisplayNode.GetID())
      modelDisplayNode.SetRepresentation(modelDisplayNode.PointsRepresentation)
      modelDisplayNode.SetPointSize(4)
      modelDisplayNode.SetOpacity(1.0)
      modelDisplayNode.SetColor(1,0,0)

    # # Temp
    # modelNodePerm = slicer.vtkMRMLModelNode() # Create new model node; temporary for showing all creations
    # slicer.mrmlScene.AddNode(modelNodePerm)
    # transformNode = self.inputTransformSelector.currentNode()
    # modelNodePerm.SetAndObserveTransformNodeID(transformNode.GetID())
    # slicer.vtkSlicerTransformLogic().hardenTransform(modelNodePerm)
    # modelPermDisplayNode = slicer.vtkMRMLModelDisplayNode()
    # modelPermDisplayNode.SetScene(slicer.mrmlScene)
    # slicer.mrmlScene.AddNode(modelPermDisplayNode)
    # modelNode.SetAndObserveDisplayNodeID(modelPermDisplayNode.GetID())
    # modelPermDisplayNode.SetRepresentation(modelPermDisplayNode.PointsRepresentation)
    # modelPermDisplayNode.SetPointSize(4)
    # modelPermDisplayNode.SetOpacity(1.0)
    # modelPermDisplayNode.SetColor(1,0,0)
    # modelPermDisplayNode.SetScalarVisibility(True)
    # modelPermDisplayNode.SetActiveScalarName("PointCloudColor")
    # modelPermDisplayNode.SetScalarRangeFlag(4)
    # # temp